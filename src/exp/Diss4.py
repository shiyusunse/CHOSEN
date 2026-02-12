from __future__ import print_function, division
import os
import glob
import numpy as np
import pandas as pd


def calculate_apfd_c(safety_series, cost_series):
    safety_series = np.array(safety_series)
    cost_series = np.array(cost_series)
    cumulative_cost = np.cumsum(cost_series)
    cumulative_faults = np.cumsum(safety_series == 1)
    area = np.trapz(cumulative_faults, cumulative_cost)
    max_cost = np.sum(cost_series)
    max_faults = np.sum(safety_series == 1)
    max_area = max_cost * max_faults if max_faults > 0 else 0
    return area / max_area if max_area > 0 else 0


def calculate_recall_effort_at_20(sorted_safety, sorted_duration):
    safety = np.array(sorted_safety)
    duration = np.array(sorted_duration)

    total_time = duration.sum()
    total_faults = (safety == 1).sum()

    if total_faults == 0 or total_time == 0:
        return 0.0, 0.0

    # Recall@20: bugs found within 20% of total time
    time_20 = 0.2 * total_time
    cumulative_time = np.cumsum(duration)
    idx_20 = np.searchsorted(cumulative_time, time_20)
    found_faults_20 = (safety[:idx_20 + 1] == 1).sum()
    recall20 = found_faults_20 / total_faults

    # Effort@20: time needed to find 20% of all bugs
    target_faults_20 = 0.2 * total_faults
    cumulative_faults = np.cumsum(safety == 1)
    idx_faults_20 = np.searchsorted(cumulative_faults, target_faults_20)
    time_to_find_20faults = cumulative_time[min(idx_faults_20, len(cumulative_time) - 1)]
    effort20 = time_to_find_20faults / total_time

    return recall20, effort20


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, on_bad_lines='warn')
    df = df.drop(columns=['start_time', 'end_time'], errors='ignore')

    if 'safety' not in df.columns:
        raise ValueError(f"File {file_path} missing 'safety' column!")

    df['safety'] = df['safety'].map({'safe': 0, 'unsafe': 1})
    if df['safety'].isnull().any():
        raise ValueError(f"File {file_path} has invalid 'safety' labels")

    feature_columns = [col for col in df.columns if col != 'safety']
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

    if df[feature_columns].isnull().any().any():
        problematic_cols = df[feature_columns].columns[df[feature_columns].isnull().any()]
        raise ValueError(f"Non-numeric features in {file_path}: {problematic_cols.tolist()}")

    return df


def parse_csv_name(filename):
    if not filename.endswith('.csv'):
        return None, None
    name_without_ext = filename[:-4]
    if name_without_ext.endswith('_Complete'):
        variant = 'Complete'
        base = name_without_ext[:-9]
    elif name_without_ext.endswith('_selected'):
        variant = 'selected'
        base = name_without_ext[:-9]
    else:
        return None, None
    if not base or base.endswith('_'):
        return None, None
    return base, variant


def evaluate_random_baseline(tgt_data, n_runs=30, random_seed=None):
    np.random.seed(random_seed)

    apfd_c_list = []
    recall20_list = []
    effort20_list = []

    safety = tgt_data['safety'].values
    if 'duration_seconds' in tgt_data.columns:
        duration = tgt_data['duration_seconds'].values
    else:
        duration = np.ones_like(safety)

    for _ in range(n_runs):
        indices = np.arange(len(safety))
        np.random.shuffle(indices)

        sorted_safety = safety[indices]
        sorted_duration = duration[indices]

        apfd_c = calculate_apfd_c(sorted_safety, sorted_duration)
        recall20, effort20 = calculate_recall_effort_at_20(sorted_safety, sorted_duration)

        apfd_c_list.append(apfd_c)
        recall20_list.append(recall20)
        effort20_list.append(effort20)

    return {
        'APFD_C_Mean': float(np.mean(apfd_c_list)),
        'Recall20_Mean': float(np.mean(recall20_list)),
        'Effort20_Mean': float(np.mean(effort20_list)),
        'APFD_C_List': apfd_c_list,
        'Recall20_List': recall20_list,
        'Effort20_List': effort20_list
    }


def main():
    # Configuration
    DATA_DIR = "../ADS_testing/src/bellwether/data"
    OUTPUT_DIR = "./random_baseline_results"
    N_RUNS = 30
    RANDOM_SEED = 42  

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")

    projects = {}
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        base, variant = parse_csv_name(filename)
        if base is None:
            print(f"⚠️ Skipping unparsable file: {filename}")
            continue
        full_name = f"{base}_{variant}"
        try:
            projects[full_name] = load_and_preprocess(file_path)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue

    if not projects:
        raise RuntimeError("No valid projects loaded!")

    print(f"\n✅ Loaded {len(projects)} projects: {sorted(projects.keys())}")

    # Run random baseline for each target
    summary_records = []

    for target_name, tgt_df in sorted(projects.items()):
        print(f"\n🎲 Evaluating random baseline for: {target_name}")
        result = evaluate_random_baseline(tgt_df, n_runs=N_RUNS, random_seed=RANDOM_SEED)

        # Save per-run details
        detail_df = pd.DataFrame({
            'Run': list(range(1, N_RUNS + 1)),
            'APFD_C': result['APFD_C_List'],
            'Recall20': result['Recall20_List'],
            'Effort20': result['Effort20_List']
        })
        detail_file = os.path.join(OUTPUT_DIR, f"{target_name}_random_runs.csv")
        detail_df.to_csv(detail_file, index=False)

        # Record summary
        summary_records.append({
            'Target_Project': target_name,
            'Method': 'Random',
            'Final_APFD_C_Mean': result['APFD_C_Mean'],
            'Final_Recall20_Mean': result['Recall20_Mean'],
            'Final_Effort20_Mean': result['Effort20_Mean']
        })

    # Save overall summary
    summary_df = pd.DataFrame(summary_records)
    summary_file = os.path.join(OUTPUT_DIR, "random_baseline_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    print(f"\n🎉 Random baseline evaluation completed!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("\nSummary (mean across 30 runs):")
    print(summary_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
    