from __future__ import print_function, division
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import os
import glob
from pathlib import Path


def get_models(random_state):
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=random_state),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state)
    }


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, on_bad_lines='warn')
    # Drop timestamp columns if present
    df = df.drop(columns=['start_time', 'end_time'], errors='ignore')

    if 'safety' not in df.columns:
        raise ValueError(f"File {file_path} lacks 'safety' column!")

    # Map safety to 0/1
    df['safety'] = df['safety'].map({'safe': 0, 'unsafe': 1})
    if df['safety'].isnull().any():
        raise ValueError(f"File {file_path} contains unmapped 'safety' labels")

    # Keep all numeric features (do not exclude duration_seconds)
    feature_columns = [col for col in df.columns if col != 'safety']
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

    if df[feature_columns].isnull().any().any():
        problematic_cols = df[feature_columns].columns[df[feature_columns].isnull().any()]
        raise ValueError(f"File {file_path} contains non‑numeric data in feature columns. Problematic columns: {problematic_cols.tolist()}")

    return df


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

    # ---------- Recall@20 ----------
    time_20 = 0.2 * total_time
    cumulative_time = np.cumsum(duration)
    # searchsorted returns the first index where cumulative_time >= time_20
    idx_20 = np.searchsorted(cumulative_time, time_20)
    # If time_20 exactly equals cumulative_time[k], searchsorted returns k.
    # We want to include the test case at idx_20 because its execution time is already consumed.
    found_faults_20 = (safety[:idx_20 + 1] == 1).sum() if idx_20 < len(safety) else (safety == 1).sum()
    recall20 = found_faults_20 / total_faults

    # ---------- Effort@20 ----------
    # Target: discover 20% of total_faults
    target_faults_20 = max(1, int(np.ceil(0.2 * total_faults)))  # at least 1 fault (if any)
    cumulative_faults = np.cumsum(safety == 1)
    # Find first position where cumulative_faults >= target_faults_20
    idx_faults_20 = np.searchsorted(cumulative_faults, target_faults_20)
    if idx_faults_20 >= len(cumulative_time):
        time_to_find_20faults = cumulative_time[-1]
    else:
        time_to_find_20faults = cumulative_time[idx_faults_20]
    effort20 = time_to_find_20faults / total_time

    return recall20, effort20


def evaluate_transfer_causal(src_data, tgt_data, model):
    try:
        # Feature selection (exclude safety)
        feature_cols = [col for col in src_data.columns if col != 'safety']

        X_src = src_data[feature_cols]
        y_src = src_data['safety']
        X_tgt = tgt_data[feature_cols]
        y_tgt = tgt_data['safety']

        # Take intersection (ensure consistent features)
        common_features = X_src.columns.intersection(X_tgt.columns)
        if len(common_features) == 0:
            return 0.0, 0.0, 0.0

        X_src = X_src[common_features]
        X_tgt = X_tgt[common_features]

        # Standardize
        scaler = StandardScaler()
        X_src_scaled = scaler.fit_transform(X_src)
        X_tgt_scaled = scaler.transform(X_tgt)

        # Train
        model.fit(X_src_scaled, y_src)

        # Predict probability (or normalize decision_function to 0..1)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_tgt_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_tgt_scaled)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

        epsilon = 1e-8
        if "duration_seconds" in tgt_data.columns:
            duration = tgt_data["duration_seconds"].replace(0, epsilon)
            score = y_prob / duration
        else:
            score = y_prob

        # Sort by score and compute metrics
        sorted_indices = np.argsort(-score)
        sorted_safety = y_tgt.values[sorted_indices]
        if "duration_seconds" in tgt_data.columns:
            sorted_duration = tgt_data["duration_seconds"].values[sorted_indices]
        else:
            sorted_duration = np.ones_like(sorted_safety)

        apfd_c = calculate_apfd_c(sorted_safety, sorted_duration)
        recall20, effort20 = calculate_recall_effort_at_20(sorted_safety, sorted_duration)

        return apfd_c, recall20, effort20

    except Exception as e:
        # Return 0.0 on error (and print log for troubleshooting)
        print(f"❌ evaluate_transfer_causal error: {e}")
        return 0.0, 0.0, 0.0



CAUSAL_ROOT_DIR = "../multi_causal_results"
GROUND_TRUTH_DIR = "../ground_truth_graph"

LEARNERS = [
    "LogisticRegression", "RandomForest", "SVM",
    "XGBoost", "NaiveBayes", "DecisionTree"
]

# TODO:'BOSS', 'CAM_UV', 'CDNOD', 'DiBS', 'DirectLiNGAM', 'FCI', 'GES', 'GRaSP', 'PC'
CAUSAL_MODEL = 'BOSS'


def normalize_edge(edge_str):
    if not isinstance(edge_str, str):
        return ""
    if ':' in edge_str:
        edge_str = edge_str.split(':', 1)[0]
    edge_str = edge_str.replace(' ', '').strip()

    if '->' not in edge_str:
        return edge_str

    parent, child = edge_str.split('->', 1)

    def normalize_node(node):
        node = node.strip()
        if node == 'prob':
            return 'safety'
        elif node == 'test_duration':
            return 'duration_seconds'
        else:
            return node

    return f"{normalize_node(parent)}->{normalize_node(child)}"


def load_graph_from_txt(file_path):
    if not os.path.exists(file_path):
        return set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        edge_lines = lines[1:] if lines and lines[0].isdigit() else lines
        edges = set()
        for line in edge_lines:
            if '->' in line:
                edge_str = normalize_edge(line)
                edges.add(edge_str)
        return edges
    except Exception:
        return set()


def jaccard_similarity(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


def get_causal_models():
    causal_dirs = glob.glob(os.path.join(CAUSAL_ROOT_DIR, "*_prediction"))
    models = []
    for d in causal_dirs:
        name = os.path.basename(d)
        if name.endswith('_prediction'):
            model_name = name[:-11]
            if model_name not in [CAUSAL_MODEL]:
                continue
            models.append(model_name)
    return sorted(set(models))


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


def discover_all_targets():
    gt_files = (
        glob.glob(os.path.join(GROUND_TRUTH_DIR, "*_Complete_graph.txt")) +
        glob.glob(os.path.join(GROUND_TRUTH_DIR, "*_selected_graph.txt"))
    )
    targets = []
    for f in gt_files:
        stem = Path(f).stem
        if not stem.endswith('_graph'):
            continue
        name_without_graph = stem[:-6]
        parts = name_without_graph.rsplit('_', 1)
        if len(parts) != 2:
            continue
        base, variant = parts
        if variant not in ['Complete', 'selected']:
            continue
        full_name = f"{base}_{variant}"
        targets.append(full_name)
    return sorted(set(targets))


def get_causal_model_ground_truth(causal_model_name, target_full):
    gt_dir = os.path.join(GROUND_TRUTH_DIR, f"{causal_model_name}_ground_truth_graph")
    candidates = [f"{target_full}_graph.txt", f"{target_full}.txt"]
    for filename in candidates:
        gt_file = os.path.join(gt_dir, filename)
        if os.path.exists(gt_file):
            return load_graph_from_txt(gt_file)
    return set()


def get_predicted_graph(target_full, source_full, learner, causal_models, base_variant_to_base):
    target_base = base_variant_to_base[target_full]
    source_base = base_variant_to_base[source_full]
    source_variant = 'Complete' if source_full.endswith('_Complete') else 'selected'

    for model in causal_models:
        pred_dir = os.path.join(CAUSAL_ROOT_DIR, f"{model}_prediction", f"graph_all_{target_base}")
        if not os.path.exists(pred_dir):
            continue
        for suffix in ["", "_graph"]:
            pred_file = os.path.join(pred_dir, f"{source_base}_{source_variant}_{learner}{suffix}.txt")
            if os.path.exists(pred_file):
                return load_graph_from_txt(pred_file)
    return set()


def gini_index(values):
    x = np.array(values)
    if np.mean(x) == 0:
        return 0.0
    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    gini = diff_sum / (2 * len(x)**2 * np.mean(x))
    return gini


def main():
    ALPHA = 0.5  
    N_FAST_RUNS = 5
    N_FINAL_RUNS = 30

    # Data directory (adjust as needed)
    data_dir = "../ADS_testing/src/bellwether/data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Step 1: Found {len(csv_files)} CSV files")

    projects = {}
    base_variant_to_base = {}

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        base, variant = parse_csv_name(filename)
        if base is None:
            print(f"⚠️ Skipping unparsable CSV: {filename}")
            continue
        full_name = f"{base}_{variant}"
        try:
            projects[full_name] = load_and_preprocess(file_path)
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}. Skipping this file.")
            continue
        base_variant_to_base[full_name] = base

    if not projects:
        raise ValueError("No valid CSV projects loaded!")

    valid_targets = sorted(projects.keys())
    print(f"\n✅ Valid CSV projects ({len(valid_targets)}): {valid_targets}")

    causal_models = get_causal_models()
    print(f"\n✅ Causal models: {causal_models}")

    output_dir = f"./hybrid_{ALPHA}_{CAUSAL_MODEL}_revised"
    os.makedirs(output_dir, exist_ok=True)

    print("\nStep 3: Computing Avg Jaccard (for each learner, averaged over all causal models)...")
    avg_jaccard_cache = {}
    jaccard_detail_records = []

    for tgt in valid_targets:
        for src in valid_targets:
            if base_variant_to_base[src] == base_variant_to_base[tgt]:
                continue
            for learner in LEARNERS:
                jacc_list = []
                for model in causal_models:
                    true_graph = get_causal_model_ground_truth(model, src)
                    pred_graph = get_predicted_graph(tgt, src, learner, [model], base_variant_to_base)
                    jacc = jaccard_similarity(pred_graph, true_graph)
                    jacc_list.append(jacc)
                    jaccard_detail_records.append({
                        'Source': src, 'Target': tgt, 'Learner': learner,
                        'Causal_Model': model, 'Jaccard': jacc,
                        'True_Graph_Size': len(true_graph),
                        'Pred_Graph_Size': len(pred_graph)
                    })
                avg_jacc = np.mean(jacc_list) if jacc_list else 0.0
                avg_jaccard_cache[(src, tgt, learner)] = avg_jacc

    jaccard_detail_df = pd.DataFrame(jaccard_detail_records)
    jaccard_detail_file = os.path.join(output_dir, "jaccard_detail.csv")
    jaccard_detail_df.to_csv(jaccard_detail_file, index=False)
    print(f"✅ Jaccard details saved: {jaccard_detail_file}")

    print("\nStep 3.1: Computing Jaccard mean and variance per (Source, Target, Learner) ...")

    jaccard_stats_records = []
    grouped = jaccard_detail_df.groupby(['Source', 'Target', 'Learner'])
    for (src, tgt, learner), subdf in grouped:
        mean_j = subdf['Jaccard'].mean()
        var_j = subdf['Jaccard'].var(ddof=1) if len(subdf) > 1 else 0.0
        jaccard_stats_records.append({
            'Source': src,
            'Target': tgt,
            'Learner': learner,
            'Mean_Jaccard': mean_j,
            'Var_Jaccard': var_j
        })
    jaccard_stats_df = pd.DataFrame(jaccard_stats_records)
    jaccard_stats_file = os.path.join(output_dir, "jaccard_stats_per_triplet.csv")
    jaccard_stats_df.to_csv(jaccard_stats_file, index=False)

    target_jacc_records = []
    grouped_tgt = jaccard_detail_df.groupby('Target')
    for tgt, subdf in grouped_tgt:
        mean_tgt = subdf['Jaccard'].mean()
        var_tgt = subdf['Jaccard'].var(ddof=1) if len(subdf) > 1 else 0.0
        target_jacc_records.append({
            'Target': tgt,
            'Mean_Jaccard_All': mean_tgt,
            'Var_Jaccard_All': var_tgt,
            'Num_Records': len(subdf)
        })
    target_jacc_df = pd.DataFrame(target_jacc_records)
    target_jacc_file = os.path.join(output_dir, "jaccard_stats_per_target.csv")
    target_jacc_df.to_csv(target_jacc_file, index=False)
    print(f"✅ Jaccard mean and variance per Target saved: {target_jacc_file}")


    bellwether_score_cache = {}
    bellwether_cache_dir = "./Bellwether_results"
    bellwether_cache_file = os.path.join(bellwether_cache_dir, "bellwether_cache.csv")

    if ALPHA == 0:
        print("\nSkipping Bellwether Score pre‑computation (ALPHA = 0)")
    else:
        os.makedirs(bellwether_cache_dir, exist_ok=True)

        if os.path.exists(bellwether_cache_file):
            cache_df = pd.read_csv(bellwether_cache_file)
            for _, row in cache_df.iterrows():
                key = (row['Source'], row['Target'], row['Learner'])
                bellwether_score_cache[key] = float(row['Bellwether_Score'])
            print(f"\n✅ Loaded Bellwether cache: {bellwether_cache_file} ({len(bellwether_score_cache)} entries)")
        else:
            print(f"\n⚠️ Bellwether cache not found, will create new file: {bellwether_cache_file}")

        # Determine all needed (src, tgt, learner) combinations
        all_needed = []
        for tgt in valid_targets:
            for src in valid_targets:
                if base_variant_to_base[src] == base_variant_to_base[tgt]:
                    continue
                for learner in LEARNERS:
                    all_needed.append((src, tgt, learner))

        missing = [key for key in all_needed if key not in bellwether_score_cache]

        if missing:
            print(f"\nStep 4: Need to compute {len(missing)} missing Bellwether Scores...")
            new_records = []
            for i, (src, tgt, learner) in enumerate(missing, 1):
                validation_projects = [p for p in valid_targets if p != src and p != tgt]
                if not validation_projects:
                    score = 0.0
                else:
                    val_scores = []
                    for val_proj in validation_projects:
                        apfd_vals = []
                        for _ in range(N_FAST_RUNS):
                            rs = np.random.randint(0, 100000)
                            model_inst = get_models(rs)[learner]
                            # causal_feats = source_causal_features.get(src, None)
                            apfd_c, _, _ = evaluate_transfer_causal(projects[src], projects[val_proj], model_inst)
                            apfd_vals.append(apfd_c)
                        val_scores.append(np.mean(apfd_vals))
                    score = np.mean(val_scores) if val_scores else 0.0

                bellwether_score_cache[(src, tgt, learner)] = score
                new_records.append({
                    'Source': src, 'Target': tgt, 'Learner': learner, 'Bellwether_Score': score
                })
                if i % 20 == 0 or i == len(missing):
                    print(f"   Computed {i}/{len(missing)}")

            # Append new records to CSV (avoid duplicates)
            new_df = pd.DataFrame(new_records)
            if os.path.exists(bellwether_cache_file):
                old_df = pd.read_csv(bellwether_cache_file)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['Source', 'Target', 'Learner'], keep='last', inplace=True)
            else:
                combined_df = new_df

            combined_df.to_csv(bellwether_cache_file, index=False)
            print(f"✅ Bellwether cache updated: {bellwether_cache_file}")
        else:
            print("\n✅ All Bellwether Scores are cached, no need to recompute")

    # === Hybrid selection and final evaluation ===
    print("\nStep 5: Hybrid selection (variance filtering + Bellwether priority)...")
    summary_records = []

    # Load overall Jaccard variance per Target from the statistics file
    target_var_map = dict(zip(target_jacc_df['Target'], target_jacc_df['Var_Jaccard_All']))
    VAR_THRESHOLD = 0.02  # adjustable parameter

    for target_full in valid_targets:
        print(f"\n🎯 Target: {target_full}")

        target_var = target_var_map.get(target_full, 0.0)
        print(f"  ℹ️ {target_full} overall Jaccard variance = {target_var:.4f}")

        # Collect candidate combinations
        candidate_records = []
        for source_full in valid_targets:
            if base_variant_to_base[source_full] == base_variant_to_base[target_full]:
                continue
            for learner in LEARNERS:
                jacc = avg_jaccard_cache.get((source_full, target_full, learner), 0.0)
                bellwether_score = bellwether_score_cache.get((source_full, target_full, learner), 0.0)
                candidate_records.append({
                    'Source': source_full,
                    'Learner': learner,
                    'Jaccard': jacc,
                    'Bellwether_Score': bellwether_score
                })

        if not candidate_records:
            print(f"  ⚠️ No valid candidates, skipping {target_full}")
            continue

        # === Check if variance is too large ===
        if target_var > VAR_THRESHOLD:
            print(f"  ⚠️ Variance {target_var:.4f} > {VAR_THRESHOLD}, considered unstable – using only Bellwether score for selection.")
            best_record = max(candidate_records, key=lambda r: r['Bellwether_Score'])
            method_used = "Bellwether_only"
            best_avg_jacc = best_record['Jaccard']
            best_bellwether_score = best_record['Bellwether_Score']
            best_hybrid_score = best_bellwether_score  # no mixing
        else:
            # === Hybrid normal flow (Bellwether + AvgJaccard weighted selection) ===
            print(f"  ℹ️ {target_full} entering Hybrid mode (Bellwether + AvgJaccard)")

            best_hybrid_score = -1
            best_record = None

            for rec in candidate_records:
                jacc = rec['Jaccard']
                bell = rec['Bellwether_Score']
                hybrid = ALPHA * bell + (1 - ALPHA) * jacc
                if hybrid > best_hybrid_score:
                    best_hybrid_score = hybrid
                    best_record = rec

            if best_record is None:
                print(f"  ⚠️ No effective combination found, skipping {target_full}")
                continue

            method_used = "Hybrid"
            best_avg_jacc = best_record['Jaccard']
            best_bellwether_score = best_record['Bellwether_Score']

        best_source_full = best_record['Source']
        best_learner = best_record['Learner']

        print(f"  ✅ Selected: Source={best_source_full}, Learner={best_learner}, "
              f"Bellwether={best_bellwether_score:.4f}, Jaccard={best_avg_jacc:.4f}, Method={method_used}")

        # === Final evaluation (N_FINAL_RUNS runs) ===
        apfd_c_list = []
        recall20_list = []
        effort20_list = []

        for run in range(N_FINAL_RUNS):
            rs = np.random.randint(0, 100000)
            model_inst = get_models(rs)[best_learner]
            # causal_feats = source_causal_features.get(best_source_full, None)
            apfd_c, recall20, effort20 = evaluate_transfer_causal(projects[best_source_full], projects[target_full], model_inst)
            apfd_c_list.append(apfd_c)
            recall20_list.append(recall20)
            effort20_list.append(effort20)

        detail_df = pd.DataFrame({
            'Run': list(range(1, N_FINAL_RUNS + 1)),
            'APFD_C': apfd_c_list,
            'Recall20': recall20_list,
            'Effort20': effort20_list,
            'Bellwether_Project': [best_source_full] * N_FINAL_RUNS,
            'Learner': [best_learner] * N_FINAL_RUNS,
            'Hybrid_Score': [best_hybrid_score] * N_FINAL_RUNS,
            'Avg_Jaccard': [best_avg_jacc] * N_FINAL_RUNS,
            'Bellwether_Score': [best_bellwether_score] * N_FINAL_RUNS,
            'Target_Project': [target_full] * N_FINAL_RUNS,
            'Method': [method_used] * N_FINAL_RUNS
        })
        detail_file = os.path.join(output_dir, f"{target_full}_hybrid_runs.csv")
        detail_df.to_csv(detail_file, index=False)

        summary_records.append({
            'Source_Project': best_source_full,
            'Learner': best_learner,
            'Target_Project': target_full,
            'Hybrid_Score': best_hybrid_score,
            'Avg_Jaccard': best_avg_jacc,
            'Bellwether_Score': best_bellwether_score,
            'Final_APFD_C_Mean': np.mean(apfd_c_list),
            'Final_APFD_C_Std': np.std(apfd_c_list),
            'Final_Recall20_Mean': np.mean(recall20_list),
            'Final_Recall20_Std': np.std(recall20_list),
            'Final_Effort20_Mean': np.mean(effort20_list),
            'Final_Effort20_Std': np.std(effort20_list),
            'Var_Jaccard_All': target_var,
            'Method': method_used
        })

    # === Summary output ===
    if summary_records:
        summary_df = pd.DataFrame(summary_records).sort_values(
            ['Target_Project', 'Hybrid_Score'], ascending=[True, False]
        )
        summary_file = os.path.join(output_dir, "hybrid_selection_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✅ Summary file saved: {summary_file}")
    else:
        print("\n❌ No valid results generated")

    print("\n🎉 Hybrid Bellwether analysis completed!")


if __name__ == "__main__":
    main()