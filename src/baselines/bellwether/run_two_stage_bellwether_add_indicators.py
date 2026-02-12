from __future__ import print_function, division
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
    df = df.drop(columns=['start_time', 'end_time'], errors='ignore')

    if 'duration_seconds' not in df.columns:
        raise ValueError(f"File {file_path} lacks 'duration_seconds' column, cannot calculate APFD_C!")

    df['safety'] = df['safety'].map({'safe': 0, 'unsafe': 1})
    if df['safety'].isnull().any():
        raise ValueError(f"File {file_path} contains unmapped 'safety' labels")

    feature_columns = [col for col in df.columns if col != 'safety']
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

    if df[feature_columns].isnull().any().any():
        problematic_cols = df[feature_columns].columns[df[feature_columns].isnull().any()]
        raise ValueError(f"file {file_path}: {problematic_cols.tolist()}")

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


# ---------------------------- Recall@20 & Effort@20 ----------------------------
def calculate_recall_effort_at_20(sorted_safety, sorted_duration):
    safety = np.array(sorted_safety)
    duration = np.array(sorted_duration)

    total_time = duration.sum()
    total_faults = (safety == 1).sum()

    if total_faults == 0 or total_time == 0:
        return 0.0, 0.0

    # ===== Recall@20 =====
    time_20 = 0.2 * total_time
    cumulative_time = np.cumsum(duration)
    idx_20 = np.searchsorted(cumulative_time, time_20)
    recall20 = (safety[:idx_20 + 1] == 1).sum() / total_faults

    # ===== Effort@20 =====
    target_faults_20 = max(1, int(np.ceil(0.2 * total_faults)))
    cumulative_faults = np.cumsum(safety == 1)
    idx_faults_20 = np.searchsorted(cumulative_faults, target_faults_20)
    time_to_find_20_faults = cumulative_time[min(idx_faults_20, len(cumulative_time) - 1)]
    effort20 = time_to_find_20_faults / total_time

    return recall20, effort20


def evaluate_transfer(src_data, tgt_data, model):
    try:
        feature_cols = [col for col in src_data.columns if col != 'safety']
        X_src = src_data[feature_cols]
        y_src = src_data['safety']
        X_tgt = tgt_data[feature_cols]
        y_tgt = tgt_data['safety']
        duration = tgt_data['duration_seconds'].values

        common_features = X_src.columns.intersection(X_tgt.columns)
        if len(common_features) == 0:
            return 0.0, 0.0, 0.0

        X_src = X_src[common_features]
        X_tgt = X_tgt[common_features]

        scaler = StandardScaler()
        X_src_scaled = scaler.fit_transform(X_src)
        X_tgt_scaled = scaler.transform(X_tgt)

        model.fit(X_src_scaled, y_src)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_tgt_scaled)[:, 1]
        else:
            prob = model.decision_function(X_tgt_scaled)
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)

        epsilon = 1e-8
        score = prob / (duration + epsilon)

        sorted_indices = np.argsort(-score)
        sorted_safety = y_tgt.values[sorted_indices]
        sorted_duration = duration[sorted_indices]

        apfd_c = calculate_apfd_c(sorted_safety, sorted_duration)
        recall20, effort20 = calculate_recall_effort_at_20(sorted_safety, sorted_duration)

        return apfd_c, recall20, effort20

    except Exception as e:
        return 0.0, 0.0, 0.0


if __name__ == "__main__":
    data_dir = "./data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if len(csv_files) != 7:
        raise ValueError("not 7 files")

    projects = {}
    for file_path in csv_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        projects[name] = load_and_preprocess(file_path)

    project_names = list(projects.keys())
    n_runs = 30
    model_names = list(get_models(0).keys())
    output_dir = "./twostage_bellwether_predictions_add_indicators_test"
    os.makedirs(output_dir, exist_ok=True)

    summary_records = []

    for target_name in project_names:
        print(f"\n🎯 Target Project: {target_name}")
        candidates = [p for p in project_names if p != target_name]

        best_bellwether = None
        best_model = None
        best_validation_score = -1

        for bellwether_candidate in candidates:
            print(f"  ➡️ Candidate Bellwether: {bellwether_candidate}")
            validation_projects = [p for p in candidates if p != bellwether_candidate]

            best_model_for_B = None
            best_val_score_for_B = -1

            for model_name in model_names:
                val_scores = []
                for val_proj in validation_projects:
                    apfd_list = []
                    for _ in range(n_runs):
                        rs = np.random.randint(0, 100000)
                        model = get_models(rs)[model_name]
                        apfd_c, _, _ = evaluate_transfer(
                            projects[bellwether_candidate],
                            projects[val_proj],
                            model
                        )
                        apfd_list.append(apfd_c)
                    avg_apfd = np.mean(apfd_list)
                    val_scores.append(avg_apfd)

                mean_val = np.mean(val_scores)
                if mean_val > best_val_score_for_B:
                    best_val_score_for_B = mean_val
                    best_model_for_B = model_name

            if best_val_score_for_B > best_validation_score:
                best_validation_score = best_val_score_for_B
                best_bellwether = bellwether_candidate
                best_model = best_model_for_B
                print(f"    ✅ new Bellwether: {bellwether_candidate} + {best_model} (APFD_C={best_val_score_for_B:.4f})")

        print(f"  📌 best model: {best_bellwether} + {best_model}, predict {target_name}")

        apfd_c_list = []
        recall20_list = []
        effort20_list = []

        for run in range(n_runs):
            rs = np.random.randint(0, 100000)
            model = get_models(rs)[best_model]
            apfd_c, recall20, effort20 = evaluate_transfer(
                projects[best_bellwether],
                projects[target_name],
                model
            )
            apfd_c_list.append(apfd_c)
            recall20_list.append(recall20)
            effort20_list.append(effort20)

        detail_df = pd.DataFrame({
            'Run': list(range(1, n_runs + 1)),
            'APFD_C': apfd_c_list,
            'Recall20': recall20_list,
            'Effort20': effort20_list,
            'Bellwether_Project': [best_bellwether] * n_runs,
            'Validation_Avg_APFD_C': [best_validation_score] * n_runs,
            'Model': [best_model] * n_runs,
            'Target_Project': [target_name] * n_runs
        })
        detail_file = os.path.join(output_dir, f"{target_name}_bellwether_runs.csv")
        detail_df.to_csv(detail_file, index=False)

        summary_records.append({
            'Source_Project': best_bellwether,
            'Target_Project': target_name,
            'Model': best_model,
            'Avg_Bellwether_Score_APFD_C': best_validation_score
        })

    summary_df = pd.DataFrame(summary_records).sort_values(
        ['Target_Project', 'Avg_Bellwether_Score_APFD_C'],
        ascending=[True, False]
    )
    summary_file = os.path.join(output_dir, "bellwether_avg_scores.csv")
    summary_df.to_csv(summary_file, index=False)
    