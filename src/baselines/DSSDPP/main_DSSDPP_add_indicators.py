import os
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.optimize import linprog
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def get_data(df):
    """Extract features, labels, and cost"""
    feat_cols = [
        'direct_distance', 'road_distance', 'num_l_turns', 'num_r_turns', 'num_straights',
        'median_angle', 'total_angle', 'mean_angle', 'std_angle', 'max_angle', 'min_angle',
        'median_pivot_off', 'mean_pivot_off', 'std_pivot_off', 'max_pivot_off', 'min_pivot_off',
        'duration_seconds'
    ]
    X = df[feat_cols].values
    Y = df['safety'].map({'unsafe': 1, 'safe': 0}).astype(int).values
    cost_series = df['duration_seconds'].values
    safety_original = df['safety'].values
    return X, Y, cost_series, safety_original, feat_cols


def mpos_sampling(X, Y, random_state=42):
    """
    MPOS oversampling
    """
    np.random.seed(random_state)
    minority_mask = Y == 1
    X_minority = X[minority_mask]
    X_majority = X[~minority_mask]
    Y_minority = Y[minority_mask]
    Y_majority = Y[~minority_mask]

    n_minor = len(X_minority)
    n_major = len(X_majority)
    if n_minor >= n_major:
        return X, Y

    n_new = n_major - n_minor
    new_minority = []
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(X_minority)

    for _ in range(n_new):
        x_idx = np.random.randint(0, n_minor)
        x = X_minority[x_idx]
        _, neighbors = nn.kneighbors(x.reshape(1, -1))
        x_near = X_minority[neighbors[0][1]]
        r = np.random.rand()
        x_new = x + r * (x_near - x)
        new_minority.append(x_new)

    X_balanced = np.vstack([X_majority, X_minority, np.array(new_minority)])
    Y_balanced = np.hstack([Y_majority, Y_minority, np.ones(n_new, dtype=int)])
    return X_balanced, Y_balanced


def cfs_feature_selection(X, y):
    """
    CFS feature selection
    """
    n_features = X.shape[1]
    if n_features == 0:
        return np.array([])

    from scipy.stats import pearsonr
    feat_label_corr = []
    for f in range(n_features):
        try:
            corr, _ = pearsonr(X[:, f], y)
            feat_label_corr.append(abs(corr))
        except:
            feat_label_corr.append(0.0)
    feat_label_corr = np.array(feat_label_corr)

    feat_corr = np.corrcoef(X, rowvar=False)
    feat_corr = np.abs(feat_corr)
    avg_feat_corr = np.mean(feat_corr, axis=1)

    cfs_scores = feat_label_corr / (avg_feat_corr + 1e-8)
    fea_id = np.where(cfs_scores > 0)[0]
    return fea_id


def dssdpp_predict(Xss, Ys, Xt, lp='linear'):
    """
    Fully reproduce DSSDPP.m + DPP.m
    """
    unique_classes = np.unique(Ys)
    C = len(unique_classes)
    nt = Xt.shape[0]
    intcon = C * nt

    Ys_adj = Ys + 1 if 0 in unique_classes else Ys.copy()
    class_labels = np.sort(np.unique(Ys_adj))

    S = np.nanstd(Xt, axis=0)
    S[S == 0] = 1.0
    Dct = np.zeros((nt, C))
    for idx, c in enumerate(class_labels):
        X_c = Xss[Ys_adj == c]
        if len(X_c) == 0:
            raise ValueError(f"Class {c} has no samples")
        center = np.mean(X_c, axis=0)
        diff = (Xt - center) / S
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        Dct[:, idx] = dist

    CC = Dct.T.flatten()

    Aeq = np.zeros((nt, intcon))
    Beq = np.ones(nt)
    for i in range(nt):
        Aeq[i, i * C:(i + 1) * C] = 1.0

    A = np.zeros((C, intcon))
    B = np.full(C, -1.0)
    for c in range(C):
        col_indices = [c + j * C for j in range(nt)]
        A[c, col_indices] = -1.0

    bounds = [(0.0, 1.0)] * intcon

    if lp == 'binary':
        raise NotImplementedError("Binary mode is not supported yet")
    else:
        result = linprog(c=CC, A_ub=A, b_ub=B, A_eq=Aeq, b_eq=Beq,
                         bounds=bounds, method='highs', options={'disp': False})

    if not result.success:
        M = np.ones((nt, C)) / C
    else:
        M_vec = result.x[:intcon]
        M = M_vec.reshape(C, nt).T  # (nt, C)

    class_idx = 1 if C >= 2 else 0
    if 1 in unique_classes:
        class_idx_arr = np.where(class_labels == 2)[0]
        if len(class_idx_arr) > 0:
            class_idx = class_idx_arr[0]
    score = M[:, class_idx]
    return score


def calculate_apfd_c(safety_series, cost_series, pred_scores):
    """APFD_c: calculated by sorting descending by score/cost"""
    df = pd.DataFrame({'safety': safety_series, 'cost': cost_series, 'score': pred_scores})
    if df['safety'].dtype == object:
        df['safety_binary'] = df['safety'].map({'safe': 0, 'unsafe': 1})
    else:
        df['safety_binary'] = df['safety'].astype(int)
    df['ratio'] = df['score'] / (df['cost'] + 1e-8)
    df = df.sort_values('ratio', ascending=False).reset_index(drop=True)

    cum_cost = np.cumsum(df['cost'].values)
    cum_faults = np.cumsum(df['safety_binary'].values)
    area = np.trapz(cum_faults, cum_cost)

    max_cost = cum_cost[-1] if len(cum_cost) > 0 else 0
    max_faults = df['safety_binary'].sum()
    max_area = max_cost * max_faults if max_faults > 0 else 1e-8

    return area / max_area if max_area > 0 else 0.0


def calculate_recall_effort_at_20(sorted_safety_binary, sorted_duration):
    """
    sorted_safety_binary: np.array of 0/1 (1 = unsafe)
    sorted_duration: np.array of execution time
    """
    safety = np.array(sorted_safety_binary)
    duration = np.array(sorted_duration)

    total_time = duration.sum()
    total_faults = (safety == 1).sum()

    if total_faults == 0 or total_time == 0:
        return 0.0, 0.0

    time_20 = 0.2 * total_time
    cumulative_time = np.cumsum(duration)
    idx_20 = np.searchsorted(cumulative_time, time_20)
    found_faults_20 = (safety[:idx_20 + 1] == 1).sum()
    recall20 = found_faults_20 / total_faults

    target_faults_20 = 0.2 * total_faults
    cumulative_faults = np.cumsum(safety == 1)
    idx_faults_20 = np.searchsorted(cumulative_faults, target_faults_20)
    time_to_find_20faults = cumulative_time[min(idx_faults_20, len(cumulative_time) - 1)]
    effort20 = time_to_find_20faults / total_time

    return recall20, effort20


def corr_source_selection(target_df, source_dfs):
    """CORR source selection: quantile-based Spearman correlation"""
    feat_cols = [
        'direct_distance', 'road_distance', 'num_l_turns', 'num_r_turns', 'num_straights',
        'median_angle', 'total_angle', 'mean_angle', 'std_angle', 'max_angle', 'min_angle',
        'median_pivot_off', 'mean_pivot_off', 'std_pivot_off', 'max_pivot_off', 'min_pivot_off',
        'duration_seconds'
    ]
    target_feat = target_df[feat_cols]
    corr_scores = []

    n_quantiles = 100

    for src_df in source_dfs:
        src_feat = src_df[feat_cols]
        rho_list = []
        for col in feat_cols:
            a = target_feat[col].dropna().values
            b = src_feat[col].dropna().values

            if len(a) == 0 or len(b) == 0:
                rho_list.append(0.0)
                continue

            q = np.linspace(0, 1, n_quantiles)
            try:
                a_q = np.nanquantile(a, q)
                b_q = np.nanquantile(b, q)
                rho, _ = spearmanr(a_q, b_q)
                rho_list.append(abs(rho))
            except Exception:
                rho_list.append(0.0)

        corr_scores.append(np.mean(rho_list))

    return np.argmax(corr_scores)


if __name__ == "__main__":
    # Parameter settings
    data_dir = "./data"
    output_dir = "./output"
    test_ratio = 0.9
    repeat_times = 30
    random_seed = 42
    method_name = "DSSDPP"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "APFD_c_Results"), exist_ok=True)

    # New: directory for sorted case logs
    sorted_cases_dir = os.path.join(output_dir, "sorted_cases")
    os.makedirs(sorted_cases_dir, exist_ok=True)

    # Load data
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    projects = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(data_dir, f))
        name = f.replace(".csv", "")
        projects.append((df, name))

    total_projects = len(projects)

    # Precompute CORR source indices
    corr_source_indices = []
    for target_idx in range(total_projects):
        target_df = projects[target_idx][0]
        candidate_sources = [projects[j][0] for j in range(total_projects) if j != target_idx]
        best_idx = corr_source_selection(target_df, candidate_sources)
        global_idx = [j for j in range(total_projects) if j != target_idx][best_idx]
        corr_source_indices.append(global_idx)
    corr_source_indices = np.array(corr_source_indices).reshape(-1, 1)

    # Main experiment loop
    running_time = np.zeros((total_projects, repeat_times))
    all_results = []
    expRESULT = []

    for target_idx in range(total_projects):
        target_df, target_name = projects[target_idx]
        X_full, Y_full, cost_full, safety_orig, feat_cols = get_data(target_df)
        src_idx = corr_source_indices[target_idx][0]
        src_df, src_name = projects[src_idx]
        Xs_raw, Ys_raw, _, _, _ = get_data(src_df)

        print(f"Processing target: {target_name} | source: {src_name}")

        # Source data preprocessing
        Xs_bal, Ys_bal = mpos_sampling(Xs_raw, Ys_raw, random_seed)
        fea_id = cfs_feature_selection(Xs_bal, Ys_bal)
        if len(fea_id) == 0:
            fea_id = np.arange(Xs_bal.shape[1])
        Xs_filtered = Xs_bal[:, fea_id]
        scaler = StandardScaler()
        Xs_norm = scaler.fit_transform(Xs_filtered)

        # Repeated experiments
        results_this_target = []
        for rep in range(repeat_times):
            np.random.seed(random_seed + rep)
            idx = np.random.permutation(len(X_full))
            n_test = int(len(X_full) * test_ratio)
            test_idx = idx[-n_test:]

            Xt_test = X_full[test_idx][:, fea_id]
            Xt_norm = scaler.transform(Xt_test)
            Yt_test = Y_full[test_idx]
            cost_test = cost_full[test_idx]
            safety_test = safety_orig[test_idx]
            original_indices = test_idx  # for manual review: original row numbers

            t_start = datetime.now()
            pred_scores = dssdpp_predict(Xs_norm, Ys_bal, Xt_norm, lp='linear')
            t_elapsed = (datetime.now() - t_start).total_seconds()
            running_time[target_idx, rep] = t_elapsed

            # =============================
            # Intelligent sorting logic (core modification point)
            # =============================
            epsilon = 1e-8
            pred_scores = np.asarray(pred_scores)
            cost_test = np.asarray(cost_test)

            # Separate positive-score (>0) and zero-score (==0) cases
            positive_mask = pred_scores > 0
            zero_mask = ~positive_mask

            # Positive-score cases: sort by score / duration descending
            if np.any(positive_mask):
                pos_indices = np.where(positive_mask)[0]
                pos_ratios = pred_scores[pos_indices] / (cost_test[pos_indices] + epsilon)
                sorted_pos = pos_indices[np.argsort(-pos_ratios, kind='stable')]
            else:
                sorted_pos = np.array([], dtype=int)

            # Zero-score cases: sort by duration ascending (shortest first)
            if np.any(zero_mask):
                zero_indices = np.where(zero_mask)[0]
                sorted_zero = zero_indices[np.argsort(cost_test[zero_indices], kind='stable')]
            else:
                sorted_zero = np.array([], dtype=int)

            # Merge: high-value cases first, then low-cost cases
            sorted_order = np.concatenate([sorted_pos, sorted_zero])

            # Extract sorted data
            sorted_safety_binary = Yt_test[sorted_order]
            sorted_duration = cost_test[sorted_order]
            sorted_scores = pred_scores[sorted_order]
            sorted_original_idx = original_indices[sorted_order]
            sorted_safety_str = safety_test[sorted_order]

            # Scores for logging (set to NaN for zero scores)
            score_ratio_for_log = np.where(
                pred_scores > 0,
                pred_scores / (cost_test + epsilon),
                np.nan
            )
            logged_ratios = score_ratio_for_log[sorted_order]

            # Calculate metrics
            apfd_c = calculate_apfd_c(safety_test, cost_test, pred_scores)
            auc = roc_auc_score(Yt_test, pred_scores)
            recall20, effort20 = calculate_recall_effort_at_20(sorted_safety_binary, sorted_duration)

            # Record results
            results_this_target.append({
                'target': target_name,
                'source': src_name,
                'repeat': rep + 1,
                'APFD_c': apfd_c,
                'AUC': auc,
                'Recall20': recall20,
                'Effort20': effort20,
                'time': t_elapsed
            })
            all_results.append(results_this_target[-1])

            # Save sorting details (for manual review)
            rank_df = pd.DataFrame({
                'Rank': np.arange(1, len(sorted_order) + 1),
                'Original_Index_In_Target_CSV': sorted_original_idx,
                'Predicted_Score': sorted_scores,
                'Duration_Seconds': sorted_duration,
                'Safety_Label': sorted_safety_str,
                'Is_Unsafe': sorted_safety_binary,
                'Score_Div_Duration': logged_ratios  # NaN for zero scores, clearly distinguishable
            })
            rank_file = os.path.join(sorted_cases_dir, f"{target_name}_run{rep + 1:02d}_sorted.csv")
            rank_df.to_csv(rank_file, index=False)

        # Save summary of metrics for 30 runs of this project
        df_rep = pd.DataFrame(results_this_target)
        df_rep.to_csv(os.path.join(output_dir, "APFD_c_Results", f"{target_name}_runs.csv"), index=False)
        expRESULT.append(results_this_target)

    # Save global results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, f"{method_name}_ALL_Results.csv"), index=False)

    # Save .mat file
    savemat(os.path.join(output_dir, f"{method_name}_EXPRESULT.mat"), {
        'expRESULT': expRESULT,
        'result': df_all.groupby('target')[['APFD_c', 'AUC', 'Recall20', 'Effort20']].mean().values.tolist(),
        'time': running_time,
        'meanstdT': f"{np.mean(np.sum(running_time, axis=0)):.3f}±{np.std(np.sum(running_time, axis=0)):.3f}"
    })

    print("✅ Experiment completed!")
    print(f"📁 Main results directory: {output_dir}")
    print(f"📁 Sorting details directory (for manual review): {sorted_cases_dir}")