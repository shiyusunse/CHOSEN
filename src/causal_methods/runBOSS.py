from causallearn.search.PermutationBased.BOSS import boss
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import glob
import warnings

INPUT_DIR = '../predictions_DriverAI_Complete'
OUTPUT_DIR = '../multi_causal_results/BOSS_prediction/graph_all_DriverAI'
MAX_VARS = 30
MIN_SAMPLES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

processed = 0
skipped = 0
failed = 0

for csv_path in csv_files:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    try:
        data = pd.read_csv(csv_path)
        data = data.select_dtypes(include=['number']).dropna()
        X = data.values.astype(float)
        columns = data.columns.tolist()

        if X.shape[0] < MIN_SAMPLES:
            skipped += 1
            continue

        var_thresh = VarianceThreshold(threshold=1e-10)
        try:
            X_high_var = var_thresh.fit_transform(X)
        except ValueError:
            skipped += 1
            continue

        selected_mask = var_thresh.get_support()
        if X_high_var.shape[1] == 0:
            skipped += 1
            continue

        data_high_var = data.loc[:, selected_mask]
        columns_high_var = data_high_var.columns.tolist()

        data_vals = data_high_var.values
        _, unique_idx = np.unique(data_vals, axis=1, return_index=True)
        unique_idx = np.sort(unique_idx)
        X_clean = data_vals[:, unique_idx]
        final_columns = [columns_high_var[i] for i in unique_idx]

        if X_clean.shape[1] > MAX_VARS:
            skipped += 1
            continue

        if X_clean.shape[0] <= X_clean.shape[1]:
            skipped += 1
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            lambda_value = max(2, X_scaled.shape[1] // 10)
            parameters = {'lambda_value': lambda_value}

            G = boss(X_scaled, parameters=parameters)

        directed_edges = []
        nodes = G.get_nodes()

        if len(nodes) != len(final_columns):
            continue

        node_name_map = {}
        for i, node in enumerate(nodes):
            node_name_map[node.get_name()] = final_columns[i]

        edges = G.get_graph_edges()
        for edge in edges:
            node1 = edge.get_node1()
            node2 = edge.get_node2()

            endpoint1 = edge.get_endpoint1().name
            endpoint2 = edge.get_endpoint2().name

            node1_name = node1.get_name()
            node2_name = node2.get_name()

            if endpoint1 == "TAIL" and endpoint2 == "ARROW":
                # node1 -> node2
                parent_name = node_name_map.get(node1_name, node1_name)
                child_name = node_name_map.get(node2_name, node2_name)
                directed_edges.append(f"{parent_name} -> {child_name}")
            elif endpoint1 == "ARROW" and endpoint2 == "TAIL":
                # node2 -> node1
                parent_name = node_name_map.get(node2_name, node2_name)
                child_name = node_name_map.get(node1_name, node1_name)
                directed_edges.append(f"{parent_name} -> {child_name}")
            else:
                print(f"  ⚠️  skip: {node1_name} - {node2_name} ({endpoint1}-{endpoint2})")

        total_edges = len(directed_edges)
        output_lines = [str(total_edges)]
        output_lines.extend(directed_edges)

        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        processed += 1

    except np.linalg.LinAlgError as e:
        failed += 1
        continue
    except Exception as e:
        failed += 1
        continue
