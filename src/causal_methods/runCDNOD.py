from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci
from causallearn.utils.cit import fisherz
import pandas as pd
import numpy as np
import os
import glob

INPUT_DIR = '../selected_csv_dataset/'
OUTPUT_DIR = '../ground_truth_graph/CDNOD_ground_truth_graph/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

if not csv_files:
    exit()


for csv_path in csv_files:
    try:

        data = pd.read_csv(csv_path)

        if data.empty:
            continue

        numeric_data = data.select_dtypes(include=[np.number])
        non_numeric_cols = data.columns.difference(numeric_data.columns)
        if len(non_numeric_cols) > 0:
            print(f"⚠️: {list(non_numeric_cols)}")

        if numeric_data.empty:
            continue

        stds = numeric_data.std(ddof=0)
        non_const_mask = stds > 1e-8
        filtered_data = numeric_data.loc[:, non_const_mask]

        if filtered_data.empty:
            print(f"⚠️ skip: {csv_path}")
            continue

        feature_names = filtered_data.columns.tolist()
        X = filtered_data.values.astype(np.float64)
        n_samples, n_vars = X.shape

        c_indx = np.arange(n_samples).reshape(-1, 1)

        cg = cdnod(
            data=X,
            c_indx=c_indx,
            alpha=0.05,
            indep_test=fisherz,
            stable=True,
            uc_rule=0,
            uc_priority=-1
        )

        G = cg.G 
        graph_matrix = G.graph  # shape: (n_vars + 1, n_vars + 1)

        directed_edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                    directed_edges.append((i, j))

        output_lines = [str(len(directed_edges))]
        for (i, j) in directed_edges:
            parent = feature_names[i]
            child = feature_names[j]
            output_lines.append(f"{parent} -> {child}")

        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
    except Exception as e:
        continue
