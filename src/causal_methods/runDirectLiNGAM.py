import sys
sys.path.append("")

import pandas as pd
import numpy as np
import os
import glob
from causallearn.search.FCMBased import lingam
from sklearn.preprocessing import StandardScaler

INPUT_DIR = '../selected_csv_dataset/'
OUTPUT_DIR = '../ground_truth_graph/DirectLiNGAM_ground_truth_graph/'
RANDOM_STATE = 100  
MEASURE = 'pwling'  


def run_direct_lingam_on_csv(csv_path, output_dir, random_state=RANDOM_STATE, measure=MEASURE):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    try:
        data = pd.read_csv(csv_path)
        data = data.select_dtypes(include=['number']).dropna()
        if data.empty:
            raise ValueError("invalid")

        X = data.values.astype(float)
        columns = data.columns.tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = lingam.DirectLiNGAM(random_state=random_state, measure=measure)
        model.fit(X_scaled)

        adj_matrix = model.adjacency_matrix_
        if adj_matrix is None:
            raise ValueError("empty")

        directed_edges = []
        n_vars = len(columns)
        for i in range(n_vars):
            for j in range(n_vars):
                if abs(adj_matrix[i, j]) > 1e-5:
                    directed_edges.append(f"{columns[i]} -> {columns[j]}")

        total_edges = len(directed_edges)
        output_lines = [str(total_edges)] + directed_edges

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        return True, total_edges

    except Exception as e:
        return False, 0

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        sys.exit(1)

    processed = 0
    failed = 0
    total_edges_all = 0

    for csv_path in csv_files:
        success, edge_count = run_direct_lingam_on_csv(
            csv_path,
            OUTPUT_DIR,
            random_state=RANDOM_STATE,
            measure=MEASURE
        )
        if success:
            processed += 1
            total_edges_all += edge_count
        else:
            failed += 1
