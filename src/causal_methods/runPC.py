from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
import os
import glob

INPUT_DIR = '../selected_csv_dataset/'
OUTPUT_DIR = '../ground_truth_graph/PC_ground_truth_graph/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

if not csv_files:
    exit()

for csv_path in csv_files:
    try:
        data = pd.read_csv(csv_path)
        feature_names = data.columns.tolist()
        X = data.values  

        cg = pc(X, alpha=0.05, indep_test='fisherz')
        G = cg.G

        directed_edges = []
        n_vars = len(feature_names)
        for i in range(n_vars):
            for j in range(n_vars):
                if G.graph[i, j] == -1 and G.graph[j, i] == 1:
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
