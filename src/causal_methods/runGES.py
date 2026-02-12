from causallearn.search.ScoreBased.GES import ges
import pandas as pd
import glob
import os

csv_dir = '../selected_csv_dataset/'  
output_dir = '../ground_truth_graph/GES_ground_truth_graph/'        

os.makedirs(output_dir, exist_ok=True)

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

if not csv_files:
    exit()


for csv_path in csv_files:
    try:
        X = pd.read_csv(csv_path)
        feature_names = X.columns.tolist()

        Record = ges(X, score_func='local_score_BIC', maxP=None, parameters=None)
        G = Record['G']

        # i -> j
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
        output_file = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))

    except Exception as e:
        continue
