from causallearn.search.PermutationBased.GRaSP import grasp
import pandas as pd
import numpy as np
import os
import glob

INPUT_DIR = '../selected_csv_dataset/'
OUTPUT_DIR = '../ground_truth_graph/GRaSP_ground_truth_graph/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

if not csv_files:
    exit(1)

for csv_path in csv_files:
    try:
        data = pd.read_csv(csv_path)
        data = data.select_dtypes(include=['number']).dropna()
        X = data.values.astype(float)
        columns = data.columns.tolist()

        G = grasp(X, depth=1, parameters={'lambda_value': 4})

        directed_edges = []
        nodes = G.get_nodes()
        node_name_map = {node: columns[i] for i, node in enumerate(nodes)}

        edges = G.get_graph_edges()
        for edge in edges:
            if edge.get_endpoint1().name == "TAIL" and edge.get_endpoint2().name == "ARROW":
                parent_node = edge.get_node1()
                child_node = edge.get_node2()
                directed_edges.append(f"{node_name_map[parent_node]} -> {node_name_map[child_node]}")
            elif edge.get_endpoint2().name == "TAIL" and edge.get_endpoint1().name == "ARROW":
                parent_node = edge.get_node2()
                child_node = edge.get_node1()
                directed_edges.append(f"{node_name_map[parent_node]} -> {node_name_map[child_node]}")

        total_edges = len(directed_edges)
        output_lines = [str(total_edges)]
        output_lines.extend(directed_edges)

        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

    except Exception as e:
        continue
