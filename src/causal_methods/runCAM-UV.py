from causallearn.search.FCMBased.lingam import CAMUV
import pandas as pd
import os
import glob
import time

input_dir = '../selected_csv_dataset/'
output_dir = '../ground_truth_graph/CAM_UV_ground_truth_graph/'
os.makedirs(output_dir, exist_ok=True)

csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
if not csv_files:
    exit(1)

for csv_path in csv_files:
    base_name = os.path.basename(csv_path)

    try:
        data = pd.read_csv(csv_path)
        data = data.select_dtypes(include=['number']).dropna()

        if len(data) > 2000:
            data = data.sample(n=2000, random_state=42)

        columns = data.columns.tolist()

        start_time = time.time()
        MAX_TIME = 600

        P, U = CAMUV.execute(data.values, 0.05, 2)  # alpha=0.05, num_folds=2

        elapsed_time = time.time() - start_time
        print(f"⏱️ : {elapsed_time:.2f} seconds")

        directed_edges = []
        for i, parents in enumerate(P):
            child_name = columns[i]
            for p in parents:
                parent_name = columns[p]
                directed_edges.append(f"{child_name} -> {parent_name}")

        total_edges = len(directed_edges)
        output_lines = [str(total_edges)] + directed_edges

        output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
    except Exception as e:
        continue

