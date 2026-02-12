import os
import pandas as pd

folder_path = "../data/BeamNG_RF_2_Complete/10_feature_GA"

csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

results = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)

    df = pd.read_csv(file_path)

    if 'APFD' not in df.columns:
        continue

    mean_apfd = df['APFD'].mean()

    results.append({
        "filename": os.path.splitext(file)[0],
        "mean_APFD": mean_apfd
    })

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="filename").reset_index(drop=True)

output_path = "../output/APFD_c_Results_summary.csv"

if os.path.exists(output_path):
    result_df.to_csv(output_path, mode='a', index=False, header=False)
else:
    result_df.to_csv(output_path, index=False)
print(result_df)
