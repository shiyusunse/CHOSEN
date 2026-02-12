import os
import pandas as pd

# Directory path
folder_path = "./output/APFD_c_Results"

# Get all CSV files
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Store results
results = []

# Iterate over each CSV file
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    # Read the file
    df = pd.read_csv(file_path)

    # Check if required columns exist
    required_cols = ['APFD_c', 'Recall20', 'Effort20']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️ Warning: File {file} is missing columns {missing_cols}, skipped.")
        continue

    # Calculate mean of each metric
    mean_apfd = df['APFD_c'].mean()
    mean_recall20 = df['Recall20'].mean()
    mean_effort20 = df['Effort20'].mean()

    # Save results
    results.append({
        "filename": os.path.splitext(file)[0],
        "mean_APFD_c": mean_apfd,
        "mean_Recall20": mean_recall20,
        "mean_Effort20": mean_effort20
    })

# Convert to DataFrame
result_df = pd.DataFrame(results)

# Sort by filename (optional)
result_df = result_df.sort_values(by="filename").reset_index(drop=True)

# Print results
print(result_df)

# Save results to CSV
output_path = "./output/APFD_c_Results_summary.csv"
result_df.to_csv(output_path, index=False)
print(f"\n✅ Average metrics results saved to {output_path}")