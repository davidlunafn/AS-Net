import pandas as pd

# Load the data
try:
    df = pd.read_csv("results/rain_evaluation.csv")
except FileNotFoundError:
    print("Error: results/rain_evaluation.csv not found.")
    exit()

# Columns to process
metrics = ['si_sdr', 'sdr', 'sir']
states = ['mixture', 'separated']
columns_to_process = [f"{metric}_{state}" for metric in metrics for state in states]

# Group by SNR and calculate mean and std
grouped = df.groupby('snr_level')[columns_to_process].agg(['mean', 'std'])

# --- Console Table Generation ---

print("Objective evaluation results (mean ± std dev) on the pseudo-real (rainfall noise) test set, broken down by input SNR.")

# Create a multi-level header for the table
header = [
    ("Input SNR (dB)", ""),
    ("SI-SDR (dB)", "Mixture"),
    ("SI-SDR (dB)", "Separated"),
    ("SDR (dB)", "Mixture"),
    ("SDR (dB)", "Separated"),
    ("SIR (dB)", "Mixture"),
    ("SIR (dB)", "Separated"),
]

# Prepare data for the table
table_data = []
for snr_level, row in grouped.iterrows():
    table_row = [f"{snr_level}"]
    for metric in metrics:
        for state in states:
            mean = row[(f'{metric}_{state}', 'mean')]
            std = row[(f'{metric}_{state}', 'std')]
            table_row.append(f"{mean:.2f} ± {std:.2f}")
    table_data.append(table_row)

# Manually format the table
# First, print the header
header_line_1 = ""
header_line_2 = ""
for col1, col2 in header:
    header_line_1 += f"{col1:^20}"
    header_line_2 += f"{col2:^20}"

print(header_line_1)
print(header_line_2)
print("-" * 20 * len(header))


# Print the data
for row in table_data:
    row_str = ""
    for item in row:
        row_str += f"{item:^20}"
    print(row_str)