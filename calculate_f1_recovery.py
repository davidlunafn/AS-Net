import pandas as pd

# Load the data
try:
    df = pd.read_csv("results/rain_evaluation.csv")
except FileNotFoundError:
    print("Error: results/rain_evaluation.csv not found.")
    exit()

# Calculate the mean F1 scores
f1_clean_mean = df['f1_clean'].mean()
f1_mixed_mean = df['f1_mixed'].mean()
f1_separated_mean = df['f1_separated'].mean()

# Calculate the percentage of recovery
recovery_percentage = ((f1_separated_mean - f1_mixed_mean) / (f1_clean_mean - f1_mixed_mean)) * 100

# Print the results in the desired format
print(f"F1_clean: {f1_clean_mean:.2f}")
print(f"F1_mixed: {f1_mixed_mean:.2f}")
print(f"F1_separated: {f1_separated_mean:.2f}")
print(f"Percentage recovery: {recovery_percentage:.2f}%")

print("\n--- Formatted Sentence ---")
sentence = (
    f"Averaged across all SNR levels, the application of AS-Net resulted in a significant increase "
    f"in the mean F1-Score from {f1_mixed_mean:.2f} (for the mixed input) to {f1_separated_mean:.2f} "
    f"(for the separated output). This represents a recovery of approximately {recovery_percentage:.0f}% "
    f"of the detection performance lost due to the rainfall noise relative to the ideal performance "
    f"on the clean signals (mean F1-Score: {f1_clean_mean:.2f}), confirming the practical utility of "
    f"AS-Net for enhancing downstream bioacoustic analyses."
)
print(sentence)
