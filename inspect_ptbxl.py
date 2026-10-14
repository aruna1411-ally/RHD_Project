
from pathlib import Path
import ast
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

# Update this path if your PTB-XL folder is elsewhere.
PROJECT = Path(r"C:\Users\HP\Desktop\RHD_Project")
DATA = PROJECT / "kaggle_ptxlb" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
OUTPUT = PROJECT / "processed" / "PTBXL"

OUTPUT.mkdir(parents=True, exist_ok=True)

metadata_file = DATA / "ptbxl_database.csv"

if not metadata_file.exists():
    raise FileNotFoundError(
        f"Cannot find {metadata_file}\n"
        "Check the PTB-XL folder location."
    )

df = pd.read_csv(metadata_file)

print("Total ECG records:", len(df))
print("Available columns:", df.columns.tolist())

# Use original 500 Hz signals for inspection.
path_column = "filename_hr"
signal_folder = DATA / "records500"
sampling_rate = 500

if not signal_folder.exists():
    # Fall back to the 100 Hz recordings if necessary.
    signal_folder = DATA / "records100"
    path_column = "filename_lr"
    sampling_rate = 100

if not signal_folder.exists():
    raise FileNotFoundError("Neither records500 nor records100 was found.")

print("Using:", signal_folder)
print("Sampling rate:", sampling_rate, "Hz")

# Read a few records to verify the paths and signal format.
sample = df.head(5)

for _, row in sample.iterrows():
    relative_path = str(row[path_column])
    record_path = DATA / relative_path

    # WFDB reads the header and binary signal together.
    signal, fields = wfdb.rdsamp(str(record_path))

    print(
        f"ECG ID={row['ecg_id']}, "
        f"shape={signal.shape}, "
        f"record_fs={fields['fs']}"
    )

# Plot the first record's 12 leads.
row = df.iloc[0]
record_path = DATA / str(row[path_column])
signal, fields = wfdb.rdsamp(str(record_path))

fig, axes = plt.subplots(12, 1, figsize=(13, 16), sharex=True)
time = [i / fields["fs"] for i in range(len(signal))]

for lead in range(signal.shape[1]):
    axes[lead].plot(time, signal[:, lead], linewidth=0.7)
    lead_name = fields["sig_name"][lead]
    axes[lead].set_ylabel(lead_name)
    axes[lead].grid(alpha=0.25)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle(f"PTB-XL ECG ID: {row['ecg_id']}")
fig.tight_layout()
fig.savefig(OUTPUT / "ptbxl_first_ecg.png", dpi=150)
plt.show()

# Export all leads of this record for inspection.
signal_df = pd.DataFrame(signal, columns=fields["sig_name"])
signal_df.insert(
    0, "time_seconds",
    [i / fields["fs"] for i in range(len(signal))]
)
signal_df.to_csv(OUTPUT / "ptbxl_first_ecg.csv", index=False)

# Save a small metadata preview.
sample.to_csv(OUTPUT / "ptbxl_metadata_preview.csv", index=False)

print("\nSUCCESS! Check:", OUTPUT)
