import numpy as np
import pandas as pd

pcg = pd.read_csv(r"processed\PCG\pcg_metadata.csv")
ecg = pd.read_csv(r"processed\ECG\ecg_metadata.csv")

pcg_data = np.load(
    r"processed\PCG\pcg_spectrograms.npy", mmap_mode="r"
)
ecg_data = np.load(
    r"processed\ECG\ecg_segments.npy", mmap_mode="r"
)

# Confirm metadata rows correspond to array samples
print("PCG rows match array:", len(pcg) == len(pcg_data))
print("ECG rows match array:", len(ecg) == len(ecg_data))

# Count segments belonging to each recording
pcg_counts = pcg.groupby("recording_id").size()
ecg_counts = ecg.groupby("recording_id").size()

print("\nPCG segments per recording:")
print(pcg_counts.describe())

print("\nECG segments per recording:")
print(ecg_counts.describe())

# Compare labels for common recordings
pcg_label = pcg.groupby("recording_id")["label"].first()
ecg_label = ecg.groupby("recording_id")["label"].first()

common = sorted(set(pcg_label.index) & set(ecg_label.index))
comparison = pd.DataFrame({
    "PCG_label": pcg_label.loc[common],
    "ECG_label": ecg_label.loc[common]
})

print("\nCommon recordings with matching labels:",
      int((comparison["PCG_label"] == comparison["ECG_label"]).sum()))
print("Common recordings with differing labels:",
      int((comparison["PCG_label"] != comparison["ECG_label"]).sum()))

print("\nFirst 10 recordings:")
print(comparison.head(10))
