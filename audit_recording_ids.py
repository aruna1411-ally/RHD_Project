import pandas as pd

pcg_path = r"processed\PCG\pcg_metadata.csv"
ecg_path = r"processed\ECG\ecg_metadata.csv"

pcg = pd.read_csv(pcg_path)
ecg = pd.read_csv(ecg_path)

print("\n--- PCG metadata ---")
print("Columns:", pcg.columns.tolist())
print(pcg.head())
print("\nPCG label counts:")
print(pcg["label"].value_counts(dropna=False))
print("Unique PCG recordings:", pcg["recording_id"].nunique())

print("\n--- ECG metadata ---")
print("Columns:", ecg.columns.tolist())
print(ecg.head())
print("\nECG label counts:")
print(ecg["label"].value_counts(dropna=False))
print("Unique ECG recordings:", ecg["recording_id"].nunique())

pcg_ids = set(pcg["recording_id"].astype(str))
ecg_ids = set(ecg["recording_id"].astype(str))
common = pcg_ids & ecg_ids

print("\n--- Recording ID audit ---")
print("Common recordings:", len(common))
print("PCG-only recordings:", len(pcg_ids - ecg_ids))
print("ECG-only recordings:", len(ecg_ids - pcg_ids))

pcg_labels = pcg.groupby("recording_id")["label"].nunique()
ecg_labels = ecg.groupby("recording_id")["label"].nunique()

print("PCG recordings with conflicting labels:",
      int((pcg_labels > 1).sum()))
print("ECG recordings with conflicting labels:",
      int((ecg_labels > 1).sum()))

print("\nAudit complete. No models were changed.")
