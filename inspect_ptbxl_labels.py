
from pathlib import Path
import ast
import pandas as pd

PROJECT = Path(r"C:\Users\HP\Desktop\RHD_Project")

DATA = (
    PROJECT
    / "kaggle_ptxlb"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
)

df = pd.read_csv(DATA / "ptbxl_database.csv")

print("Total records:", len(df))
print("Unique patients:", df["patient_id"].nunique())

print("\nMissing values:")
print(df[["age", "sex", "patient_id", "scp_codes"]].isna().sum())

print("\nFirst 10 diagnostic-code entries:")
print(df[["ecg_id", "scp_codes"]].head(10).to_string(index=False))

all_codes = []

for value in df["scp_codes"].dropna():
    try:
        codes = ast.literal_eval(value)
        all_codes.extend(codes.keys())
    except (ValueError, SyntaxError):
        continue

print("\nMost frequent diagnostic codes:")
print(pd.Series(all_codes).value_counts().head(25).to_string())

print("\nSUCCESS: Diagnostic-label inspection completed.")
