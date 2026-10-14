
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

# Inspect all diagnostic codes and their frequencies
counts = {}

for value in df["scp_codes"].dropna():
    try:
        codes = ast.literal_eval(value)
        for code in codes:
            counts[code] = counts.get(code, 0) + 1
    except (ValueError, SyntaxError):
        pass

code_counts = pd.Series(counts).sort_values(ascending=False)

# Search for relevant terms in the available metadata
terms = ["rheumatic", "rheumatic heart", "RHD", "valvular", "mitral stenosis"]

print("=== Metadata text search ===")
for term in terms:
    matches = df[
        df.astype(str).apply(
            lambda col: col.str.contains(
                term, case=False, regex=False
            )
        ).any(axis=1)
    ]
    print(f"{term}: {len(matches)} matching records")
    if len(matches):
        print(matches[["ecg_id", "report", "scp_codes"]].head(3).to_string(index=False))

print("\n=== Diagnostic code names ===")
print(code_counts.to_string())

print("\nNote: No matching text does not prove that no RHD cases exist.")
print("SUCCESS")
