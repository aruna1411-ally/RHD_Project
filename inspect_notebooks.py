
from pathlib import Path
import nbformat

folder = Path("notebooks")

files = [
    "03_ecg_dataset_audit.ipynb",
    "04_multimodal_fusion.ipynb",
]

terms = [
    "ecg_segments",
    "label",
    "sampling_rate",
    "resample",
    "normalize",
    "train_test_split",
    "rhd_probability",
    "fit(",
]

for filename in files:
    path = folder / filename

    print(f"\n{'=' * 60}")
    print(f"NOTEBOOK: {filename}")
    print(f"{'=' * 60}")

    if not path.exists():
        print("Notebook not found:", path)
        continue

    notebook = nbformat.read(path, as_version=4)

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source
        if any(term in source.lower() for term in terms):
            print(f"\n--- CELL {index} ---")
            print(source[:3500])
