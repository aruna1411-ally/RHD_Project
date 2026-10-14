
import numpy as np
import pandas as pd
import tensorflow as tf
import os

x = np.load("processed/ECG/ecg_segments.npy")
d = pd.read_csv("processed/ECG/ecg_metadata.csv")

print("Labels:", d["label"].value_counts().to_dict())

for name in ["best_ecg_1dcnn.keras", "best_ecg_1dcnn_clean.keras"]:
    path = os.path.join("processed", "ECG", name)
    model = tf.keras.models.load_model(path, compile=False)

    print("\nModel:", name)

    for label in sorted(d["label"].unique()):
        indices = np.where(d["label"].to_numpy() == label)[0][:200]
        samples = x[indices].astype("float32")[..., None]
        scores = model.predict(samples, verbose=0).ravel()

        print(
            "Label:", label,
            "| Count:", len(scores),
            "| Mean:", round(float(scores.mean()), 4),
            "| Min:", round(float(scores.min()), 4),
            "| Max:", round(float(scores.max()), 4)
        )
