import tensorflow as tf

paths = {
    "PCG": r"processed\PCG\best_pcg_cnn_clean.keras",
    "ECG": r"processed\ECG\best_ecg_1dcnn_clean.keras",
}

for name, path in paths.items():
    print("\n" + "=" * 60)
    print(f"{name} MODEL")
    model = tf.keras.models.load_model(path, compile=False)

    for i, layer in enumerate(model.layers):
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            shape = "unavailable"

        print(
            f"{i:2} | {layer.name:30} | "
            f"{layer.__class__.__name__:25} | {shape}"
        )
