import tensorflow as tf

models = [
    r"processed\PCG\best_pcg_cnn_clean.keras",
    r"processed\ECG\best_ecg_1dcnn_clean.keras",
    r"processed\PCG_ECG_best_attention_fusion_clean.keras",
]

for path in models:
    print("\n" + "=" * 60)
    print(path)
    try:
        model = tf.keras.models.load_model(path, compile=False)
        print("Inputs:", [(x.name, tuple(x.shape)) for x in model.inputs])
        print("Outputs:", [(x.name, tuple(x.shape)) for x in model.outputs])
        print("Parameters:", model.count_params())
        model.summary()
    except Exception as e:
        print("LOAD ERROR:", repr(e))
