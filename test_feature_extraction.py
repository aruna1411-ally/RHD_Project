import tensorflow as tf
import numpy as np

pcg_path = r"processed\PCG\best_pcg_cnn_clean.keras"
ecg_path = r"processed\ECG\best_ecg_1dcnn_clean.keras"

pcg_model = tf.keras.models.load_model(pcg_path, compile=False)
ecg_model = tf.keras.models.load_model(ecg_path, compile=False)

pcg_encoder = tf.keras.Model(
    inputs=pcg_model.inputs,
    outputs=pcg_model.get_layer("dense_2").output
)

ecg_encoder = tf.keras.Model(
    inputs=ecg_model.inputs,
    outputs=ecg_model.get_layer("dense_3").output
)

print("PCG encoder output:", pcg_encoder.output_shape)
print("ECG encoder output:", ecg_encoder.output_shape)

pcg_data = np.load(r"processed\PCG\pcg_spectrograms.npy", mmap_mode="r")
ecg_data = np.load(r"processed\ECG\ecg_segments.npy", mmap_mode="r")

print("PCG input data:", pcg_data.shape)
print("ECG input data:", ecg_data.shape)

# Test with one sample from each modality
pcg_sample = np.asarray(pcg_data[:1], dtype=np.float32)[..., np.newaxis]
ecg_sample = np.asarray(ecg_data[:1], dtype=np.float32)[..., np.newaxis]

print("PCG feature output:", pcg_encoder(pcg_sample, training=False).shape)
print("ECG feature output:", ecg_encoder(ecg_sample, training=False).shape)
