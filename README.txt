RHD MULTIMODAL DEMO - QUICK SETUP

1) Copy app.py and requirements.txt into the RHD_Project root folder (the same level as data/, notebooks/, processed/).
2) Ensure these existing model files are present:
   processed/PCG/best_pcg_cnn_clean.keras
   processed/ECG/best_ecg_1dcnn_clean.keras
   processed/PCG_ECG_best_attention_fusion_clean.keras
3) Open terminal/Command Prompt in the RHD_Project folder.
4) Install dependencies:  pip install -r requirements.txt
5) Start app:             streamlit run app.py
6) Open the local URL printed by Streamlit (usually http://localhost:8501).

INPUTS
- PCG: .wav file, at least 1.2 seconds.
- ECG: .csv/.txt numeric samples or .npy 1-D array, at least 4000 samples.
- Set the ECG sampling rate to match the source data. The default is 2000 Hz, matching the dataset used during development.

NOTES
- Reuses the trained PCG, ECG, and attention-fusion models; does not retrain.
- Current trained task is binary Normal/Abnormal only. No clinical-data branch or severity prediction is implemented.
- Grad-CAM views explain the separate PCG and ECG branch models, not the fusion layer directly.
- Research prototype only; not clinically validated and not for diagnosis/treatment decisions.
