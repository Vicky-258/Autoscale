from uncertainty.calibrate import calibrate

calibrate(
    model_path="predictor/weights/gru_v1.pt",
    scaler_path="data/processed/scaler.pkl",
    X_path="data/processed/X.npy",
    y_path="data/processed/y.npy",
    output_path="uncertainty/calibration.json",
)