import joblib

scaler = joblib.load("data/processed/scaler.pkl")
scale_range = scaler.data_max_[0] - scaler.data_min_[0]

normalized_bound = 0.064  # t+1 value
raw_bound = normalized_bound * scale_range

print(raw_bound)