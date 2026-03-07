import torch
from predictor.gru_model import GRUModel
from predictor.predictor import GRUPredictor
from burst_detection.detector import BurstDetector

model = GRUModel()
model.load_state_dict(torch.load("predictor/weights/gru_v1.pt"))
model.eval()

predictor = GRUPredictor(model)

detector = BurstDetector()

x = torch.randn(1, 30, 1)

forecast = predictor.predict_next_12(x)

actual = 120.0
lower  = 100.0
upper  = 110.0

state, explanation = detector.update(
    actual=actual,
    lower=lower,
    upper=upper
)

print("Next 12-step forecast:", forecast.tolist())
print("System state:", state)
print("Explanation:", explanation)
