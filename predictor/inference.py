import torch
from predictor.gru_model import GRUModel
from predictor.gru_predictor import GRUPredictor

model = GRUModel()
model.load_state_dict(torch.load("predictor/weights/gru_v1.pt"))

predictor = GRUPredictor(model)

# example input
x = torch.randn(1, 30, 1)

forecast = predictor.predict_next_12(x)
print("Next 12-step forecast:", forecast.tolist())
