import torch
from torch import nn
import joblib
from pathlib import Path


class GRUPredictor:
    def __init__(self, model: nn.Module, scaler_path: Path, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.scaler = joblib.load(scaler_path)

    def predict_next_12(self, x):
        """
        x: Tensor of shape (1, 30, 1) — normalized
        returns: Tensor of shape (12,) — RAW RPS
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            preds_norm = self.model(x).squeeze().cpu().numpy()

        # Inverse transform to raw RPS
        preds_raw = self.scaler.inverse_transform(
            preds_norm.reshape(-1, 1)
        ).flatten()

        return torch.tensor(preds_raw)