import torch
from torch import nn

class GRUPredictor:
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def fit(self, train_loader, epochs, optimizer, criterion):
        self.model.train()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

    def predict_next_12(self, x):
        """
        x: Tensor of shape (1, 30, 1)
        returns: Tensor of shape (12,)
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            preds = self.model(x)
            return preds.squeeze().cpu()
