from torch import nn

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=32,
            batch_first=True
        )
        self.fc = nn.Linear(32, 12)

    def forward(self, x):
        out, _ = self.gru(x)       # (batch, 30, 32)
        last = out[:, -1, :]       # last timestep
        return self.fc(last)