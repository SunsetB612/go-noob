import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from extract_features import prepare_dataset

class CNNVAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B, 40, T]
        return self.net(x).squeeze(1)  # [B, T]

def train():
    X, Y = prepare_dataset("data/wav", "data/txt")  # [N, 40], [N]
    print(f"Total samples: {X.shape[0]}")

    # 以帧为单位切片，例如每100帧一组
    window_size = 100
    num_segments = X.shape[0] // window_size
    X = X[:num_segments * window_size].reshape(num_segments, window_size, 40)
    Y = Y[:num_segments * window_size].reshape(num_segments, window_size)

    X = np.transpose(X, (0, 2, 1))  # [B, 40, T]
    Y = Y.astype(np.float32)

    print(f"训练样本数: {X.shape[0]}, 每段帧数: {X.shape[2]}")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)), batch_size=32)

    model = CNNVAD()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)

        print(f"Epoch {epoch+1}: Train Loss={train_loss / len(train_loader.dataset):.4f}, "
              f"Val Loss={val_loss / len(val_loader.dataset):.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/cnn_vad_model.pth")

if __name__ == "__main__":
    train()
