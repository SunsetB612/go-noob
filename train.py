import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from extract_features import prepare_dataset

class CNNVAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, 40, 1]
        return self.net(x).squeeze(-1).squeeze(-1)  # 输出 [B]

def train():
    X, Y = prepare_dataset("data/wav", "data/txt")  # X: [N, 40], Y: [N]
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)


    X = X.astype(np.float32)
    Y = (Y > 0).astype(np.float32)

    print(f"正样本比例: {Y.sum() / len(Y):.4f}")

    # 分割数据
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # reshape 为 CNN 输入格式: [B, C=40, T=1]
    X_train = X_train[:, :, np.newaxis]
    X_val = X_val[:, :, np.newaxis]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = CNNVAD()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/cnn_vad_model.pth")

if __name__ == '__main__':
    train()
