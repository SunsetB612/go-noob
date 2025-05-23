import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from extract_features import extract_features  # 特征提取函数
from utils import postprocess_prediction       # 后处理函数


# 模型结构应与 train.py 中完全一致
class CNNVAD(nn.Module):
    def __init__(self):
        super(CNNVAD, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, 40, T]
        return self.net(x).squeeze(1)  # → [B, T]


def infer(wav_dir="data/wav", model_path="model/cnn_vad_model.pth", output_dir="output"):
    # 加载模型
    model = CNNVAD()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    wav_dir = Path(wav_dir)
    for wav_file in wav_dir.glob("*.wav"):
        print(f"\n🔍 Processing {wav_file.name}...")

        # 1. 提取 MFCC 特征（shape: [T, 40]）
        mfcc = extract_features(str(wav_file))  # 每帧40维

        # 2. 转为输入格式 [1, 40, T]
        X = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)  # [T, 40] -> [1, 40, T]

        # 3. 推理 [1, T]
        with torch.no_grad():
            pred = model(X).squeeze().numpy()  # [T]

        # 4. 后处理
        intervals = postprocess_prediction(pred, threshold=0.5, min_len=3)

        # 5. 打印并保存结果
        print(f"✅ Detected {len(intervals)} speech segments:")
        for i, (start, end) in enumerate(intervals):
            print(f"   Segment {i+1}: Frames {start} to {end}")

        # 输出到文件
        out_path = Path(output_dir) / f"{wav_file.stem}_vad.txt"
        with open(out_path, "w") as f:
            for start, end in intervals:
                f.write(f"{start} {end}\n")


if __name__ == '__main__':
    infer()
