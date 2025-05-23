import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from extract_features import extract_features

def postprocess_prediction(probs, threshold=0.5, min_len=3):
    speech = probs > threshold
    segments = []
    start = None
    for i, s in enumerate(speech):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None
    if start is not None and len(speech) - start >= min_len:
        segments.append((start, len(speech)))
    return segments

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

def infer(wav_dir="data/wav", model_path="model/cnn_vad_model.pth", output_dir="output"):
    model = CNNVAD()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for wav_path in Path(wav_dir).glob("*.wav"):
        print(f"\n🔍 Processing {wav_path.name}...")
        mfcc = extract_features(str(wav_path))  # [T, 40]
        mfcc = mfcc.T  # [40, T]

        X = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, 40, T]
        with torch.no_grad():
            probs = model(X).squeeze(0).numpy()  # [T]

        segments = postprocess_prediction(probs, threshold=0.5, min_len=3)
        print(f"✅ Detected {len(segments)} speech segments.")
        for i, (start, end) in enumerate(segments):
            print(f"   Segment {i+1}: Frame {start} to {end}")

        with open(Path(output_dir) / f"{wav_path.stem}_vad.txt", "w") as f:
            for start, end in segments:
                f.write(f"{start} {end}\n")

if __name__ == "__main__":
    infer()
