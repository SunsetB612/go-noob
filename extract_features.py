import numpy as np
import librosa
import os

def extract_mfcc(wav_path, sr=16000, n_mfcc=40, win_length=400, hop_length=160):
    y, _ = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=512, hop_length=hop_length, win_length=win_length)
    return mfcc.T  # [T, 40]

# 推理时统一接口
def extract_features(wav_path):
    return extract_mfcc(wav_path)

# 数据准备，帧级标签
def prepare_dataset(wav_dir, label_dir):
    X = []
    Y = []

    for fname in os.listdir(wav_dir):
        if not fname.endswith(".wav"):
            continue

        wav_path = os.path.join(wav_dir, fname)
        label_path = os.path.join(label_dir, fname.replace(".wav", ".txt"))

        mfcc = extract_mfcc(wav_path)  # [T, 40]

        with open(label_path, "r") as f:
            labels = [int(line.strip()) for line in f if line.strip() in ("0", "1")]

        labels = np.array(labels)
        T = min(len(mfcc), len(labels))
        mfcc = mfcc[:T]
        labels = labels[:T]

        X.append(mfcc.T)     # [40, T]
        Y.append(labels)     # [T]

    X = np.concatenate(X, axis=1)  # [40, N]
    Y = np.concatenate(Y)          # [N]
    return X.T, Y  # [N, 40], [N]
