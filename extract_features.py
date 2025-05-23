import os
import numpy as np
import librosa
from pathlib import Path

SAMPLE_RATE = 8000
FRAME_LENGTH = 0.03  # 30ms
FRAME_STEP = 0.015   # 15ms


def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=int(FRAME_LENGTH*sr),
        hop_length=int(FRAME_STEP*sr), n_mels=40
    )
    mel_db = librosa.power_to_db(mel_spec)
    return mel_db.T  # (num_frames, 40)


def load_label(txt_path):
    with open(txt_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return np.array(labels, dtype=np.float32)


def prepare_dataset(wav_dir, label_dir):
    X, Y = [], []
    for wav_file in Path(wav_dir).glob("*.wav"):
        name = wav_file.stem
        feat = extract_features(str(wav_file))
        label_path = Path(label_dir) / f"{name}.txt"
        if not label_path.exists():
            continue
        label = load_label(str(label_path))
        min_len = min(len(feat), len(label))
        X.append(feat[:min_len])
        Y.append(label[:min_len])
    return np.concatenate(X), np.concatenate(Y)