import numpy as np

def load_labels(label_file):
    """从帧级标签文件中读取 0/1 序列"""
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f]
    return np.array(labels, dtype=np.int32)

def postprocess_prediction(pred, threshold=0.5, min_len=3):
    """将模型预测的概率序列转为语音活动区间"""
    pred_bin = (pred > threshold).astype(int)
    intervals = []
    start = None
    for i, val in enumerate(pred_bin):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start >= min_len:
                intervals.append((start, i))
            start = None
    if start is not None and len(pred_bin) - start >= min_len:
        intervals.append((start, len(pred_bin)))
    return intervals
