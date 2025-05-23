import os
import numpy as np

# 参数设置
FRAME_SIZE = 240          # 每帧采样点数
AUDIO_LENGTH = 80000      # 假设音频最大长度（可根据实际情况动态处理）
LABEL_DIR = './data/label'
OUTPUT_DIR = './data/txt'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 遍历所有 label 文件
for file in os.listdir(LABEL_DIR):
    if not file.endswith('.txt'):
        continue

    label_path = os.path.join(LABEL_DIR, file)

    # 加载区间标签
    segments = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue
            start, end = map(int, parts)
            segments.append((start, end))

    # 计算帧数
    if not segments:
        print(f"跳过空标签文件: {file}")
        continue
    max_sample_index = max(end for (_, end) in segments)
    frame_count = (max_sample_index + FRAME_SIZE - 1) // FRAME_SIZE
    frame_labels = np.zeros(frame_count, dtype=np.int32)

    # 设置帧标签
    for (start, end) in segments:
        start_frame = start // FRAME_SIZE
        end_frame = end // FRAME_SIZE
        frame_labels[start_frame:end_frame + 1] = 1

    # 保存帧级标签
    output_path = os.path.join(OUTPUT_DIR, file)
    with open(output_path, 'w') as f:
        for label in frame_labels:
            f.write(f"{label}\n")

    print(f"生成帧标签: {output_path}（{len(frame_labels)}帧）")
