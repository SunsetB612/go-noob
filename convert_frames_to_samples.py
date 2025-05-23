def convert_frame_to_sample(input_txt, output_txt, frame_shift=160):
    with open(input_txt, "r") as fin, open(output_txt, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            start_frame, end_frame = map(int, parts)
            start_sample = start_frame * frame_shift
            end_sample = end_frame * frame_shift
            fout.write(f"{start_sample}, {end_sample}\n")

if __name__ == "__main__":
    files = [("output/data_1_vad.txt", "output/data_1_samples.txt"),
             ("output/data_2_vad.txt", "output/data_2_samples.txt"),
             ("output/data_3_vad.txt", "output/data_3_samples.txt")]

    for input_path, output_path in files:
        convert_frame_to_sample(input_path, output_path)
        print(f"✅ 转换完成：{output_path}")
