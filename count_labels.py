import os

label_dir = "./data/txt"
count_0 = 0
count_1 = 0
other = {}

for fname in os.listdir(label_dir):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, fname), "r") as f:
        for line in f:
            val = line.strip()
            if val == "0":
                count_0 += 1
            elif val == "1":
                count_1 += 1
            else:
                if val not in other:
                    other[val] = 0
                other[val] += 1

print(f"标签统计结果：")
print(f"  0 的数量: {count_0}")
print(f"  1 的数量: {count_1}")
if other:
    print("  其他值（建议检查是否异常）:")
    for k, v in other.items():
        print(f"    值 = {k} : {v} 次")
