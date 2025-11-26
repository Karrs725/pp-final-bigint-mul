import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_FILE = "inp"   # ← 寫死檔名

def parse_single_block(lines, idx):
    """解析 single() 格式的一筆資料，回傳 (benchmark, size, time), 以及下一行索引"""
    idx += 1  # skip
    benchmark = lines[idx].split()[2]
    idx += 1

    size = int(lines[idx].split()[5])
    if size % 10:
        size += 1
    idx += 1

    time = int(lines[idx].split()[3])
    idx += 1

    idx += 2  # skip last two lines

    return benchmark, size, time, idx


def main():
    with open(INPUT_FILE, "r") as f:
        text = f.read().strip().splitlines()

    raw = defaultdict(list)  # benchmark → list of (size, time)

    i = 0
    n = len(text)
    while i < n:
        bench, size, t, i = parse_single_block(text, i)
        raw[bench].append((size, t))

    # ===========================
    #  同 size 的 time 取平均
    # ===========================
    averaged = {}

    for bench, arr in raw.items():
        bucket = defaultdict(list)
        for size, t in arr:
            bucket[size].append(t)

        # 算平均
        averaged[bench] = [(size, sum(times)/len(times)) for size, times in bucket.items()]

        # 按 size 排序
        averaged[bench].sort()

    # ===========================
    #        畫圖
    # ===========================
    plt.figure(figsize=(10, 6))

    for bench, arr in averaged.items():
        xs = [x for x, _ in arr]
        ys = [y for _, y in arr]
        plt.plot(xs, ys, marker="o", label=bench)

    plt.xlabel("Size")
    plt.ylabel("Time")
    plt.xscale("log")  
    #plt.yscale("log")  
    plt.title("Benchmark Performance (Averaged)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
