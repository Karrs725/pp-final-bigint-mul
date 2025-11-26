import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

'''
need :
input file : inp
把所有東西丟裡面就好
'''


INPUT_FILE = "inp"

def parse_single_block(lines, idx):
    idx += 1
    benchmark = lines[idx].split()[2]
    idx += 1

    size = int(lines[idx].split()[5])
    if size % 10:
        size += 1
    idx += 1

    time = int(lines[idx].split()[3])
    idx += 1

    idx += 2
    return benchmark, size, time, idx

def main():
    with open(INPUT_FILE, "r") as f:
        text = f.read().strip().splitlines()

    raw = defaultdict(list)

    i = 0
    n = len(text)
    while i < n:
        bench, size, t, i = parse_single_block(text, i)
        raw[bench].append((size, t))

    # --- 每組 benchmark 取平均 ---
    averaged = {}
    all_sizes = set()

    for bench, arr in raw.items():
        bucket = defaultdict(list)
        for size, t in arr:
            bucket[size].append(t)

        #averaged[bench] = {s: sum(ts)/len(ts) for s, ts in bucket.items()}
        averaged[bench] = {s: min(ts) for s, ts in bucket.items()}
        all_sizes.update(averaged[bench].keys())

    all_sizes = sorted(all_sizes)

    # --- 必須有 long 當 baseline ---
    if "long-cpu" not in averaged:
        print("Error: no 'long' benchmark found in input.")
        return

    long_map = averaged["long-cpu"]   # size → time baseline
    # --- 開始畫柱狀圖（不畫 long） ---
    plt.figure(figsize=(12, 6))

    # 其他 bench 的名稱
    benches = [b for b in averaged.keys() if b != "long-cpu"]

    bar_width = 0.8 / len(benches)
    x = np.arange(len(all_sizes))

    for idx, bench in enumerate(benches):
        ys = []
        mp = averaged[bench]

        for size in all_sizes:
            if size in mp and size in long_map:
                ys.append( long_map[size]/mp[size] )
            else:
                ys.append(0)

        plt.bar(x + idx * bar_width, ys, width=bar_width, label=bench)

    # x 軸顯示 size
    plt.xticks(x + bar_width * (len(benches)-1)/2, all_sizes)

    plt.xlabel("Size")
    plt.ylabel("Normalized Time  (time / long time)")
    plt.title("Benchmark Performance Normalized to 'long'")

    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
