import sys

sys.set_int_max_str_digits(0)

for d in range(3, 7):
    for i in range(5):
        f = open(f"../benchmarks/random/nd{d}_{i+1}.in", "r")
        a = f.readline()
        b = f.readline()
        f.close()
        f = open(f"../benchmarks/random/nd{d}_{i+1}.ans", "w")
        f.write(str(int(a) * int(b)) + "\n")
        f.close()