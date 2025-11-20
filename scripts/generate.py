import sys
from random import *

sys.set_int_max_str_digits(0)

for d in range(3, 7):
    low = 0
    hei = 10 ** (10 ** d)
    for i in range(5):
        f = open(f"../benchmarks/random/nd{d}_{i+1}.in", "w")
        a = randint(low, hei)
        b = randint(low, hei)
        f.write(f"{a}\n{b}")
        f.close()