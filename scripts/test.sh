set -e

# cmake -B build && cmake --build build -j

rm *.out || true
for ((d=3;d<=5;d=d+1)) do
    for ((i=1;i<=5;i=i+1)) do
        echo "Processing nd${d}_${i}.in"
        ../build/bin/benchmark --impl long-cpu --file ../benchmarks/random/nd${d}_${i}.in >> long.out
        ../build/bin/benchmark --impl long-cpu-simd --file ../benchmarks/random/nd${d}_${i}.in >> long-simd.out
        ../build/bin/benchmark --impl long-cpu-thread --file ../benchmarks/random/nd${d}_${i}.in >> long-thread.out
        ../build/bin/benchmark --impl long-cpu-thread-simd --file ../benchmarks/random/nd${d}_${i}.in >> long-thread-simd.out
        ../build/bin/benchmark --impl long-cpu-omp --file ../benchmarks/random/nd${d}_${i}.in >> long-omp.out
    done
done

for ((d=3;d<=6;d=d+1)) do
    for ((i=1;i<=5;i=i+1)) do
        echo "Processing nd${d}_${i}.in"
        ../build/bin/benchmark --impl ntt-cpu --file ../benchmarks/random/nd${d}_${i}.in >> ntt.out
        ../build/bin/benchmark --impl ntt-cpu-thread --file ../benchmarks/random/nd${d}_${i}.in >> ntt-thread.out
        ../build/bin/benchmark --impl ntt-cpu-omp --file ../benchmarks/random/nd${d}_${i}.in >> ntt-omp.out
        ../build/bin/benchmark --impl ntt-cpu-thread-old --file ../benchmarks/random/nd${d}_${i}.in >> ntt-thread-old.out
    done
done