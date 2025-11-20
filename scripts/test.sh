set -e

# for ((i=0;i<3;i=i+1)) do
#     python3 generate.py > input
#     python3 ans.py < input > ans
#     time ./$1 input > output
#     diff output ans || break
# done
# rm input output ans
# echo "[PASSED]"


python3 generate.py
python3 ans.py

for ((d=3;d<=6;d=d+1)) do
    for ((i=1;i<=5;i=i+1)) do
        echo "Testing nd${d}_${i}..."
        time ./$1 ../benchmarks/random/nd${d}_${i}.in > output
        diff output ../benchmarks/random/nd${d}_${i}.ans || ((echo "xxx") && exit 1)
        echo "[PASSED]"
        echo ""
    done
done
rm output