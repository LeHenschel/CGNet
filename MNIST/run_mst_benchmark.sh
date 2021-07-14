#!/bin/bash

declare -a mst_weight_type=(default random sum)
num_iterations=5

path=$(pwd)
main_script=$path/main.py

# python3 -u $1 --num-epoch 30 --lmax 11 --batch-size 100 --skip 1 --norm 1  --unrot-test --dropout 0.5 $2 $3

csv_file=/code/results/multi_mnist_augmentation/c4_augm_run.csv

for mst_weight in "${mst_weight_type[@]}"; do
    for ((iteration=0; iteration<$num_iterations; iteration++)); do
         bash ./launch_mnist_run.sh "$main_script" --mst "--mst_weight $mst_weight" 
    done
done


for ((iteration=0; iteration<$num_iterations; iteration++)); do
     bash ./launch_mnist_run.sh "$main_script"
done
