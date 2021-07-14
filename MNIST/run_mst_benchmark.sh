#!/bin/bash

declare -a mst_weight_type=(cost random sum none)
num_iterations=5

path=$(pwd)
main_script=$path/main.py

csv_file=$path/benchmark.csv

for mst_weight in "${mst_weight_type[@]}"; do
    for ((iteration=0; iteration<$num_iterations; iteration++)); do
         bash ./launch_mnist_run.sh "$main_script" --mst "--mst_weight $mst_weight" "--csv_file $csv_file"  # For HPC change bash to sbatch
    done
done


for ((iteration=0; iteration<$num_iterations; iteration++)); do
     bash ./launch_mnist_run.sh "$main_script" "--csv_file $csv_file"
done
