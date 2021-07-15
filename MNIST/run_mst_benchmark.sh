#!/bin/bash

declare -a mst_weight_type=("" "--mst --mst_weight cost" "--mst --mst_weight random" "--mst --mst_weight sum" "--mst --mst_weight none")
num_iterations=1

path=$(pwd)
main_script=$path/MNIST/main.py

csv_file=$path/MNIST/benchmark.csv

for ((iteration=0; iteration<$num_iterations; iteration++)); do
    for mst_weight in "${mst_weight_type[@]}"; do
         bash ./MNIST/launch_mnist_run.sh "$main_script" "$mst_weight" "--csv_file $csv_file"  # For HPC change bash to sbatch
    done
done


# for ((iteration=0; iteration<$num_iterations; iteration++)); do
#      bash ./launch_mnist_run.sh "$main_script" "--csv_file $csv_file"
# done
