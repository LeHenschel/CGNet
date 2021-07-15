#!/bin/bash

#SBATCH -A SNIC2020-33-37 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1
#SBATCH -t 0-02:00:00

python3 -m MNIST.main --num-epoch 1 --lmax 11 --batch-size 100 --skip 1 --norm 1  --unrot-test --dropout 0.5 $2 $3 $4
