#!/bin/bash
echo $@

echo " python3 -u $1 --num-epoch 30 --lmax 11 --batch-size 100 --skip 1 --norm 1  --unrot-test --dropout 0.5 $2 $3"
