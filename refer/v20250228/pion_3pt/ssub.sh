#!/bin/bash
bash  ./clean.sh
for i in $(seq 0 20); do
    while (($(squeue -p gpu-debug -u zhangxin | wc -l) > 2)); do
        sleep 30
    done
    echo "sbatch ./run.sh $i"
    sbatch ./run.sh $i
done
