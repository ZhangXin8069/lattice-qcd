#!/bin/bash
bash ./clean.sh
for i in $(seq 0 59); do
    while (($(squeue -p gpu-debug -u zhangxin | wc -l) > 2)); do
        sleep 10
    done
    echo "sbatch ./run.sh $i"
    sbatch ./run.sh $i
done
