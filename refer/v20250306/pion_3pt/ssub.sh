#!/bin/bash
for i in $(seq 0 20); do
    while (($(squeue -p gpu-debug -u zhangxin | wc -l) > 2)); do
        sleep 30
    done
    echo "sbatch ./beta6.20_mu-0.2770_ms-0.2400_L24x72_creat.sh $i"
    sbatch ./beta6.20_mu-0.2770_ms-0.2400_L24x72_creat.sh $i
done
