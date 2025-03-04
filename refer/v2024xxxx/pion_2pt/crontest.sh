#!/bin/sh
#sleep 4.6h
for i in {5950..9150..50}
do
        
while (( $(squeue -p gpu-debug -u zhangxin| wc -l) > 3))
do
        sleep 30s
done
sbatch /public/home/zhangxin/lattice-lqcd/meson_run1110/meson_run1110/submit/submit.$i.sh
done
done
done
