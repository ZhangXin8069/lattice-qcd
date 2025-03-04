#!/bin/bash
#SBATCH --job-name=analyse
##SBATCH --partition=gpu-debug
#SBATCH --partition=cpu6248R,cpueicc
##SBATCH --partition=na100-ins
##SBATCH --partition=nv100-ins,nv100-sug
##SBATCH -w gpu028
#SBATCH --output=test_analyse.out
#SBATCH --nodes=1
#SBATCH -n 2
##SBATCH --cpus-per-task=2
output_1_file=/public/home/sush/3pt_distillation/analyse/pion/test.log
exe=/public/home/sush/3pt_distillation/analyse/pion/test.py
python -u ${exe} > ${output_1_file} 2>&1