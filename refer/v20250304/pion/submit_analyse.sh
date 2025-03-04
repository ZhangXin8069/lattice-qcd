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
# input_file=./analyse_input.sh
# # input_file=./analyse_input_copy.sh
# exe=/public/home/sush/3pt_distillation/analyse/pion/analyse.py
# output_file=./analyse.log
# echo "analyse job starts at" `date` > ${output_file}
# python -u $exe ${input_file} >> ${output_file} 2>&1
output_1_file=/public/home/sush/3pt_distillation/analyse/pion/analyse.log
exe=/public/home/sush/3pt_distillation/analyse/pion/analyse.py
python -u ${exe} > ${output_1_file} 2>&1
# chake the data which is not exist 
# for i in {10000..17500..50}; do if [ -f corr_uud_plus_Px0Py0Pz0_conf${i}_2pt.dat ];then continue; else echo $i; fi; done