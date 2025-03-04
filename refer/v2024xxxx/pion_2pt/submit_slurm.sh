#!/bin/bash
#SBATCH --job-name=pi=CONF=
#SBATCH --partition=gpu-debug
#SBATCH --output=lap.=CONF=.out
#SBATCH --error=lap.=CONF=.out
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --gres=gpu:1
# conda deactivate
# module load cuda/11.4.4-gcc-10.3.0
# conda activate cupy114
source /public/home/zhangxin/env.sh
run_dir=.
input_dir=${run_dir}
exe=/public/home/xinghy/distillation_contract_example/cupy_contract_code/contrac_meson_cupy.py
echo "=CONF= job starts at" `date` > $run_dir/output/output_=CONF=.log
/public/home/xinghy/anaconda3-2023.03/bin/python $exe $input_dir/input/input_=CONF= >> $run_dir/output/output_=CONF=.log 2>&1 
echo "=CONF= job ends at" `date` >> $run_dir/output/output_=CONF=.log
