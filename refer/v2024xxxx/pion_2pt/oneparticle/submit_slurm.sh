#!/bin/bash
#SBATCH --job-name=test  #meson_=CONF=
#SBATCH --partition=gpu-debug
#SBATCH --output=lap.=CONF=.out
#SBATCH --error=lap.=CONF=.out
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
##SBATCH --gres=cpu:1
# conda deactivate
# module load cuda/11.4.4-gcc-10.3.0
# conda activate cupy114
run_dir=.
input_dir=${run_dir}
exe=/public/home/zhangxin/lattice-lqcd/meson_run1110/meson_run1110/oneparticle/contrac_2pt_meson_multiprocess_test.py
# exe=/public/home/xinghy/distillation_contract_example/cupy_contract_code/contrac_meson_cupy.py
echo "=CONF= job starts at" `date` > $run_dir/output_=CONF=.log
/public/home/xinghy/anaconda3-2023.03/bin/python -u $exe $input_dir/input_=CONF= >> $run_dir/output_=CONF=.log 2>&1 
echo "=CONF= job ends at" `date` >> $run_dir/output_=CONF=.log
