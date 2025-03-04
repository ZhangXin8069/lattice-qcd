#!/usr/bin/env bash
#SBATCH --partition=gpu-debug# you should modify it to your partition
## SBATCH --reservation zhangqa_71 # you should modify it to your reservation
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --gres gpu:1  #! necessary for GPU nodes
#SBATCH --job-name cupy_test
#SBATCH --output %j.log
module purge
module load gcc/10.3.0-gcc-4.8.5
module load git/2.40.0-gcc-10.3.0
module load cmake/3.22.2-gcc-10.3.0
module load cuda/11.4.4-gcc-10.3.0
module load hpcx/2.14/hpcx-mt
# source /public/home/jiangxy/venv.public/bin/activate
source ~/zhang_env.sh
# python -c "import cupy;print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
python g_2pt.py
# python Quark_propagator.py