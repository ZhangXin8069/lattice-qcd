#!/usr/bin/env bash
#SBATCH --partition nv100-ins
#SBATCH --reservation user1_75
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 7
#SBATCH --mem 40G
#SBATCH --gres gpu:1
#SBATCH --job-name user6-test
#SBATCH --output %j.log
module purge #unload all loaded modules
module load gcc/10.3.0-gcc-4.8.5
module load git/2.40.0-gcc-10.3.0
module load cmake/3.22.2-gcc-10.3.0
module load cuda/11.4.4-gcc-10.3.0
module load hpcx/2.14/hpcx-mt
source /public/home/jiangxy/venv.public/bin/activate
python -c "import cupy as cp;print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
