#!/bin/bash
#SBATCH --job-name=b7_0.145
##SBATCH --partition=cpu6248R,i72c512g,cpueicc
##SBATCH --partition=gpu-debug
##SBATCH --partition=nv100-ins,nv100-sug
##SBATCH --partition=na800-sug,na800-pcie
##SBATCH --partition=na100-40g
#SBATCH --partition=na100-sug
##SBATCH --partition=na100-ins
#SBATCH -w gpu042
#SBATCH --output=beta7.0_mu-0.1600_ms-0.1450_L48x64_creat.2.out
#SBATCH --array=0-75
#SBATCH --nodes=1
##SBATCH -n 
#SBATCH --cpus-per-task=4
##SBATCH --time="30:00"
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH --time="3-00:00:00"
module purge
# module load compiler/devtoolset/7.3.1
# module load mpi/openmpi/4.1.4/gnu-with-ucx
# module load cmake/3.22.2-gcc-10.3.0
# module load gcc/10.3.0-gcc-4.8.5
# module load hpcx/2.14/hpcx
module load cuda/11.4.4-gcc-10.3.0
# module load libxml2/2.9.12-gcc-10.3.0
module load openmpi/4.1.5-gcc-10.3.0 
chroma=/public/home/sush/3pt_test/chroma/install_chroma/chroma_a100/chroma_a100/build/chroma/mainprogs/main/chroma
# chroma=/public/home/chenc/Deuteron/run_chroma/chroma_gpu
# chroma=/public/home/sunp/chroma-llvm_A100_hpcx_AMD/install/sm_80_omp/chroma-double_quda/bin/chroma
conf_start=1000
gap=20
hadron=pion
tsep=36
mass=-0.1450
conf_type=beta7.0_mu-0.1600_ms-0.1450_L48x64
conf=$[${conf_start}+${gap}*${SLURM_ARRAY_TASK_ID}]
exe=./${conf_type}_creat.py
xml=./xml/${conf_type}/${hadron}_2pt_tsep${tsep}_${conf}.2.xml
python ${exe} ${hadron} ${conf} ${tsep} ${mass} > ${xml}
export QUDA_RESOURCE_PATH=./xml/${conf_type}/mass&{mass}
QUDA_ENABLE_TUNING=1 \
mpirun -x QUDA_RESOURCE_PATH -n 4 ${chroma} -geom 1 1 2 2 -i ${xml} > ./output/${conf_type}_${conf}_2 2>&1  
