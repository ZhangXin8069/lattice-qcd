#!/bin/bash
#SBATCH --job-name=pion_3pt
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --time=00-00:30:00
#SBATCH --output=ssub.out
#SBATCH --error=ssub.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangxin8069@qq.com
#SBATCH --gres=gpu:1
unset
export LD_LIBRARY_PATH=''
module purge
module load cuda/11.4.4-gcc-10.3.0
module load openmpi/4.1.5-gcc-10.3.0 
module load python/3.9.10-gcc-10.3.0
chroma=/public/home/sush/3pt_test/chroma/install_chroma/chroma_a100/chroma_a100/build/chroma/mainprogs/main/chroma
mkdir -p ./xml
mkdir -p ./iog
mkdir -p ./log
mkdir -p ./resource
conf_start=10000
gap=50
index=$@
conf=$((${conf_start} + ${gap} * ${index}))
hadron=pion
mass=-0.2770
tsep=10
conf_type=beta6.20_mu-0.2770_ms-0.2400_L24x72
xml=./xml/${hadron}_tsep${tsep}_mass${mass}_${conf}.xml
echo "######INDEX:${index}~${hadron}_tsep${tsep}_mass${mass}_${conf} is running!!!#######"
python3 ./main.py ${conf} >${xml}
QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=./resource mpirun -n 1 ${chroma} -geom 1 1 1 1 -i ${xml} >./log/${conf_type}_${conf} 2>&1
echo "######INDEX:${index}~${hadron}_tsep${tsep}_mass${mass}_${conf} is done!!!#######"