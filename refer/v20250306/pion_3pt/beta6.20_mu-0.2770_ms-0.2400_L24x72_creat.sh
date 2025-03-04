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
module purge
module load cuda/11.4.4-gcc-10.3.0
module load openmpi/4.1.5-gcc-10.3.0
chroma=/public/home/sush/3pt_test/chroma/install_chroma/chroma_a100/chroma_a100/build/chroma/mainprogs/main/chroma
chack_data=0
conf_start=10000
gap=50
conf=$((${conf_start} + ${gap} * $@))
echo ${conf}
hadron=pion
mass=-0.2770
tsep=10
conf_type=beta6.20_mu-0.2770_ms-0.2400_L24x72
exe=./${conf_type}_creat.pion.py
xml_path=./xml/${conf_type}/${hadron}
mkdir -p ${xml_path}
xml=${xml_path}/${hadron}_tsep${tsep}_mass${mass}_${conf}.xml
quda_resource_path=./xml/${conf_type}/${hadron}/resource
mkdir -p ${quda_resource_path}
python3 ${exe} ${conf} >${xml}
mkdir -p ./output
QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=${quda_resource_path} mpirun -n 1 ${chroma} -geom 1 1 1 1 -i ${xml} >./output/${conf_type}_${conf} 2>&1
