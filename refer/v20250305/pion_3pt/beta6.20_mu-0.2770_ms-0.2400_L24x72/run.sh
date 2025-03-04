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
chroma=/public/home/zhangxin/external-libraries/software/chroma/mainprogs/main/chroma
mkdir -p ./xml
mkdir -p ./resource
mkdir -p ./iog
mkdir -p ./log
chack_data=0
conf_start=10000
gap=50
index=$@
echo "index:${index}"
conf=$((${conf_start} + ${gap} * ${index}))
hadron=pion
mass=-0.2770
tsep=10
conf_type=beta6.20_mu-0.2770_ms-0.2400_L24x72
exe=./main.py
xml=./xml/${hadron}_tsep${tsep}_mass${mass}_${conf}.xml
quda_resource_path=./resource
iog_file=./iog/${hadron}_3pt_Px0Py0Pz0_ENV-1_conf${conf}_tsep${tsep}_mass${mass}_linkdir2_linkmax10.iog
if [ ${chack_data} -eq 1 ]; then
    if [ -f ${iog_file} ]; then
        echo iog file exist
    else
        python ${exe} ${conf} >${xml}
        QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=${quda_resource_path} mpirun -n 1 ${chroma} -geom 1 1 1 1 -i ${xml} >./log/${conf_type}_${conf} 2>&1
    fi
else
    python ${exe} ${conf} >${xml}
    QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=${quda_resource_path} mpirun -n 1 ${chroma} -geom 1 1 1 1 -i ${xml} >./log/${conf_type}_${conf} 2>&1
fi
