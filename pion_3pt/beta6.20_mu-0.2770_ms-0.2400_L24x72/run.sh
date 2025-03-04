#!/bin/bash
#SBATCH --job-name=ssub
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --time=00-00:30:00
#SBATCH --output=ssub.out
#SBATCH --error=ssub.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangxin8069@qq.com
#SBATCH --gres=gpu:2
source /public/home/zhangxin/env.sh
bash ./clean.sh
chack_data=0
conf_stare=10000
gap=50
conf=$((${conf_stare} + ${gap} * ${SLURM_ARRAY_TASK_ID}))
hadron=pion
mass=-0.2770
tsep=10
conf_type=beta6.20_mu-0.2770_ms-0.2400_L24x72
output_file=./output/${hadron}_3pt_Px0Py0Pz0_ENV-1_conf${conf}_tsep${tsep}_mass${mass}_linkdir2_linkmax10.iog
exe=./main.py
xml=./xml/${hadron}_tsep${tsep}_mass${mass}_${conf}.xml
quda_resource_path=./xml/resource
mkdir -p ${quda_resource_path}
mkdir -p ./output
if [ ${chack_data} -eq 1 ]; then
    if [ -f ${output_file} ]; then
        echo output file exist
    else
        python ${exe} ${conf} >${xml}
        QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=${quda_resource_path} mpirun -n 2 ${chroma} -geom 1 1 1 2 -i ${xml} > ./output/${conf_type}_${conf} 2>&1
    fi
else
    python ${exe} ${conf} >${xml}
    QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=${quda_resource_path} mpirun -n 2 ${chroma} -geom 1 1 1 2 -i ${xml} > ./output/${conf_type}_${conf} 2>&1
fi