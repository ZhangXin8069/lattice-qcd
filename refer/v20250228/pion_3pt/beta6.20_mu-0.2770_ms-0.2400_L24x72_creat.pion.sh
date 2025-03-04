#!/bin/bash
#SBATCH --job-name=peram_test
#SBATCH --output=test_creat_ENV.out
#SBATCH --array=0-45
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
##SBATCH --partition=gpu-debug
##SBATCH -n 2
##SBATCH --gres=gpu:2
##SBATCH --time="30:00"
#SBATCH --partition=gpup1
##SBATCH -w gpu005
#SBATCH -x gpu009
#SBATCH -n 2
#SBATCH --gres=gpu:2
##SBATCH --begin=now+6hour
#SBATCH --time="3-00:00:00"
module purge
module load openmpi/4.1.5-gcc-11.3.0
module load cuda/11.8.0-gcc-4.8.5
chroma=/public/home/sushihao/chroma/chroma_a100/build/chroma/mainprogs/main/chroma
# chroma=/public/home/chenc/Deuteron/run_chroma/chroma_gpu
# chroma=/public/home/sunp/chroma-llvm_A100_hpcx_AMD/install/sm_80_omp/chroma-double_quda/bin/chroma
export PATH=/public/home/sushihao/chroma/llvm-project-14.0.6.src/build/bin:$PATH
export LD_LIBRARY_PATH=/public/home/sushihao/chroma/llvm-project-14.0.6.src/build/lib:$LD_LIBRARY_PATH
chack_data=0
conf_start=10000
gap=50
conf=$[${conf_start}+${gap}*${SLURM_ARRAY_TASK_ID}]
# conf=$[${conf_start}+${gap}*${SLURM_ARRAY_TASK_ID}]
hadron=pion
mass=-0.2770
tsep=10
conf_type=beta6.20_mu-0.2770_ms-0.2400_L24x72
result_file=/public/home/sushihao/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L24x72/${hadron}_3pt_Px0Py0Pz0_ENV-1_conf${conf}_tsep${tsep}_mass${mass}_linkdir2_linkmax10.iog
exe=./beta6.20_mu-0.2770_ms-0.2400_L24x72_creat.pion.py
xml=./xml/beta6.20_mu-0.2770_ms-0.2400_L24x72/${hadron}/${hadron}_tsep${tsep}_mass${mass}_${conf}.xml
quda_resource_path=/public/home/sushihao/chroma/chroma_run/xml/${conf_type}/proton/resource
if [ ${chack_data} -eq 1 ]; then
    if [ -f ${result_file} ]; then
        echo result file exist
    else
        python ${exe} ${conf} > ${xml}
        QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=${quda_resource_path} mpirun -n 2 ${chroma} -geom 1 1 1 2 -i ${xml} > ./output/beta6.20_mu-0.2770_ms-0.2400_L24x72_${conf} 2>&1  
    fi
else
    python ${exe} ${conf} > ${xml}
    QUDA_ENABLE_TUNING=1 QUDA_RESOURCE_PATH=${quda_resource_path} mpirun -n 2 ${chroma} -geom 1 1 1 2 -i ${xml} > ./output/beta6.20_mu-0.2770_ms-0.2400_L24x72_${conf} 2>&1
fi
