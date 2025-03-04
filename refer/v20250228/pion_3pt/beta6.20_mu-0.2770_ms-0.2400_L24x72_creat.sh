#!/bin/bash
#SBATCH --job-name=b6.2_0.277
##SBATCH --partition=cpu6248R,i72c512g,cpueicc
#SBATCH --partition=gpu-debug
##SBATCH --partition=nv100-ins,nv100-sug
##SBATCH --partition=na800-sug,na800-pcie
##SBATCH --partition=na100-40g
##SBATCH --partition=na100-sug
##SBATCH --partition=na100-ins
##SBATCH -w gpu042
#SBATCH --output=beta6.20_mu-0.2770_ms-0.2400_L24x72_creat.out
#SBATCH --array=0
#SBATCH --nodes=1
##SBATCH -n 
#SBATCH --cpus-per-task=4
#SBATCH --time="30:00"
##SBATCH --time="3-00:00:00"
#SBATCH -n 2
#SBATCH --gres=gpu:2
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
chack_data=0
conf_start=10000
gap=50
SLURM_ARRAY_TASK_ID=0
conf=$[${conf_start}+${gap}*${SLURM_ARRAY_TASK_ID}]
# conf=$[${conf_start}+${gap}*${SLURM_ARRAY_TASK_ID}]
hadron=pion
mass=-0.2770
tsep=10
conf_type=beta6.20_mu-0.2770_ms-0.2400_L24x72
# result_file=/public/home/yanght/3pt/chroma/beta6.20_mu-0.2770_ms-0.2400_L24x72/${hadron}_3pt_Px0Py0Pz0_ENV-1_conf${conf}_tsep${tsep}_mass${mass}_linkdir2_linkmax10.iog
result_file=./result/${hadron}_3pt_Px0Py0Pz0_ENV-1_conf${conf}_tsep${tsep}_mass${mass}_linkdir2_linkmax10.iog
exe=./beta6.20_mu-0.2770_ms-0.2400_L24x72_creat.pion.py
xml=./xml/beta6.20_mu-0.2770_ms-0.2400_L24x72/${hadron}/${hadron}_tsep${tsep}_mass${mass}_${conf}.xml
quda_resource_path=/public/home/yanght/3pt/chroma_run/xml/${conf_type}/proton/resource
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