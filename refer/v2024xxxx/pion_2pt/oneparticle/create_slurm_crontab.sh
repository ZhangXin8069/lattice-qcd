#!/bin/bash
# run_dir="/public/home/liuming/LapH/contraction/python/Pentaquarks_v4/run_beta6.41_mu-0.2295_ms-0.2050_L32x96/part2/nv100-sug"
run_dir="/public/home/zhangxin/lattice-lqcd/meson_run1110/meson_run1110/oneparticle"
peram_dir="/public/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light"
#for conf in {3800..10000..50}
for conf in {1000..1000..50}
do
# if [ -d $peram_dir/${conf} ] && [ ! -f ${corr_dir}/corr_Sig_fig2_conf${conf}.npz ]; then
 if [ -d $peram_dir/${conf} ]; then
   if [ ! -f ${run_dir}/tag_${conf} ]; then
#      squeue -p nv100-sug -u liuming > ${run_dir}/joblist
      squeue -u changx > ${run_dir}/joblist
      # jobnumber=`grep --regexp="$" --count ${run_dir}/joblist`
      jobnumber=`grep -c 'meson' ${run_dir}/joblist`
      if [ $jobnumber -lt 200 ]; then
        sed "s/=CONF=/$conf/g" ${run_dir}/submit_slurm.sh > ${run_dir}/submit.$conf.sh
        chmod +x ${run_dir}/submit.$conf.sh
        sbatch ${run_dir}/submit.$conf.sh
        echo "${conf} job submitted" > ${run_dir}/tag_${conf}
      fi
    fi
  fi
done
