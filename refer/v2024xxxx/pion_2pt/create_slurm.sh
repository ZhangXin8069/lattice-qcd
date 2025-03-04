#!/bin/bash
peram_dir="/public/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light"
#peram_dir="/public/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96"
for conf in {5000..9150..50}
# for conf in {11000..11000..50}
do
# if [ -d $peram_dir/${conf} ]; then
 sed "s/=CONF=/$conf/g" submit_slurm.sh > submit/submit.$conf.sh
 chmod +x submit/submit.$conf.sh
 #sbatch submit.$conf.sh
 #fi
done
