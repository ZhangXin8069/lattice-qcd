#!/bin/bash
cat << EOF
Nt 96
Nx 32
conf_id $1
Nev1 100
Px 0
Py 0
Pz 0
eig_dir /public/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/$1
peram_u_dir /public/group/lqcd/liuming/LapH/data/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/$1
eigen_dir /public/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/$1
VdV_dir /public/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/$1/VdV
corr_pion_dir ./result
EOF
#/public/group/lqcd/liuming/LapH/data/perambulators/
# /public/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/$1