#!/bin/bash
cat << EOF
Nt 96
Nx 32
conf_id $1
Nev 100
Nev1 100
NevVVV 100
NevVdV 100
tsource_interval 96
tsource_start 0
tsep_min 0
tsep_max 40
number_of_processes 1
peram_u_dir /public/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/$1
peram_c_dir /public/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/charm/$1
peram_s_dir /public/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/strange/$1
VVV_dir /public/group/lqcd/VVV/beta6.41_mu-0.2295_ms-0.2050_L32x96/Nev100/$1
VdV_dir /public/group/lqcd/VdaggerV/beta6.41_mu-0.2295_ms-0.2050_L32x96/Nev100/$1
corr_dir /public/home/zhangxin/lattice-lqcd/meson_run1110/meson_run1110/result1/beta6.41_mu-0.2295_ms-0.2050_L32x96
EOF