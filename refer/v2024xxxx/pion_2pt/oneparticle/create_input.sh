#!/bin/bash
peram_dir="/public/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light"
# 13500..36400..50
for conf in {1000..1000..50}
# for conf in {11000..11000..50}
do
if [ -d ${peram_dir}/${conf} ]; then
./input.sh $conf > input_${conf}
fi
done