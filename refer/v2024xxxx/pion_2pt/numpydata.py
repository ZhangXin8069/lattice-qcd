import numpy as np
import sys
sys.path.append("/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/di_lambda_c")
from lsq_tools import *
data=read_dat_file("/public/group/lqcd/corr/beta6.41_mu-0.2295_ms-0.2050_L48x96/2pt/Pion/corr_uu_gamma5_Px0Py0Pz0_conf",".dat",False)
print(data.shape)
np.save("/public/home/zhangxin/lattice-lqcd/meson_run1110/meson_run1110/numpy/pion48.npy",data)