import os
parity = "su"
for i in range(500,3490,10):
    filepath = "/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/proton/C48P29/result/"
    file = F"{filepath}/corr_Nucleon_pp_Px0Py0Pz1.conf{i}.dat"
    if os.path.exists(file):
        pass
        #print(F" {i} exists")
    else:
        print(F"\033[91m {i} doesn't exist\033[0m")
        
