#!/beegfs/home/liuming/software/install/python/bin/python3
# -*-coding:gb2312-*-
# test file to check whether first term has problem
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lsq_tools import *
state = "F32P30"
object = "$\Lambda_c \Lambda_c$"    #  "$D_s^{+}$"  # "$\overline{D}$$^{*0}$"
a = 0.077
N = 567
P = np.arange(4)
Momset=["[0,0,0],[0,0,1]","[0,1,0],[0,-1,1]","[1,1,0],[-1,-1,1]","[0,0,-1],[0,0,2]"]
Nt = 96
conf_name="beta6.41_mu-0.2295_ms-0.2050_L32x96"
savefile = f"/public/home/changx/meson_run1110/apicture/all_mom.jpg"
contract = []
color = ["red", "green", "purple", "blue", "cornflowerblue"]
sub=".dat"   
pre=f"/public/home/zhangxin/lattice-qcd/laph/di_lambda_c/001/result/beta6.41_mu-0.2295_ms-0.2050_L32x96/A1/corr/corr_dilambdac_P01_M00"
contract.append(read_dat_file(pre,sub,False))
pre=f"/public/home/zhangxin/lattice-qcd/laph/di_lambda_c/001/result/beta6.41_mu-0.2295_ms-0.2050_L32x96/A1/corr/corr_dilambdac_P12_M11"
contract.append(read_dat_file(pre,sub,False))
pre=f"/public/home/zhangxin/lattice-qcd/laph/di_lambda_c/001/result/beta6.41_mu-0.2295_ms-0.2050_L32x96/A1/corr/corr_dilambdac_P23_M22"
contract.append(read_dat_file(pre,sub,False))
pre=f"/public/home/zhangxin/lattice-qcd/laph/di_lambda_c/001/result/beta6.41_mu-0.2295_ms-0.2050_L32x96/A1/corr/corr_dilambdac_P14_M33"
contract.append(read_dat_file(pre,sub,False))
contract=np.array(contract)
N = contract.shape[1]
print(N)
Nt = 40
contract= contract[:,:,:Nt] 
print(contract.shape)
x = np.arange(Nt)
plt.figure()
for p in range(4):
    usedcorr=bootstrap(contract[p],3000)
    R_t = np.roll(usedcorr,1,1) /usedcorr
    mass = np.log(R_t)
    mean=np.mean(mass,0)
    erros = np.std(mass, 0) #* np.sqrt(N - 1)
    plt.errorbar( x=(x + (p - 2) * 0.15),y=mean,yerr=erros, mec=color[p],ecolor=color[p], \
                 marker="o",alpha=0.7,markerfacecolor="none",linestyle="None",capsize=2,capthick=1,label="P=%s" %(Momset[p]),)
plt.legend(loc=1)
# XiccN=1.471103328+0.419889836
# err=0.000402084+0.001878333
# plt.plot([0,40],[XiccN,XiccN],color="black",alpha=0.8,linewidth=1,zorder=1)
# plt.text(40.2,XiccN-err/2,"$\Xi_{cc}N$", fontsize=5.5)
# dila=2*0.946625213	
# err=2*0.001130141
# plt.plot([0,40],[dila,dila],color="black",alpha=0.8,linewidth=1,zorder=0)
# plt.text(40.2,dila-err/3,"$\Lambda_c \Lambda_c$", fontsize=5.5)
plt.title("%s moving frame $M_{eff}$" % (state ))
plt.xlabel("t")
plt.ylabel("$aE$")
plt.ylim(1.8 , 2.2)
plt.xlim(0 , 40)
# plt.show()
plt.savefig(f"{savefile}",dpi=400)
