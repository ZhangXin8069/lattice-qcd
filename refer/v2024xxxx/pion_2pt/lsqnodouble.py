#!/public/home/xinghy/anaconda3-2023.03/bin/python
# test file to check whether first term has problem
import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
import lsqfit
import gvar as gv
from html.entities import entitydefs
# import seaborn as sns
# from prettytable import PrettyTable
import matplotlib.patches as mpatches
#import pandas as pd
state = "F32P30"
object = "Pion"  # "$\overline{D}^{0*}$" $D_s^+$
parity = "uu_gamma5"
savename = "D"
#'Ds$^+$$\overline{D}$$^{0*}$ - k$^1$ eigenval0'# without c$\overline{c}$'
channel = ""
Nt0 = 96
N = 500
eigenval_id = 0
Nt = 96
px = 0
py = 0
pz = 0
Na = 0.077
start = 24  # start t
beta = 35  # end t
t0 = 24  # fit t
deltat = 3  # fit section width
# infile=fileinput.input(files=('/public/home/changx/meson_run1110/input.sh'))
# for line in infile:
#   tmp=line.split()
#   if(tmp[0]=='Nt'):
#     Nt=int(tmp[1])
#   if(tmp[0]=='Px'):
#     Px=int(tmp[1])
#   if(tmp[0]=='Py'):
#     Py=int(tmp[1])
#   if(tmp[0]=='Pz'):
#     Pz=int(tmp[1])
xmin = 5
xmax = 48
ymin = 0.300
ymax = 0.315
# file = "/public/home/changx/meson_run1110/result_uu_%s%s%s" % (px,py,pz)
file = "/public/home/changx/meson_run1110/result_uu_{px}{py}{pz}"
print(file)
exit(1)
# contract = np.load(f"{file}/corr_us_g5D3_n168.npy")
contract = []
for i in range(N):
    # print(f"{file}/corr_{parity}_Px%sPy%sPz%s_conf%s.dat" % (px, py, pz, 1000 + i * 50))
    # print("{}/corr_{}_Px{}Py{}Pz{}_conf{}.dat".format(file,parity,px, py, pz, 1000 + i * 50))
    # print("{file}/corr_{parity}_Px{px}Py{py}Pz{pz}_conf{1000 + i * 50}.dat")
    try:
        f = open(
            f"{file}/corr_{parity}_Px{px}Py{py}Pz{pz}_conf{1000 + i * 50}.dat"
            # "{}/corr_{}_Px{}Py{}Pz{}_conf{}.dat".format(file,parity,px, py, pz, 1000 + i * 50)
        )  # Px0Py0Pz0_conf
        real = []
        for line in f:
            tmp = line.split()
            real.append(float(tmp[1]))
        del real[0]
        contract.append(real)
        f.close()
    except:
        continue
contract = np.array(contract)
Nt = int(Nt / 2) + 1
contract = contract[:, :Nt]
contract_sum = np.sum(contract, 0)
average_jk = ((contract_sum) - contract) / (N - 1)
N = average_jk.shape[0]
print("N=", N)
sum_average = np.mean(average_jk, 0)
sq_sum = 0
jk_err = np.zeros(Nt)
for t in range(Nt):
    sq_sum = 0
    for i in range(N):
        sq_sum += (sum_average[t] - average_jk[i][t]) ** 2
    jk_err[t] = float(np.sqrt(sq_sum * (N - 1) / N))
jk_err = jk_err / sum_average[0]
average_jk = average_jk / sum_average[0]
sum_average = sum_average / sum_average[0]
M_a = average_jk
M_b = sum_average
M = np.zeros((Nt, Nt))
M_ij = np.zeros((Nt, Nt))
for k in range(N):
    for i in range(Nt):
        for j in range(Nt):
            M_ij[i, j] = (M_a[k][i] - M_b[i]) * (M_a[k][j] - M_b[j])
    M += M_ij
M_dig = np.tril(M)
M_dig = np.triu(M_dig)
# print(M_dig)
# print (M)
# Nt=int((Nt-1)*2)
def efunc(x, p):
    return p["a0"] * (np.exp(-p["E0"] * (Nt0 - x)) + np.exp(-p["E0"] * x))
# ——————————————————————————拟合corr fitting
# Nt=int(Nt/2+1)
pr_arr_corr = np.zeros([N, 3])
pr_arr = np.zeros([N, 3])  # uncorr
time = np.arange(Nt)
# t1_all=np.arange(14,21)
# t2_all=np.zeros(7)+39
chi_square = np.zeros([1, deltat])
mass_mean = np.zeros([1, deltat])
error_mean = np.zeros([1, deltat])
for t2 in range(1):
    for t1 in range(start, start + deltat):
        print("t1=", t1, "t2=", t2 + beta)
        p0 = dict(a0='0.3', E0='0.12')
        #prior = {"a0":gv.gvar(0.3, 1.0e5),"E0":gv.gvar(0.3, 5.0)}
        gv_corr = gv.gvar(np.mean(average_jk,0), M)
        fit_mean = lsqfit.nonlinear_fit(data=(time[t1 : t2 + beta + 1], gv_corr[t1 : t2 + beta + 1]),\
            fcn=efunc,svdcut=1e-12,p0=p0)
        chi_mean = fit_mean.chi2/fit_mean.dof
        for fid in range(N):
            # p0 = dict(a0='0.3', E0='0.9')   #->  {"a0":0.3,"E0":0.9}
            # 
            # prior["a0"] = gv.gvar(0.3, 1.0e5)
            # prior["E0"] = gv.gvar(0.3, 5.0)
            gv_corr = gv.gvar(average_jk[fid], M)
            fit = lsqfit.nonlinear_fit(
                data=(time[t1 : t2 + beta + 1], gv_corr[t1 : t2 + beta + 1]),
                fcn=efunc,
                p0=p0
                #prior=prior,
            )  # 拟合
            pr_corr = fit.p
            pr_arr_corr[fid][0] = pr_corr["E0"].mean
            pr_arr_corr[fid][1] = pr_corr["a0"].mean
            pr_arr_corr[fid][2] = fit.chi2 / (fit.dof - 2)  #if prior 
            #pr_arr_corr[fid][2] = fit.chi2 / (fit.dof)  #if p0
        # print(fit.chi2/fit.dof)
        pr_arr_corr = np.array(pr_arr_corr)
        pr_aver_corr = np.array([0.0, 0.0, 0.0])
        for i in range(N):
            pr_aver_corr += pr_arr_corr[i]
        pr_aver_corr = pr_aver_corr / N
        print("E", pr_aver_corr[0], "dof:", fit.dof, "chi:", chi_mean)
        #    print('E1_mean=',pr_aver_corr[1]+pr_arr_corr[fid][0])#拟合质量的平均值
        #    print('E_mean=',pr_aver_corr[0])
        mass_mean[t2][t1 - start] = pr_aver_corr[0]
        chi_square[t2][t1 - start] = chi_mean
        mass_err_corr0 = 0
        sq_sum_corr0 = 0
        for i in range(N):
            sq_sum_corr0 += (pr_aver_corr[0] - pr_arr_corr[i][0]) ** 2
            mass_err_corr0 = np.sqrt((N - 1) / (N) * sq_sum_corr0)
        #    print('E_error=',mass_err_corr0)
        #    print('E1_error=',mass_err_corr)  #拟合质量的jkerror
        error_mean[t2][t1 - start] = mass_err_corr0
    # print(t1_all)
# print(chi_square)
data_M = np.zeros((N, Nt))  # 拟合出的数据矩阵
data_sum_ave = np.zeros(Nt)
for j in range(Nt):
    for i in range(N):
        data_M[i][j] = pr_arr_corr[i][1] * (
            np.exp(-pr_arr_corr[i][0] * (Nt - time[j]))
            + np.exp(-pr_arr_corr[i][0] * time[j])
        )
# print(data_M)
for i in range(N):
    data_sum_ave += data_M[i]
data_sum_ave = data_sum_ave / N  # 数据均值
sq_sum = 0
data_M_err = np.zeros(Nt)
for t in range(Nt):
    sq_sum = 0
    for i in range(N):
        sq_sum += (data_sum_ave[t] - data_M[i][t]) ** 2
    data_M_err[t] = np.sqrt(sq_sum * (N - 1) / N)  # 拟合数据jk误差
# print (pr_aver[0],pr_aver[1])
# fig1=plt.figure()
# plt.plot(time,data_M[0],color='red',label='dependent fn')
# plt.errorbar(x=(time-0.1),y=data_sum_ave,yerr=data_M_err,color='indianred',elinewidth=2,capthick=1,label='corr fitting',marker='.')
# plt.errorbar(x=(time-0.1),y=np.log(data_sum_ave/(average_jk[0][0]-average_jk[0][32])*(np.e-1)+1),yerr=np.log(data_M_err/(average_jk[0][0]-average_jk[0][32])*(np.e-1)+1),color='indianred',elinewidth=2,capthick=1,label='corr fitting',linestyle='none',marker='.')
# plt.legend(loc='upper right')
# -------------------------seaborn.heatmap
# print('t0=',t0,'delta=',deltat,'beta=',beta)
# print('chi2/dof=',chi_square)
print("mass=", mass_mean)
print("error=", error_mean)
# twostatefit = PrettyTable(
#     [
#         "start timeslice(end=" + str(beta) + ")",
#         "chi2/dof",
#         "mass mean",
#         "fitting error",
#     ]
# )
# for i in range(deltat):
#     twostatefit.add_row(
#         [
#             np.arange(deltat)[i] + start,
#             chi_square[0][i],
#             mass_mean[0][i],
#             error_mean[0][i],
#         ]
#     )
# print(twostatefit)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.bar(
    x=np.arange(deltat) + start,
    height=chi_square[0],
    label="$\chi^2/d.o.f.$",
    color="cornflowerblue",
)
ax1.set_yscale("log")
ax1.set_ylim(0.1, 1000)
ax2.set_ylabel("chi2/dof", labelpad=50)
mass_mean = mass_mean / Na * 0.1974
error_mean = error_mean / Na * 0.1974
lns2 = ax2.errorbar(
    x=np.arange(deltat) + start,
    y=mass_mean[0],
    yerr=error_mean[0],
    ecolor="red",
    linestyle="none",
    mec="red",
    marker="o",
    alpha=0.7,
    markerfacecolor="none",
    capsize=2,
    capthick=1,
    label="$M_{eff}$",
)
for i in range(deltat):
    ax1.text(start+i, chi_square[0,i]+0.1, str(round(chi_square[0,i], 2)), ha='center')
ax1.set_xlabel("start timeslice")
ax1.set_ylabel("$M_{eff}$ fitting result/Gev", labelpad=40)
# fig.legend(loc='upper right', bbox_to_anchor=(0.34, 1), bbox_transform=ax1.transAxes)
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
ax1.yaxis.tick_right()
ax2.yaxis.tick_left()
ax2.set_ylim(mass_mean[0,t0-start]-30*error_mean[0,t0-start],mass_mean[0,t0-start]+10*error_mean[0,t0-start])
plt.title(
    "%s %s P%s%s%s $M_{eff}$ and $\chi^2$(t=%s)" % (state, object, px, py, pz, beta)
)
plt.savefig(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/test1.jpg" , dpi=400)
# plt.show()
# -------------------------------effective mass
t = 0
R_t = 0
meson_mass_average = np.zeros(Nt - 2)  # ×ÜÖÊÁ¿µÄÆ½¾ù
meson_mass = 0
for i in range(1, Nt - 1):
    R_t = (sum_average[i - 1] + sum_average[i + 1]) / (2 * sum_average[i])
    meson_mass = np.arccosh(R_t) / Na * 0.1974
    meson_mass_average[i - 1] = meson_mass
# mason_mass_average=np.array(meson_mass_average)
print(meson_mass_average.shape)
meson_mass_jk = []  # ½«mass_average_notsumÐ´Èë¸Ãlist
mass_notsum = 0
mass_average_notsum = []  # ÉÙÒ»¸öÊý¾ÝµÄÖÊÁ¿Æ½¾ùÖµ
# print(contract)
for i in range(N):
    for k in range(1, Nt - 1):
        R_t = (average_jk[i][k - 1] + average_jk[i][k + 1]) / (2 * average_jk[i][k])
        mass_notsum = np.arccosh(R_t) / Na * 0.1974
        mass_average_notsum.append(mass_notsum)
    meson_mass_jk.append(mass_average_notsum)  # N*62Î¬¶È
    mass_average_notsum = []  # ÉÙÒ»¸öÊý¾ÝµÄÖÊÁ¿Æ½¾
jkerros = []
average_jk_notsum = []
for t in range(Nt - 2):
    ave_diff_square = 0  # Æ½·½²î
    #  average_jk_notsum=[] #2´Îjk
    for i in range(N):
        #    average_jk_notsum.append((mass_jk_sum[t]- pion_mass_jk[i][t])/N)
        #    ave_diff_square += (mass_jk_average[t]- average_jk_notsum[i])**2
        ave_diff_square += (meson_mass_average[t] - meson_mass_jk[i][t]) ** 2  # Çó
    jkerros.append(np.sqrt((N - 1) / (N) * ave_diff_square))
jkerros = np.array(jkerros)
print(jkerros.shape)
# -----------------------picture
end = beta
E_mean = mass_mean[0][t0 - start]
E_error = error_mean[0][t0 - start]
time = np.arange(Nt - 2) + 1
fig = plt.figure()
# plt.plot([t0,t0],[0,10],color='black',alpha=0.3)
plt.errorbar(
    x=time,
    y=meson_mass_average,
    yerr=jkerros,
    ecolor="cornflowerblue",
    linestyle="none",
    mec="cornflowerblue",
    marker="o",
    alpha=0.7,
    markerfacecolor="none",
    capsize=2,
    capthick=1,
    label="$M_{eff}$",
)
plt.legend(loc='upper right')
left, bottom, width, height = (t0, (E_mean - E_error), end - t0, 2 * E_error)
rect = mpatches.Rectangle(
    (left, bottom),
    width,
    height,
    # fill=False,
    alpha=0.4,
    facecolor="red",
    label="fit band",
)
plt.gca().add_patch(rect)
# plt.text(t0-0.1, pion_mass_average[t0-start], 't0', ha='center', va='bottom', fontsize=9)
plt.legend(loc='upper right')
plt.title("%s %s P%s%s%s" % (state, object, px, py, pz))
plt.xlabel("t")
plt.ylabel("$M_{eff}$/Gev")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
# plt.yticks([0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
# plt.xticks([5,10,15,20,25])
plt.show(block=True)
plt.savefig(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/test.jpg" , dpi=400)
