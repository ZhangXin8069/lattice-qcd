#!/public/home/xinghy/anaconda3-2023.03/bin/python

# test file to check whether first term has problem

import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt

# import copy
import lsqfit
import pandas as pd
import gvar as gv
import scipy as sp

# from html.entities import entitydefs
# import seaborn as sns
from prettytable import PrettyTable
import matplotlib.patches as mpatches

from analyse_fun import *

# state = "F32P30"
# object = "Ds$^+$$\overline{D}$$^{0*}$-K$_1$ derivative"

Nt = 64
Nt0 = 64
# N = 43
N = 30
px = 0
py = 0
pz = 0
Na = 0.04
start = 15  # start t
beta = 30  # end t
t0 = 7  # fit t
deltat = 11  # fit section width
mom = 0
# (filepath, Nx, Nt, P, ENV, N_stare, gap, Ncnfg, tsep_array, link_max, type)
# inputfile = "/public/home/donghx/Lattice/Example/result/"
inputfile = np.asarray(["/public/home/sush/share_work/chroma/beta7.0_mu-0.1600_ms-0.1450_L%dx%d/wall_source/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.1450.iog"])
contract = read_iog(
    inputfile,
    48,
    64,
    np.asarray([[0,0,0]]),
    np.asarray([-1]),
    800,
    20,
    30,
    np.asarray([-1]),
    0,
    '2pt'
    )
print(np.shape(contract))
contract=(contract[...,0]).reshape(30,64)
N = contract.shape[0]  # your final number of samples
print("N=", N)

contract_sum = np.sum(contract, 0)
average_jk = ((contract_sum) - contract) / (N - 1)
sum_average = np.mean(contract, 0)

meson_mass_average = (
    np.arccosh((np.roll(sum_average, -1) + np.roll(sum_average, 1)) / (2 * sum_average))
    / Na
    * 0.1974
)
meson_mass_jk = (
    np.arccosh(
        (np.roll(average_jk, -1, 1) + np.roll(average_jk, 1, 1)) / (2 * average_jk)
    )
    / Na
    * 0.1974
)
jkerros = np.std(meson_mass_jk, 0) * np.sqrt(N - 1)
print(meson_mass_average)

# meson_mass_average = np.log(sum_average / np.roll(sum_average, -1)) / Na * 0.1974

# meson_mass_jk = np.log(average_jk / np.roll(average_jk, -1)) / Na * 0.1974

# jkerros = np.std(meson_mass_jk, 0) * np.sqrt(N - 1)


# ------------------------fitting
M_ij = np.zeros((Nt, Nt))

for n in range(N):
    for i in range(Nt):
        M_ij[i] += (average_jk[n, i] - sum_average[i]) * (average_jk[n] - sum_average)

M_dig = np.tril(M_ij)
M_dig = np.triu(M_dig)


def efunc(x, p):
    return p["a0"] * (np.exp(-p["E0"] * (Nt0 - x)) + np.exp(-p["E0"] * x))


pr_arr_corr = np.zeros([N, 3])
time = np.arange(Nt)
chi_square = np.zeros([1, deltat])
mass_mean = np.zeros([1, deltat])
error_mean = np.zeros([1, deltat])
for t2 in range(1):  # right side
    for t1 in range(start, start + deltat):  # left side
        print("t1=", t1, "t2=", t2 + beta)
        for fid in range(N):
            # p0 = dict(a0='0.3', E0='0.9')
            prior = {}
            prior["a0"] = gv.gvar(0.3, 1.0e5)
            prior["E0"] = gv.gvar(0.3, 5.0)

            gv_corr = gv.gvar(average_jk[fid], M_ij)
            fit = lsqfit.nonlinear_fit(
                data=(time[t1 : t2 + beta], gv_corr[t1 : t2 + beta]),
                fcn=efunc,
                prior=prior,
            )
            pr_corr = fit.p
            pr_arr_corr[fid][0] = pr_corr["E0"].mean / Na * 0.1974
            pr_arr_corr[fid][1] = pr_corr["a0"].mean
            pr_arr_corr[fid][2] = fit.chi2 / (fit.dof - 2.0)

        pr_arr_corr = np.array(pr_arr_corr)
        pr_aver_corr = np.mean(pr_arr_corr, 0)

        print(fit.p, "chis = ", fit.chi2 / (fit.dof - 2.0))

        mass_mean[t2][t1 - start] = pr_aver_corr[0]
        chi_square[t2][t1 - start] = pr_aver_corr[2]

        mass_err_corr0 = np.std(pr_arr_corr, 0)[0] * np.sqrt(N - 1)
        error_mean[t2][t1 - start] = mass_err_corr0

data_M = np.zeros((N, Nt))  # fitting data N*Nt
data_sum_ave = np.zeros(Nt)
time = np.arange(Nt)
for n in range(N):
    data_M[n] = pr_arr_corr[n, 1] * (
        np.exp(-pr_arr_corr[n, 0] * (Nt - time)) + np.exp(-pr_arr_corr[n, 0] * time)
    )
# print(data_M)

data_sum_ave = np.mean(data_M, 0)  # data mean
data_M_err = np.std(data_M, 0) * np.sqrt(N - 1)  # data error
# --------------pretty table
print("mass=", mass_mean)
print("error=", error_mean)
twostatefit = PrettyTable(
    ["start timeslice(end=" + str(beta) + ")", "chi2/dof", "mass mean", "fitting error"]
)
for i in range(deltat):
    twostatefit.add_row(
        [
            np.arange(deltat)[i] + start,
            chi_square[0][i],
            mass_mean[0][i],
            error_mean[0][i],
        ]
    )
print(twostatefit)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.bar(
    x=np.arange(deltat) + start,
    height=chi_square[0],
    label="chi2/dof",
    color="cornflowerblue",
)
ax1.set_yscale("log")
ax1.set_ylim(0.05, 50)
ax2.set_ylabel("chi2/dof", labelpad=30)
# plt.legend(loc=0)
# -----------------------picture
end = beta
E_mean = mass_mean[0][t0 - start]
E_error = error_mean[0][t0 - start]

# Nt=int(Nt/2+1)
time = np.arange(Nt)

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
    label="P%s%s%s Meff" % (px, py, pz),
)
# plt.legend(loc=0)
left, bottom, width, height = (t0, (E_mean - E_error), end - t0, 2 * E_error)
rect = mpatches.Rectangle(
    (left, bottom),
    width,
    height,
    # fill=False,
    alpha=0.4,
    facecolor="red",
    label="correlation fitting band",
)
plt.gca().add_patch(rect)
# plt.text(t0-0.1, pion_mass_average[t0-start], 't0', ha='center', va='bottom', fontsize=9)
plt.legend(loc=0)
# plt.title("%s %s P%s%s%s" % (state, object, px, py, pz))
plt.title("mom_%s  (P%s%s%s)" % (mom, px, py, pz))
plt.xlabel("t")
plt.ylabel(r"m$_{eff}$ (P$_{%s%s%s}$)" % (px, py, pz))
plt.xlim(0, Nt)
plt.ylim(0.6, 1.0)
# plt.yticks([0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
plt.xticks([0, 5, 10, 15, 20])
plt.show(block=True)
plt.savefig(
    "./beta7.0_mu-0.1600_ms-0.1450_L48x64/em_mom%s_P%s%s%s_nopol.png"
    % (mom, px, py, pz),
    dpi=400,
)
