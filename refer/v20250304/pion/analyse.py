import sys as sy
from math import ceil
import lsqfit
import gvar as gv
import sympy as sp
import time
from proplot import rc
from analyse_fun import *
import numpy as np
import scipy.optimize
import math
    # '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_link_indx%d_U_1000_nosmear.dat',
filepath = np.array([ 
    '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_link_indx%d_U_1000.dat',
    '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_link_indx%d_U_1000_peram_phase.dat', 
    '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d/conf%d/corr_ud_2pt_1000.dat', 
    '/public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/proton/nucleon_U_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d.iog', 
    '/public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/proton/nucleon_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog',
    ])
# filepath = np.array([
#     '/public/home/sush/share_work/chroma/beta7.0_mu-0.1600_ms-0.1450_L%dx%d/mom_grid_source/new/pion_2pt_Px%dPy%dPz%d_ENV%d_conf1300_tsep-1_mass-0.1450.iog'
#     ])
analyse = data_analyse(
    num_data=3,
    hadron='pion',
    filepath=filepath,
    alttc=0.1053,
    Nx=24,
    Nt=72,
    time_fold=True,
    P=np.asarray([[0,0,0]]),
    ENV=np.asarray(range(998,1000,1)),
    N_start=10050,
    gap=50,
    Ncnfg_data=4,
    Ncnfg_iog=1,
    tsep=np.asarray([10]),
    link_max=10,
    save_path='/public/home/sush/3pt_distillation/analyse/pion/1000',
    analyse_type='ratio',
    meff_type='cosh',
    read_type='iog',
)
meff_range = [0.285,0.30]
# analyse.meff_2pt('data')
# analyse.meff_2pt('iog')
# analyse.PDF('data', 'U', link_fold=True)
# analyse.PDF('data', 'D', link_fold=True)
analyse.PDF('iog', 'U', link_fold=True)
# analyse.link_analyse('data','U')
# analyse.link_analyse('data','D')
analyse.link_analyse('iog','U')
# print(analyse.link_y['link_data_D'][:,:,:,0])
print(analyse.link_y['link_iog_U'][:,:,:,:])
# A = analyse.link_y['link_data_U'][...,10]
# for i in range(A.shape[1]-1):
#     print(A[0,i+1] - A[0,i])
# print(analyse.link_y['link_data_U'][...,10])
# analyse.plot_C2pt_ENV([0,0.05])
# analyse.plot_meff_ENV(meff_range)
# analyse.plot_meff_2pt('data',meff_range)
# analyse.plot_meff_2pt('iog',meff_range)
# analyse.plot_link_ratio([0,1.25], num=1)
# analyse.plot_link_C3pt_C3pt([0,1.21],num=1)
# analyse.plot_PDF_ENV('U',[1.1,1.21])
# analyse.plot_link_indx_ENV(4, [0.25,0.3], normal=0)
print('complete')
# X = analyse.ENV
# Y = analyse.link_y['link_data'][0, :, 0, analyse.link_max]
# def func(x,a,b,c,d,e):
#     return a*(np.log(b*x**2 + c*x**1 + d*x**0) ) + e 
# # def func(x,a,b,c,d):
# #     return a*(np.log10(b*x + c)) + d 
# def func2(x,a,b):
#     return a*x+b
# peram_1_log, peram_2_log = scipy.optimize.curve_fit(func,X,Y)
# peram_1_liner, peram_2_liner = scipy.optimize.curve_fit(func2, X[-2:], Y[-2:])
# print(Y)
# print('log')
# print(np.asarray([func(i, peram_1_log[0], peram_1_log[1], peram_1_log[2], peram_1_log[3], peram_1_log[4]) for i in range(50,3001,50)]).reshape(6,10))
# n1 = sp.symbols('n1')
# function_data = sp.solve(peram_1_log[0]*(sp.log(peram_1_log[1]*n1**2 + peram_1_log[2]*n1**1 + peram_1_log[3]*n1**0) ) + peram_1_log[4] - 1.21, n1)
# print(func(40000, peram_1_log[0], peram_1_log[1], peram_1_log[2], peram_1_log[3], peram_1_log[4]))
# print(function_data)
# print('liner')
# print(np.asarray([func2(i, peram_1_liner[0], peram_1_liner[1]) for i in range(50,3001,50)]).reshape(6,10))
# print(func2(40000, peram_1_liner[0], peram_1_liner[1]))
# n2 = sp.symbols('n2')
# function_data = sp.solve(peram_1_liner[0]*n2 + peram_1_liner[1] - 1.21, n2)
# print(function_data)
# function_data = sp.solve(peram_1[0] * n + peram_1[1] - 1.2, n)
'''
A,B,C,D,E,F,G = sp.symbols('A B C D E F G')
X=np.asarray([A,B,C,D,E,F,G])
X0=np.asarray([ 5.86773431e-02, 3.30134565e-01, -8.94173319e-01, 5.69510929e-01, 3.36807580e-06, 10, 9.17546155e-01])
n=20
# # # X_0 = [1.1000212053487315, -0.00026742260269824853, -0.0005119883785774843, 7.494130096351822e-06]
# # # y = [4.1, 8.9, 18.05, 31.3, 47.5, 68.2, 94.6, 122.1, 156.9, 194]
y = analyse.link_y['link_data'][0, :, 0, analyse.link_max]
fit_function = sum((y[x] - (A*sp.log(B*x**3+C*x**2+D*x**1+E,F)+G) )**2 for x in range(n))# + C*(x)**2 + D*(x)**3 + E*(x)**4 + F*(x)**5 + G*(x)**6 + H*(x)**7)
X_0 , funvale = min_fun_descent(fit_function, X, X0, type='log')
# # # X_0 = [0.597626724011288, 0.244526961770759, -0.0728620698330173, 0.0131375815708221, -0.00139672072512821, 8.52722719375410e-5, -2.75641259777434e-6, 3.64834649039291e-8]
print(y[:])
# x = sp.symbols('x')
# A = sp.solve((X_0[0] +X_0[1]*x-1.21),x)
# print(A[0]*20+(400-(20*(n-1))))
for i in range(n):
    print( float((X_0[0]*(np.log(X_0[1]*i+X_0[3])/np.log(X_0[2]))+X_0[4]))) # + X_0[2]*i**2 + X_0[3]*i**3 + X_0[4]*i**4 + X_0[5]*i**5 + X_0[6]*i**6 + X_0[7]*i**7 
x = sp.symbols('x')
fun = X_0[0] +X_0[1]*x + X_0[2]*x**2 +X_0[3]*x**3 - 1.2
sp.solve(fun,x)
lsqfit
for tsep_indx in range(N_tsep):
    t_ary = np.asarray(range(tsep_indx+1))-int(tsep_indx/2)
    ini_prr = {'C0': '2(0.5)', 'C1': '0(0.5)', 'C3': '0(0.5)', 'C5': '0(0.5)', 'E0': '0.1(0.5)'}
    # fit_all = np.zeros((data_n,fit_n),dtype=classmethod)
    fit_parameter = np.zeros(3,dtype=float)
    def ft_mdls(t_dctnry, p):
        mdls = {}
        ts = t_dctnry['C3pt']
        mdls['C3pt'] = (p['C0'] + p['C1']*(np.exp(-p['E0']*(ts - int(tsep_indx/2))*inp.alttc/fm2GeV) + np.exp(-p['E0']*(ts + int(tsep_indx/2))*inp.alttc/fm2GeV)) + p['C3']*np.exp(-p['E0']*tsep_indx*inp.alttc/fm2GeV)) /(1 + p['C5']*np.exp(-p['E0']*tsep_indx*inp.alttc/fm2GeV))
        return mdls
    t_dctnry = {'C3pt': t_ary[:]}
    data_dctnry = {'C3pt': gv.gvar(Re_ratio_3pt_2pt_mean[0,-1,0], Re_ratio_3pt_2pt_cov[0,-1,0])}
    fit = lsqfit.nonlinear_fit(data=(t_dctnry, data_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) 
    fit_parameter[0] = (float)(fit.chi2/fit.dof)
    fit_parameter[1] = (float)(fit.Q)
    fit_parameter[2] = (float)(fit.logGBF)
    print(fit.format(True))
'''
