import sys as sy
from math import ceil
import lsqfit
import gvar as gv
import sympy as sp
import time
from proplot import rc
from analyse_fun import *
import numpy as np
filepath = np.array([
    '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_link_indx%d_U.dat', 
    '/public/home/sush/3pt_distillation/proton/result/%dx%d/Px%dPy%dPz%d/ENV_%d_24x72/corr_uuu_conf%d_2pt.dat',
    '/public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d.iog', 
    '/public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog',
    ])
analyse = data_analyse(
    num_quark=2,
    hadron='proton',
    filepath=filepath,
    alttc=0.1057,
    Nx=24,
    Nt=72,
    P=np.asarray([[0,0,0]]),
    ENV=np.asarray(range(10,101,10)),
    N_start=10000,
    gap=50,
    Ncnfg_data=100,
    Ncnfg_iog=40,
    tsep=np.asarray([8]),
    time_fold=0,
    save_path='/public/home/sush/3pt_distillation/analyse/proton/',
    link_max=0,
    analyse_type='2pt',
    meff_type='log',
    read_type='data',
)
meff_range = [1,1.2]
analyse.meff_2pt('data')
# analyse.meff_2pt('iog')
# analyse.PDF('data')
# analyse.PDF('iog')
# analyse.link_analyse('data')
# analyse.link_analyse('iog')
# analyse.plot_C2pt_ENV([0,0.05])
analyse.plot_meff_ENV(meff_range)
analyse.plot_meff_2pt('data',meff_range)
# analyse.plot_meff_2pt('iog',meff_range)
# analyse.plot_link_ratio([0,1.22])
# analyse.plot_link_C3pt_C3pt([0,1.01])
# analyse.plot_PDF_ENV('U',[0,1.22])
# analyse.plot_link_indx_ENV(1,[0.829,0.86],1)
print('complete')
# A,B = sp.symbols('A B')
# X=[A,B]
# X0=[0.6, 0.01]
# n=3
# # X_0 = [1.1000212053487315, -0.00026742260269824853, -0.0005119883785774843, 7.494130096351822e-06]
# # y = [4.1, 8.9, 18.05, 31.3, 47.5, 68.2, 94.6, 122.1, 156.9, 194]
# y = analyse.link_y['link_data'][0, :, 0, analyse.link_max]
# fit_function = sum((y[x] - (A + B*(x+n)) )**2 for x in range(-n,0,1))# + C*(x)**2 + D*(x)**3 + E*(x)**4 + F*(x)**5 + G*(x)**6 + H*(x)**7)
# X_0 , funvale = min_fun_descent(fit_function,X,X0)
# # X_0 = [0.597626724011288, 0.244526961770759, -0.0728620698330173, 0.0131375815708221, -0.00139672072512821, 8.52722719375410e-5, -2.75641259777434e-6, 3.64834649039291e-8]
# print(y[(20-n):])
# x = sp.symbols('x')
# A = sp.solve((X_0[0] +X_0[1]*x-1.21),x)
# print(A[0]*20+(400-(20*(n-1))))
# for i in range(n):
#     print( float(X_0[0] +X_0[1]*i )) # + X_0[2]*i**2 + X_0[3]*i**3 + X_0[4]*i**4 + X_0[5]*i**5 + X_0[6]*i**6 + X_0[7]*i**7 
# x = sp.symbols('x')
# fun = X_0[0] +X_0[1]*x + X_0[2]*x**2 +X_0[3]*x**3 - 1.2
# sp.solve(fun,x)
# lsqfit
# for tsep_indx in range(N_tsep):
#     t_ary = np.asarray(range(tsep_indx+1))-int(tsep_indx/2)
#     ini_prr = {'C0': '2(0.5)', 'C1': '0(0.5)', 'C3': '0(0.5)', 'C5': '0(0.5)', 'E0': '0.1(0.5)'}
#     # fit_all = np.zeros((data_n,fit_n),dtype=classmethod)
#     fit_parameter = np.zeros(3,dtype=float)
#     def ft_mdls(t_dctnry, p):
#         mdls = {}
#         ts = t_dctnry['C3pt']
#         mdls['C3pt'] = (p['C0'] + p['C1']*(np.exp(-p['E0']*(ts - int(tsep_indx/2))*inp.alttc/fm2GeV) + np.exp(-p['E0']*(ts + int(tsep_indx/2))*inp.alttc/fm2GeV)) + p['C3']*np.exp(-p['E0']*tsep_indx*inp.alttc/fm2GeV)) /(1 + p['C5']*np.exp(-p['E0']*tsep_indx*inp.alttc/fm2GeV))
#         return mdls
#     t_dctnry = {'C3pt': t_ary[:]}
#     data_dctnry = {'C3pt': gv.gvar(Re_ratio_3pt_2pt_mean[0,-1,0], Re_ratio_3pt_2pt_cov[0,-1,0])}
#     fit = lsqfit.nonlinear_fit(data=(t_dctnry, data_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) 
#     fit_parameter[0] = (float)(fit.chi2/fit.dof)
#     fit_parameter[1] = (float)(fit.Q)
#     fit_parameter[2] = (float)(fit.logGBF)
#     print(fit.format(True))
