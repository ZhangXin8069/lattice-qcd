from include import *
import numpy as np
filepath = np.array([ 
    '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_link_indx%d_U_1000.dat',
    '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_link_indx%d_U_1000_peram_phase.dat', 
    '/public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d/conf%d/corr_ud_2pt_1000.dat', 
    '/public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/proton/nucleon_U_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d.iog', 
    '/public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/proton/nucleon_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog',
    ])
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
analyse.plot_link_C3pt_C3pt([0,1.21],num=1)
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
