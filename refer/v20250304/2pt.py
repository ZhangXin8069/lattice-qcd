import numpy as np
import gvar as gv
from scipy.optimize import fsolve
import math
# read in the initial data
data = {}
data['N0'] = 35
data['Nt'] = 72 // 2
data['link_max'] = 0
data['init'] = np.zeros((data['N0'], data['Nt']), dtype=float)
# data['init_3pt'] = np.zeros((data['link_max'], data['N0'], data['Nt']), dtype=float)
for conf in range(35):
    data['init'][conf] = (np.load(f"/public/home/user5/sush_IMP/work/homework/corr/data/proton_conf{conf * 1000 + 10000}.npy").real)[:data['Nt']]
    # for link_indx in range(-data['link_max'], data['link_max'], 1):
    #     data['init_3pt'][link_indx, conf] = (np.load(f"/public/home/user5/sush_IMP/work/homework/corr/data/pion_conf{conf * 1000 + 10000}.npy").real)[:data['Nt']]
        
# data['init'] = data['init']
data['init_normal_mean'] = np.mean(data['init'] / np.max(data['init']), axis = 0)
data['init_normal_err'] = np.std(data['init'] / np.max(data['init']), axis = 0) / np.sqrt(data['N0'] - 1)
data['init_normal_cov'] = np.cov((data['init'] / np.max(data['init'])).T)
def corr_meff_2pt(data_sample:np.ndarray, corr_type:str, analyse_type:str='jackknife'):
    N0 = data_sample.shape[0]
    Nt = data_sample.shape[1]
    
    data_meff = {}
    
    if corr_type == 'cosh':
        data_sample_ini = data_sample.astype(np.float64)
        t_ary = np.array(range(Nt-1))
        ini = 0.2 * np.ones_like(t_ary)
        def eff_mass_eqn(c2pt):
            return lambda E0: (c2pt[:-1]/c2pt[1:]) - np.cosh(E0*(Nt-t_ary)) / np.cosh(E0*(Nt-(t_ary+1)))
        def fndroot(eqnf,ini):
            sol = fsolve(eqnf,ini, xtol=1e-5)
            return sol
        data_meff['sample'] = np.array([fndroot(eff_mass_eqn(c2pt),ini) for c2pt in data_sample_ini[:,:]])
        data_meff['mean'] = np.mean(data_meff['sample'], axis=-2)
        if analyse_type == 'Jackknife':
            data_meff['err'] = np.sqrt(N0 - 1) * np.std(data_meff['sample'], axis=-2)
        elif analyse_type == 'Bootstrap':
            data_meff['err'] = np.std(data_meff['sample'], axis=-2)
        else:
            raise print("don't have the analyse type" )
        
    elif corr_type == 'log':
        data_meff['mean'] = np.mean(data_sample, axis = 0)
        data_meff['mean'] = np.array(np.log(np.real(data_meff['mean'][...,:-1])/np.real(data_meff['mean'][...,1:])))
        data_meff['sample'] = np.array(np.log(np.real(data_sample[...,:-1])/np.real(data_sample[...,1:]))) # * (fm2GeV / alttc)
        
        if analyse_type == 'Jackknife':
            data_meff['err'] = np.std(data_meff['sample'], axis=-2) * np.sqrt(N0 - 1)
        elif analyse_type == 'Bootstrap':
            data_meff['err'] = np.std(data_meff['sample'], axis=-2)
        else:
            raise print("don't have the analyse type") 
        
    else:
        raise print("don't have the corr type") 
    
    return data_meff
# JackknifeJackknife: 
# 样本平均值是固定的，缩小样本的大小，减小方差；
# 但是不适用于样本量比较小的情况，会导致数据分析出现偏差；并且不适用于非线性的分析
def Jackknife(data:np.ndarray, analyse:bool=False) -> dict:
    N0 = data.shape[0]
    N1 = data.shape[1]
    
    data_analyse = {}
    data_sum = np.sum(data, axis = 0)
    
    data_analyse['analysed_sample'] = - (data - data_sum) / (N0 - 1) # solve the jackknife sample
    if analyse == True:
        data_analyse['analysed_mean'] = np.mean(data, axis = 0) # solve the mean value of data   
        data_analyse['analysed_err'] = np.sqrt(N0 - 1) * np.std(data_analyse['analysed_sample']) # solve the standard deviation of data
        data_analyse['analysed_cov'] = np.cov(data_analyse['analysed_sample'].T)
        
    return data_analyse
#Bootstrap: 随机抽取N次原样本中的M个数据，
def Bootstrap(data:np.ndarray, N:int, M:int, analyse:bool=False) -> dict:
    N0 = data.shape[0]
    N1 = data.shape[1]
    
    if M > N1:
        raise print(f'M must less than N0. M{M}, N0{N0}.')
    
    if N > math.factorial(N0) / (math.factorial(N0 - M) * math.factorial(M)):
        raise print(f'N must less than C{N0}{N}. M{M}, N0{N0}.')
    
    accept_ratio = M / N0
    data_sample = np.zeros((N, N1))
    data_analyse = {}
    for i in range(N):
        random_array = np.random.random((N0))
        while ((random_array - accept_ratio) < 0).all():
            random_array = np.random.random((N0))
        data_sample[i,:] = np.sum(data[random_array > accept_ratio, :], axis = 0) / np.sum(random_array > accept_ratio)
    
    data_analyse['analysed_sample'] = data_sample
    if analyse == True:
        data_analyse['analysed_mean'] = np.mean(data_sample, axis = 0)
        data_analyse['analysed_err'] = np.std(data_sample, axis = 0)
        data_analyse['analysed_cov'] = np.cov(data_sample.T)
    return data_analyse
import lsqfit
# data.update(Bootstrap(data['init'], N=1000, M = 30, corr_type=''))
data.update(Jackknife(data['init']))
data_meff = corr_meff_2pt(data['analysed_sample'], corr_type='log', analyse_type='Jackknife')
data['analysed_mean'] = data_meff['mean']
data['analysed_err']  = data_meff['err']
data['analysed_cov'] = np.cov(data_meff['sample'].T)
X = np.arange(data['Nt'])
print(data.keys())
ini_prr = {'A0': '0.6(1)', 'E0': '0.5(1)', 'A1': '0.1(1)', 'E1': '0(1)'}
# fit_all = np.zeros((data_n, fit_n),dtype=classmethod)
fit_parameter = np.zeros(3,dtype=float)
def ft_mdls(t_dctnry, p):
    mdls = {}
    ts = t_dctnry['C3pt']
    mdls['C3pt'] = (p['A0'] *  np.exp(- p['E0'] * ts) * (1 + p['A1'] *  np.exp(- p['E1'] * ts)))
    return mdls
t_dctnry = {'C3pt': X[2:13]}
data_dctnry = {'C3pt': gv.gvar(data['init_normal_mean'][2:13], data['init_normal_cov'][2:13, 2:13])}
fit = lsqfit.nonlinear_fit(data=(t_dctnry, data_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) 
fit_parameter[0] = (float)(fit.chi2/fit.dof)
fit_parameter[1] = (float)(fit.Q)
fit_parameter[2] = (float)(fit.logGBF)
print(fit.format(True))
# fitted function values
import matplotlib.pyplot as plt
t_ary = fit.data[0]['C3pt']
t_lst = np.arange(0.2,15,0.1)
data_fit_fcn_gvar = fit.fcn({'C3pt':t_lst}, fit.p)['C3pt']
data_fit_mean = np.array([c2.mean for c2 in data_fit_fcn_gvar])
data_fit_err = np.array([c2.sdev for c2 in data_fit_fcn_gvar])
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
ax.errorbar(X[:16], data['init_normal_mean'][:16], yerr=data['init_normal_err'][:16], fmt = 'ro',alpha=0.5, capsize=3.5, capthick=1.5, label='data', linestyle='none', elinewidth=2) 
ax.plot(t_lst, data_fit_mean, color="b", label="best fit") 
ax.fill_between(t_lst, data_fit_mean - data_fit_err, data_fit_mean + data_fit_err,  alpha=0.3) 
ax.set_ylim([0,1])
plt.xlabel('t')
plt.ylabel('$C_{2pt}$')
ax.legend(loc='upper center', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, borderpad=0.3, \
            ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=1.5)
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
fig.show
plt.rcParams.update({'font.size':25})
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
ax.set_ylim([0.3, 0.8])
ax.set_xlim([0, 20])
ax.set_xlabel('%s'%('t'))
ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
ax.errorbar(np.arange(data['Nt'])[:-1], data['analysed_mean'], yerr=data['analysed_err'], fmt = '*', alpha=0.5, capsize=3.5, capthick=1.5, label='data', linestyle='none', elinewidth=2) # fmt = 'bs'
plt.legend()
plt.show
