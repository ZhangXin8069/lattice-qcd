import math
import numpy as np
# read in the initial data
# JackknifeJackknife: 
# 样本平均值是固定的，缩小样本的大小，减小方差；
# 但是不适用于样本量比较小的情况，会导致数据分析出现偏差；并且不适用于非线性的分析
def Jackknife(data:np.ndarray, name:str='analysed', analyse:bool=False) -> dict:
    N0 = data.shape[-2]
    N1 = data.shape[-1]
    
    shape = list(data.shape)
    
    shape_mean_err = shape.copy()
    shape_mean_err[-2] = 1
    
    shape_cov = shape.copy()
    shape_cov[-2] = shape_cov[-1]
    
    N_sum = int(np.prod(shape[:-2]))
    data.reshape(N_sum, N0, N1)
    data_cov = np.zeros((N_sum, N1, N1))
    
    data_analyse = {}
    data_sum = np.sum(data, axis = -2)
    data_analyse[f'{name}_sample'] = - (data - data_sum[...,np.newaxis,:]) / (N0 - 1) # solve the jackknife sample
    
    if analyse == True:
        data_analyse[f'{name}_mean'] = np.squeeze((np.mean(data, axis = -2)).reshape(shape_mean_err), axis=-2) # solve the mean value of data   
        data_analyse[f'{name}_err'] = np.squeeze((np.sqrt(N0 - 1) * np.std(data_analyse[f'{name}_sample'], axis=-2)).reshape(shape_mean_err), axis=-2) # solve the standard deviation of data
        
        for i in range(N_sum):
            data_cov[i] = np.cov((data_analyse[f'{name}_sample'].reshape(N_sum, N0, N1))[i].T)
        data_analyse[f'{name}_cov'] = data_cov.reshape(shape_cov)
        
    return data_analyse
#Bootstrap: 随机抽取N次原样本中的M个数据，
def Bootstrap(data:np.ndarray, N:int = 0, M:int = 0, name:str='analysed', analyse:bool=False) -> dict:
    N0 = data.shape[-2]
    N1 = data.shape[-1]
    
    if M == 0:
        M = N0 - 5
    if N == 0:
        N = N0 * 4
        
    if M > N1:
        raise print(f'M must less than N0. M{M}, N0{N0}.')
    
    if N > math.factorial(N0) / (math.factorial(N0 - M) * math.factorial(M)):
        raise print(f'N must less than C{N0}{M}. M{M}, N0{N0} and N0 must > 5')
    
    shape = list(data.shape)
    
    shape_mean_err = shape.copy()
    shape_mean_err[-2] = 1
    
    shape_cov = shape.copy()
    shape_cov[-2] = shape_cov[-1]
    
    N_sum = int(np.prod(shape[:-2]))
    data.reshape(N_sum, N0, N1)
    data_cov = np.zeros((N_sum, N1, N1))
    
    accept_ratio = M / N0
    data_sample = np.zeros((N_sum, N, N1))
    data_analyse = {}
    for i in range(N):
        random_array = np.random.random((N0))
        while ((random_array - accept_ratio) < 0).all():
            random_array = np.random.random((N0))
        data_sample[:,i,:] = np.sum(data[:, random_array > accept_ratio, :], axis = -2) / np.sum(random_array > accept_ratio)
    
    data_analyse[f'{name}_sample'] = data_sample
    if analyse == True:
        
        data_analyse[f'{name}_mean'] = np.squeeze(np.mean(data_sample, axis = -2).reshape(shape_mean_err), axis=-2)
        data_analyse[f'{name}_err'] = np.squeeze(np.std(data_sample, axis = -2).reshape(shape_mean_err), axis=-2)
        
        for j in range(N_sum):
            data_cov[j] = np.cov(data_sample[j].T)   
        data_analyse[f'{name}_cov'] = data_cov.reshape(shape_cov)
    return data_analyse
def link_analyse(data_2pt:dict, data_3pt:dict, analyse_type:str='Jackknife') -> dict:
    link_max = data_3pt['link_max']
    tsep = data_3pt['tsep']
    C2pt = data_2pt['init'][:, tsep]
    C3pt = data_3pt['init']
        
    C2pt = (C2pt.T).reshape(1, len(tsep), -1, 1)
    # ratio = C3pt / C2pt
    ratio = np.zeros_like(C3pt)
    for i in range(len(tsep)):
        ratio[...,i,:,:tsep[i]+1] = np.sqrt(C3pt[...,i,:,:tsep[i]+1] * C3pt[...,i,:,:tsep[i]+1][...,::-1] / (C2pt[:,i] * C2pt[:,i]))
    
    if analyse_type == 'Jackknife':
        return Jackknife(ratio, name='link', analyse=True)
    elif analyse_type == 'Bootstrap':
        return  Bootstrap(ratio, name='link', analyse=True)
    
    
    
data_2pt = {}
data_2pt['N0'] = 38
data_2pt['Nt'] = 12 
data_2pt['init'] = np.zeros((data_2pt['N0'], data_2pt['Nt']), dtype=float)
for conf in range(data_2pt['N0']): # _link_max{data_2pt['link_max']}
    data_2pt['init'][conf] = np.sum((np.load(f"/public/home/huaj/School_2024/result/proton_2pt_{conf * 1000 + 10000}.npy").real), axis=0)[:data_2pt['Nt']]
data_3pt = {}
data_3pt['N0'] = 38
data_3pt['Nt'] = 10
data_3pt['link_max'] = 0
data_3pt['tsep'] = [6, 7, 8, 9]
data = data_3pt.copy()
data_3pt['init'] = np.zeros((2 * data_3pt['link_max'] + 1, len(data_3pt['tsep']), data_3pt['N0'], data_3pt['Nt']), dtype=float)
for conf in range(data_3pt['N0']): # _link_max{data_3pt['link_max']}
    for tsep in range(len(data_3pt['tsep'])):
        data_3pt['init'][:, tsep, conf] = (
            np.sum((np.load(f"/public/home/huaj/School_2024/result/protonA_seq{data_3pt['tsep'][tsep]}_{conf * 1000 + 10000}.npy").real),axis=0)[:data_3pt['Nt']]
            # - np. sum((np.load(f"/public/home/huaj/School_2024/result/protonV_seq{data_3pt['tsep'][tsep]}_{conf * 1000 + 10000}.npy").real),axis=0)[:data_3pt['Nt']]
            )
# print(data_3pt['init'][0, :, 0, :])
# print(data_2pt['init'][0, data_3pt['tsep']])
data.update(link_analyse(data_2pt=data_2pt, data_3pt=data_3pt, analyse_type='Jackknife'))
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':25})
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
ax.set_ylim([0, 0.5])
ax.set_xlabel('%s'%('t'))
ax.set_ylabel('%s'%('$ratio$')) # $E_{\mathrm{2pt}}$/Gev
ax.errorbar(np.arange(data['Nt'])[1:data_3pt['tsep'][0]] - 3, data['link_mean'][0, 0, 1:data_3pt['tsep'][0]], yerr=data['link_err'][0, 0, 1:data_3pt['tsep'][0]],
            fmt = '*', alpha=0.5, capsize=3.5, capthick=1.5, label='sep=6', linestyle='none', elinewidth=2) # fmt = 'bs'
ax.errorbar(np.arange(data['Nt'])[1:data_3pt['tsep'][1]] - 3.5, data['link_mean'][0, 1, 1:data_3pt['tsep'][1]], yerr=data['link_err'][0, 1, 1:data_3pt['tsep'][1]],
            fmt = '+', alpha=0.5, capsize=3.5, capthick=1.5, label='sep=7', linestyle='none', elinewidth=2) # fmt = 'bs'
ax.errorbar(np.arange(data['Nt'])[1:data_3pt['tsep'][2]] - 4, data['link_mean'][0, 2, 1:data_3pt['tsep'][2]], yerr=data['link_err'][0, 2, 1:data_3pt['tsep'][2]],
            fmt = 'o', alpha=0.5, capsize=3.5, capthick=1.5, label='sep=8', linestyle='none', elinewidth=2) # fmt = 'bs'
ax.errorbar(np.arange(data['Nt'])[1:data_3pt['tsep'][3]] - 4.5, data['link_mean'][0, 3, 1:data_3pt['tsep'][3]], yerr=data['link_err'][0, 3, 1:data_3pt['tsep'][3]],
            fmt = '.', alpha=0.5, capsize=3.5, capthick=1.5, label='sep=9', linestyle='none', elinewidth=2) # fmt = 'bs'
plt.legend()
plt.show
import lsqfit
import gvar as gv
# data.update(Bootstrap(data['init'], N=1000, M = 30, corr_type=''))
tsep = data['tsep']
fmt = ['*','+','o','s']
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
for i in range(len(data['tsep'])):
    # i = -1
    X = np.arange(data['Nt'])
    ini_prr = {'C0': '0.4(1000)', 'C1':'0.5(1000)', 'C2':'0.18(30)' }
    # fit_all = np.zeros((data_n, fit_n),dtype=classmethod)
    fit_parameter = np.zeros(3,dtype=float)
    def ft_mdls(t_dctnry, p):
        mdls = {}
        ts = t_dctnry['C3pt']
        mdls['C3pt'] = (p['C0'] * (1 + p['C1'] * (np.exp(- p['C2'] * ts) +  np.exp(- p['C2'] * (tsep[i] - ts)))))
        # mdls['C3pt'] = (p['C0'] * (1 + p['C1'] * (np.exp(-0.18 * ts) +  np.exp(-0.18 * (tsep[i] - ts)))))
        return mdls
    t_dctnry = {'C3pt': X[1:tsep[i]]}
    data_dctnry = {'C3pt': gv.gvar(data['link_mean'][0, i, 1:tsep[i]], data['link_err'][0, i, 1:tsep[i]])}
    fit = lsqfit.nonlinear_fit(data=(t_dctnry, data_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) 
    fit_parameter[0] = (float)(fit.chi2/fit.dof)
    fit_parameter[1] = (float)(fit.Q)
    fit_parameter[2] = (float)(fit.logGBF)
    # if i == 3:
    print(fit.format(True))
    # fitted function values
    t_ary = fit.data[0]['C3pt']
    t_lst = np.arange(0.8,tsep[i]-0.6,0.1)
    data_fit_fcn_gvar = fit.fcn({'C3pt':t_lst}, fit.p)['C3pt']
    data_fit_mean = np.array([c2.mean for c2 in data_fit_fcn_gvar])
    data_fit_err = np.array([c2.sdev for c2 in data_fit_fcn_gvar])
    ax.errorbar(X[1:tsep[i]] - (tsep[i]) / 2 , data['link_mean'][0, i, 1:tsep[i]], yerr=data['link_err'][0, i, 1:tsep[i]], fmt = fmt[i], alpha=0.5, capsize=3.5, capthick=1.5, label='tsep=%d'%(tsep[i]), linestyle='none', elinewidth=2) 
    ax.plot(t_lst - (tsep[i]) / 2, data_fit_mean, label="best_fit_tsep=%d"%(tsep[i])) # , color="b"
    ax.fill_between(t_lst - (tsep[i]) / 2, data_fit_mean - data_fit_err, data_fit_mean + data_fit_err,  alpha=0.3) 
    ax.set_ylim([0.2,0.42])
    plt.ylabel('ratio')
    plt.xlabel('tsep-tsep/2')
    ax.legend(loc='lower center', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, borderpad=0.3, \
                ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=1.5)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
fig.show