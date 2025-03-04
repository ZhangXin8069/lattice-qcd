import sys as sy
from analyse_fun import *
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import lsqfit
import gvar as gv
import sympy as sp
import time
from proplot import rc
st_io = time.time()
infile=fileinput.input()
inp = read_input(infile)
fm2GeV = 0.1973
# fit points
f_start = 8
f_end = 15
n_ti = f_end - f_start
ed_io = time.time()
print("analyse input file readed, use time: %.3f s"%(ed_io-st_io))
st_read_file = time.time()
# data path
if inp.type == 'ratio' or inp.type == '3pt':
    filepath = np.array([inp.data_quark_3pt_corr_path, inp.data_corr_2pt_path, inp.iog_quark_3pt_corr_path, inp.iog_corr_2pt_path])
else:
    filepath = np.array([inp.iog_quark_3pt_corr_path, inp.iog_corr_2pt_path])
if inp.time_fold == 1:
    meff_Nt_2pt=inp.Nt//2
    Nt_3pt = inp.Nt//2+1
else:
    meff_Nt_2pt=inp.Nt-1
    Nt_3pt = inp.Nt
tsep_array = np.asarray(range(inp.t_sep_start, inp.t_sep_end+1, inp.t_sep_gap))
ENV_array = np.asarray(range(inp.ENV_start,inp.ENV_end+1,inp.ENV_gap))
P_array = np.asarray([[pz,py,px] for pz in range(inp.Pz_start, inp.Pz_end+1,1) for py in range(inp.Py_start, inp.Py_end+1,1) for px in range(inp.Px_start, inp.Px_end+1,1)])
link_array = np.asarray(range(-inp.link_max,inp.link_max+1,1))
if inp.read_type == 'data':
    data_readed = read_data(filepath[:], inp.Nx, inp.Nt, P_array, ENV_array, inp.N_start, inp.gap, inp.Ncnfg, tsep_array, inp.link_max, inp.type) # data, P, ENV, tsep, link, inp.Ncnfg, inp.Nt, (re,im)
elif inp.read_type == 'iog':
    data_readed = read_iog(filepath[:], inp.Nx, inp.Nt, P_array, ENV_array, inp.N_start, inp.gap, inp.Ncnfg, tsep_array, inp.link_max, inp.type)
elif inp.read_type == 'both':
    data_readed = []
    data_readed[:inp.num_quark] = read_data(filepath[:inp.num_quark], inp.Nx, inp.Nt, P_array, ENV_array, inp.N_start, inp.gap, inp.Ncnfg, tsep_array, inp.link_max, inp.type)
    data_readed[inp.num_quark:] = read_iog(filepath[inp.num_quark:], inp.Nx, inp.Nt, P_array, ENV_array, inp.N_start, inp.gap, inp.Ncnfg, tsep_array, inp.link_max, inp.type)
    data_readed = np.asarray(data_readed)
ed_read_file = time.time()
print("read data in data of python, use time %.3f s"%(ed_read_file-st_read_file))
N_data = data_readed.shape[0]
N_ENV = ENV_array.shape[0]
N_P = P_array.shape[0]
N_tsep = tsep_array.shape[0]
N_link = link_array.shape[0]
N_ratio = int(N_data/2)
N_C2pt = ceil(N_data/2)
data_readed_Re = np.zeros((N_data, N_P, N_ENV, N_tsep, N_link, inp.Ncnfg, Nt_3pt), dtype = np.double)
if inp.time_fold == 1:
    for i in range(N_ratio):
        data_readed_Re[2*i] = (1.0*(data_readed[2*i,...,1:32+2,0] - data_readed[2*i,...,32-1:,0][...,::-1]))# data, P, ENV, tsep, link, inp.Ncnfg, inp.Nt, dtype=double
        data_readed_Re[2*i+1] = (1.0*(data_readed[2*i+1,...,:32+1,0] + data_readed[2*i+1,...,32-1:,0][...,::-1]))
else:
    data_readed_Re = 1.0*data_readed[:,:,:,:,:,:,:,0]
st_analyse = time.time()
Re_ratio_3pt_2pt_mean = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link, Nt_3pt), dtype=np.double)
Re_ratio_3pt_2pt_err = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link, Nt_3pt), dtype=np.double)
Re_ratio_3pt_2pt_cov = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link, Nt_3pt, Nt_3pt), dtype=np.double)
for i in range(N_ratio):
    Re_ratio_3pt_2pt_mean[i], Re_ratio_3pt_2pt_err[i], Re_ratio_3pt_2pt_cov[i] = PDF_3pt_2pt(data_readed_Re[2*i:2*i+2], tsep_array)
ENV_y = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link), dtype=np.double) # N_ratio, N_P, N_ENV, N_tsep, N_link
ENV_err = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link), dtype=np.double)
ENV_cov = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link), dtype=np.double)
for tsep_indx in range(N_tsep):
    ENV_y[:,:,:,tsep_indx,:] = np.mean(Re_ratio_3pt_2pt_mean[:,:,:,tsep_indx,:,tsep_array[tsep_indx]//2-5:tsep_array[tsep_indx]//2+5],axis=-1)
    ENV_err[:,:,:,tsep_indx,:] = np.std(Re_ratio_3pt_2pt_mean[:,:,:,tsep_indx,:,tsep_array[tsep_indx]//2-5:tsep_array[tsep_indx]//2+5],axis=-1)
Re_2pt_mean = np.zeros((N_C2pt, N_ENV, N_P, meff_Nt_2pt), dtype=np.double)
Re_2pt_err = np.zeros((N_C2pt, N_ENV, N_P, meff_Nt_2pt),dtype=np.double)
for C2pt_indx in range(N_C2pt):
    for P_indx in range(N_P):
        if C2pt_indx==0:
            N_ENV_change = N_ENV
        else:
            N_ENV_change = 1
        for ENV_indx in range(N_ENV_change):
            Re_2pt_mean[C2pt_indx,ENV_indx,P_indx,:meff_Nt_2pt-1] = (meff_2pt(data_readed_Re[C2pt_indx*2+1,P_indx,ENV_indx,0,0], dtype='%s'%(inp.C2pt_type), time_fold=inp.time_fold)[0]*fm2GeV/inp.alttc)
            Re_2pt_err[C2pt_indx,ENV_indx,P_indx,:meff_Nt_2pt-1]  = (meff_2pt(data_readed_Re[C2pt_indx*2+1,P_indx,ENV_indx,0,0], dtype='%s'%(inp.C2pt_type), time_fold=inp.time_fold)[1]*fm2GeV/inp.alttc)
print(Re_2pt_mean)
ed_analyse = time.time()
print("ratio(C3pt/C2pt), ENV_ratio and C2pt analyse complete, use time %.3f s"%(ed_analyse-st_analyse))
y_range=np.array([[0,0.5],[0,0.05],[0.85,1.25],[0,1.25],[0,0.05],[0.29,0.31],[0.29,0.31]])
# plot fig
marker_array = np.array(['s','*','+','x','p','h','v','X','D','P','H','o'])
if inp.type == '2pt' or inp.type == 'ratio':
    plt.rcParams.update({'font.size':25})
    st_plot = time.time()
    fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax.set_title('C2pt_meff_%dx%d_Nt_cosh_mass-0.1450.png'%(inp.Nx,inp.Nt),fontdict={'fontsize':30,'fontweight':'light'})
    ax.set_ylim(y_range[-1])
    ax.set_xlabel('%s'%('t/a'))
    ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
    # ax.set_xticklabels([np.asarray(range(0,t_hlf,3))],fontdict=xylabel_font)
    # ax.set_yticklabels(y_range[-1],fontdict=xylabel_font)
    for P_indx in range(N_P):
        ax.errorbar(np.asarray(range(meff_Nt_2pt)), Re_2pt_mean[0,-1,P_indx,:meff_Nt_2pt], yerr=Re_2pt_err[0,-1,P_indx,:meff_Nt_2pt], alpha=0.5, marker = marker_array[P_indx], capsize=3.5, capthick=1.5, label='P=(%d,%d,%d)'%(P_array[P_indx,0],P_array[P_indx,1],P_array[P_indx,2]), linestyle='none',elinewidth=2) # fmt = 'bs'
    plt.legend(fontsize=18)
    fig.savefig("%s/%s"%(inp.save_path, 'C2pt_meff_%dx%d_Nt_cosh_mass-0.1450.png'%(inp.Nx,inp.Nt)))
    
    fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax.set_title('meff_2pt_ENV',fontdict={'fontsize':30,'fontweight':'light'})
    ax.set_ylim(y_range[-2])
    ax.set_xlabel('%s'%('N_ENV'))
    ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
    for P_indx in range(N_P):
        Re_2pt_mean_ENV = Re_2pt_mean[0,:,P_indx,12]
        Re_2pt_err_ENV = Re_2pt_err[0,:,P_indx,12]
        ax.errorbar(np.asarray(range(inp.ENV_start, inp.ENV_end+1, inp.ENV_gap)), Re_2pt_mean_ENV, yerr=Re_2pt_err_ENV, alpha=0.5, marker = marker_array[P_indx], capsize=3.5, capthick=1.5, label='P=(%d,%d,%d)'%(P_array[P_indx,0],P_array[P_indx,1],P_array[P_indx,2]), linestyle='none',elinewidth=2) # fmt = 'bs'
    plt.legend()
    fig.savefig("%s/%s"%(inp.save_path, 'meff_2pt_ENV_distillation.png'))
    fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax.set_title('C_2pt_ENV',fontdict={'fontsize':30,'fontweight':'light'})
    ax.set_ylim(y_range[-3])
    ax.set_xlabel('%s'%('N_ENV'))
    ax.set_ylabel('%s'%('$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
    C2pt_mean = np.zeros((N_tsep,N_ENV))
    C2pt_err = np.zeros((N_tsep,N_ENV))
    for t_sep_indx in range(N_tsep):
        C2pt_tsep = data_readed[-1,0,:,0,0,:,tsep_array[t_sep_indx],0].T # data, P, ENV, tsep, link, inp.Ncnfg, inp.Nt, (re,im)
        C2pt_ENV = C2pt_tsep / (ENV_array * inp.Nt)
        C2pt_mean[t_sep_indx] = jackknife_ctr_err(C2pt_ENV)[0]
        C2pt_err[t_sep_indx] = jackknife_ctr_err(C2pt_ENV)[1]
        ax.errorbar(np.asarray(range(inp.ENV_start, inp.ENV_end+1, inp.ENV_gap)), C2pt_mean[t_sep_indx], yerr=C2pt_err[t_sep_indx], alpha=0.5, marker = marker_array[t_sep_indx], capsize=3.5, capthick=1.5, label='tsep%d'%(tsep_array[t_sep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
        plt.legend()
    fig.savefig("%s/%s"%(inp.save_path, 'C_2pt_ENV_distillation.png'))
    
if inp.type == '3pt' or inp.type == 'ratio':
    # ENV_y: N_data-1, N_P, N_ENV, N_tsep, N_link
    for data_indx in range(1):
        if data_indx==0:
            data_type = 'data'
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title('C_3pt_C_2pt_ENV_%s'%(data_type),fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range[data_indx])
        ax.set_xlabel('%s'%('N_ENV'))
        ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$/$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
        for t_sep_indx in range(N_tsep):
            ax.errorbar(np.asarray(range(inp.ENV_start, inp.ENV_end+1, inp.ENV_gap)), ENV_y[data_indx,0,:,t_sep_indx,inp.link_max].T, yerr=ENV_err[data_indx,0,:,t_sep_indx,inp.link_max].T, alpha=0.5, marker = marker_array[t_sep_indx], capsize=3.5, capthick=1.5, label='tsep%d'%(tsep_array[t_sep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
            plt.legend(loc=2)
        fig.savefig("%s/%s"%(inp.save_path, 'C_3pt_C_2pt_ENV_%s_distillation.png'%(data_type)))
    for data_indx in range(1):
        if data_indx==0:
            data_type = 'data'
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title('C_3pt_ENV_%s'%(data_type),fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range[-4])
        ax.set_xlabel('%s'%('N_ENV'))
        ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$')) # $E_{\mathrm{2pt}}$/Gev
        C3pt_mean = np.zeros((N_tsep,N_ENV))
        C3pt_err = np.zeros((N_tsep,N_ENV))
        for t_sep_indx in range(N_tsep):
            C3pt = data_readed[data_indx,0,:,t_sep_indx,inp.link_max,:,:,0] # data, P, ENV, tsep, link, inp.Ncnfg, inp.Nt, (re,im)
            C3pt_tmean = (np.mean(C3pt[:,:,1:(t_sep_indx+1)*inp.t_sep_gap],axis=-1)).T
            C3pt_ENV = C3pt_tmean / (ENV_array * inp.Nt)
            C3pt_mean[t_sep_indx] = jackknife_ctr_err(C3pt_ENV)[0]
            C3pt_err[t_sep_indx] = jackknife_ctr_err(C3pt_ENV)[1]
            
            ax.errorbar(np.asarray(range(inp.ENV_start, inp.ENV_end+1, inp.ENV_gap)), C3pt_mean[t_sep_indx], yerr=C3pt_err[t_sep_indx], alpha=0.5, marker = marker_array[t_sep_indx], capsize=3.5, capthick=1.5, label='C%dpt|tsep%d'%(3,tsep_array[t_sep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
            plt.legend()
            
        fig.savefig("%s/%s"%(inp.save_path, 'C_3pt_ENV_%s_distillation.png'%(data_type)))
    together_C3pt_C2pt_mean = np.append(C3pt_mean,C2pt_mean).reshape(N_tsep*2,N_ENV)
    together_C3pt_C2pt_err = np.append(C3pt_err,C2pt_err).reshape(N_tsep*2,N_ENV)
    fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax.set_title('C_3pt_C_2pt_ENV',fontdict={'fontsize':30,'fontweight':'light'})
    ax.set_ylim(y_range[-3])
    ax.set_xlabel('%s'%('N_ENV'))
    ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$ and $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
    for t_sep_indx in range(N_tsep*2):
        Cpt=3
        if t_sep_indx >=  N_tsep:Cpt=2
        ax.errorbar(np.asarray(range(inp.ENV_start, inp.ENV_end+1, inp.ENV_gap)), together_C3pt_C2pt_mean[t_sep_indx], yerr=together_C3pt_C2pt_err[t_sep_indx], alpha=0.5, marker = marker_array[t_sep_indx], capsize=3.5, capthick=1.5, label='C%dpt|tsep%d'%(Cpt,tsep_array[(t_sep_indx)%N_tsep]),linestyle='none',elinewidth=2) # fmt = 'bs'
        plt.legend()
        
    fig.savefig("%s/%s"%(inp.save_path, 'C_3pt_C_2pt_ENV_distillation.png'))
    ed_plot = time.time()
    fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax.set_title('ratio_link',fontdict={'fontsize':30,'fontweight':'light'})
    ax.set_ylim(y_range[-4])
    ax.set_xlabel('%s'%('link_indx'))
    ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
    ENV_y_change = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link), dtype=np.double)
    ENV_err_change = np.zeros((N_ratio, N_P, N_ENV, N_tsep, N_link), dtype=np.double)
    for data_indx in range(N_ratio):
        if data_indx==0:
            data_type = 'distillation'
            ENV_indx = range(-5,0,1)
        else:
            data_type = 'chroma'
            ENV_indx = range(1)
        for ENV_indx in ENV_indx:
            ENV_y_change[data_indx] = ENV_y[data_indx] # / ENV_y[data_indx, 0, ENV_indx, 0, inp.link_max]
            ENV_err_change[data_indx] = ENV_err[data_indx] # / ENV_y[data_indx, 0, ENV_indx, 0, inp.link_max]
            ENV_array_change = ENV_array
            ENV_array_change[0]=-1
            ax.errorbar(link_array, ENV_y_change[data_indx,0,ENV_indx,0], yerr=ENV_err_change[data_indx,0,ENV_indx,0], alpha=0.5, marker = marker_array[ENV_indx], capsize=3.5, capthick=1.5, label='%s-ENV%d'%(data_type,ENV_array_change[ENV_indx]),linestyle='none',elinewidth=2) # fmt = 'bs'
            plt.legend()
                
    fig.savefig("%s/%s"%(inp.save_path, 'ratio_link_%s.png'%(inp.read_type)))
    
    fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax.set_title('ratio_t',fontdict={'fontsize':30,'fontweight':'light'})
    ax.set_ylim(y_range[-5])
    ax.set_xlabel('%s'%('tsep-tsep/2'))
    ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
    for data_indx in range(N_ratio):
        if data_indx==0:
            data_type = 'distillation'
            ENV_indx = range(-5,0,1)
        else:
            data_type = 'chroma'
            ENV_indx = range(1)
            ENV_array_change = ENV_array
            ENV_array_change[0]=-1
        for ENV_indx in ENV_indx:
            ax.errorbar(np.asarray(range(21))-10, Re_ratio_3pt_2pt_mean[data_indx,0,ENV_indx,0,inp.link_max,tsep_array[tsep_indx]//2-10:tsep_array[tsep_indx]//2+11], yerr=Re_ratio_3pt_2pt_err[data_indx,0,ENV_indx,0,inp.link_max,tsep_array[tsep_indx]//2-10:tsep_array[tsep_indx]//2+11], alpha=0.5, marker = marker_array[ENV_indx], capsize=3.5, capthick=1.5, label='%s-ENV%d'%(data_type,ENV_array_change[ENV_indx]),elinewidth=2) # fmt = 'bs'
            plt.legend(loc=5)
                
    fig.savefig("%s/%s"%(inp.save_path, 'ratio_t_%s.png'%(inp.read_type)))
    
    ed_plot = time.time()
        
    print("plot png figure use time %.3f s"%(ed_plot-st_plot))
    print("****************************** all complete use time %.3f s*********************************"%(ed_plot-st_io))
# for data_indx in range(N_ratio):
#     if data_indx==0:
#         data_type = 'distillation'
#         ENV_indx = range(-5,0,1)
#     else:
#         data_type = 'chroma'
#         ENV_indx = range(1)
#     for ENV_indx in ENV_indx:
#         ENV_y[data_indx] = ENV_y[data_indx] / ENV_y[data_indx, 0, ENV_indx, 0, inp.link_max]
#         ENV_err[data_indx] = ENV_err[data_indx] / ENV_y[data_indx, 0, ENV_indx, 0, inp.link_max]
# A,B,C,D = sp.symbols('A B C D')
# X=[A,B,C,D]
# X0=[1,0.2,0.1,-0.05]
# print(ENV_y)
# fit_function = sum((ENV_y[0, 0, -1, 0, x+inp.link_max] - A*sp.exp(-B*x+C) - D)**2 for x in range(11))
# print(fit_function)
# X_0 , funvale = min_fun_descent(fit_function,X,X0)
# print(ENV_y[0, 0, -1, 0, inp.link_max:])
# for i in range(11):
#     print(float(float(X_0[0]) * np.exp(float(-X_0[1]) * i + X_0[2]) + X_0[3]) )
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
