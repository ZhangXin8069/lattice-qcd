from analyse_fun import *
import matplotlib.pyplot as plt
import numpy as np
import lsqfit
import gvar as gv
import sys as sy
import time
from proplot import rc
st_io = time.time()
infile=fileinput.input()
for line in infile:
    tmp=line.split()
    if(tmp[0]=='num_quark'):
        num_quark=int(tmp[1])
    if(tmp[0]=='Nt'):
        Nt=int(tmp[1])
    if(tmp[0]=='Nx'):
        Nx=int(tmp[1])
    if(tmp[0]=='alttc'):
        alttc=float(tmp[1])
    if(tmp[0]=='Ncnfg'):
        Ncnfg=int(tmp[1])
    if(tmp[0]=='gap'):
        gap=int(tmp[1])
    if(tmp[0]=='N_start'):
        N_start=int(tmp[1])
    if(tmp[0]=='link_max'):
        link_max=int(tmp[1])
        
    # the P part
    if(tmp[0]=='Px_start'):
        Px_start=int(tmp[1])
    if(tmp[0]=='Py_start'):
        Py_start=int(tmp[1])
    if(tmp[0]=='Pz_start'):
        Pz_start=int(tmp[1])
    if(tmp[0]=='Px_end'):
        Px_end=int(tmp[1])
    if(tmp[0]=='Py_end'):
        Py_end=int(tmp[1])
    if(tmp[0]=='Pz_end'):
        Pz_end=int(tmp[1])
        
    # the num of t_sep
    if(tmp[0]=='t_sep_start'):
        t_sep_start=int(tmp[1])
    if(tmp[0]=='t_sep_end'):
        t_sep_end=int(tmp[1])
    if(tmp[0]=='t_sep_gap'):
        t_sep_gap=int(tmp[1])
        
    # the num of ENV
    if(tmp[0]=='ENV_start'):
        ENV_start=int(tmp[1])
    if(tmp[0]=='ENV_end'):
        ENV_end=int(tmp[1])
    if(tmp[0]=='ENV_gap'):
        ENV_gap=int(tmp[1])
    # the file path part
    if(tmp[0]=='first_quark_3pt_corr_path'):
        first_quark_3pt_corr_path=tmp[1]
    if(tmp[0]=='second_quark_3pt_corr_path'):
        second_quark_3pt_corr_path=tmp[1]
    if(tmp[0]=='save_path'):
        save_path=tmp[1]
    # plot parameter
    if(tmp[0]=='C2pt_type'):
        C2pt_type=tmp[1]
    if(tmp[0]=='corr_2pt_path'):
        corr_2pt_path=tmp[1]
ed_io = time.time()
print("analyse input file readed, use time: %.3f s"%(ed_io-st_io))
# initial data
fm2GeV = 0.1973
t_hlf = int(Nt/2)
# fit points
f_start = 8
f_end = 15
n_ti = f_end - f_start
# data path
if (num_quark==3):
    filepath=np.array([first_quark_3pt_corr_path,second_quark_3pt_corr_path,corr_2pt_path])
elif (num_quark==2):
    filepath=np.array([first_quark_3pt_corr_path,corr_2pt_path])
        
t_sep_array = np.asarray(range(t_sep_start,t_sep_end+1,t_sep_gap))
ENV = np.asarray(range(ENV_start,ENV_end+1,ENV_gap))
P = np.asarray([[pz,py,px] for pz in range(Pz_start,Pz_end+1,1) for py in range(Py_start,Py_end+1,1) for px in range(Px_start,Px_end+1,1)])
# data bulk
N_data = filepath.shape[0]
N_ENV = ENV.shape[0]
N_P = P.shape[0]
N_tsep = t_sep_array.shape[0]
N_link = 2*link_max+1
data_readed = np.zeros((N_tsep, N_data, N_ENV, N_P, Ncnfg, Nt, 2),dtype=np.double)
for tsep_indx in range(N_tsep):
    st_read_file = time.time()
    data_readed[tsep_indx] = read_data(N_start, gap, Ncnfg, ENV, P, Nx, Nt, filepath, (tsep_indx+1)*t_sep_gap) # (N_tsep, N_data, N_ENV, N_P, Ncnfg, Nt, (re,im))
    ed_read_file = time.time()
    print("read data in python of t_sep %d, use time %.3f s"%(t_sep_array[tsep_indx],ed_read_file-st_read_file))
data_readed_complex = 1.0*data_readed[:,:,:,:,:,:,0]+1.0j*data_readed[:,:,:,:,:,:,1]
Re_ratio_3pt_2pt_mean = np.zeros((N_tsep,N_data-1,N_ENV,N_P,t_sep_end+1), dtype=np.double)
Re_ratio_3pt_2pt_err = np.zeros((N_tsep,N_data-1,N_ENV,N_P,t_sep_end+1), dtype=np.double)
Re_ratio_3pt_2pt_cov = np.zeros((N_tsep,N_data-1,N_ENV,N_P,t_sep_end+1,t_sep_end+1), dtype=np.double)
Re_2pt_mean = np.zeros((1, N_ENV, N_P, Nt-1), dtype=np.double)
Re_2pt_err = np.zeros((1, N_ENV, N_P, Nt-1),dtype=np.double)
ENV_y = np.zeros((N_tsep,N_data-1,N_ENV, N_P), dtype=np.double)
ENV_err = np.zeros((N_tsep,N_data-1,N_ENV, N_P), dtype=np.double)
ENV_cov = np.zeros((N_tsep,N_P,N_ENV,N_ENV), dtype=np.double)
for i in range(N_tsep):
    for j in range(N_ENV):
        for k in range(N_P):
            st_analyse = time.time()
            ratio_3pt_2pt_data, ratio_3pt_2pt_cov = PDF_3pt_2pt(data_readed_complex[i,:,j,k],(i+1)*t_sep_gap)
            Re_ratio_3pt_2pt_mean[i,:,j,k,:(i+1)*t_sep_gap+1] = np.real(ratio_3pt_2pt_data[0])
            Re_ratio_3pt_2pt_err[i,:,j,k,:(i+1)*t_sep_gap+1]  = np.real(ratio_3pt_2pt_data[1])
            Re_ratio_3pt_2pt_cov[i,:,j,k,:(i+1)*t_sep_gap+1,:(i+1)*t_sep_gap+1]  = np.real(ratio_3pt_2pt_cov)
            ENV_y[i,:,j,k] = np.mean(Re_ratio_3pt_2pt_mean[i,:,j,k,1:(i+1)*t_sep_gap],axis=-1)
            ENV_err[i,:,j,k] = np.std(Re_ratio_3pt_2pt_mean[i,:,j,k,1:(i+1)*t_sep_gap],axis=-1)
            ENV_cov[i,k] = np.cov(ENV_y[i,:,:,k].T)
            Re_2pt_mean[0,j,k] = meff_2pt(data_readed_complex[-1,-1,j,k,:,:], dtype='%s'%(C2pt_type))[0]*fm2GeV/alttc
            Re_2pt_err[0,j,k]  = meff_2pt(data_readed_complex[-1,-1,j,k,:,:], dtype='%s'%(C2pt_type))[1]*fm2GeV/alttc
            ed_analyse = time.time()
            print("t_sep: %d; ENV: %d; P: (%d,%d,%d); ratio(C3pt/C2pt), ENV_ratio and C2pt analyse complete, use time %.3f s"%(t_sep_array[i],ENV[j],P[k,0],P[k,1],P[k,2], ed_analyse-st_analyse))
print(ENV_y,ENV_err)
if num_quark==3:
    y_range=np.array([[0.8,2],[0.2,1.0],[0,0.00002],[0,0.00001],[1,1.15],[0.9,1.2]])
if num_quark==2:
    y_range=np.array([[0,1],[0,0.05],[0,0.05],[0.285,0.3],[0.28,0.31]])
# plot fig
plt.rcParams.update({'font.size':25})
st_plot = time.time()
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
ax.set_title('C2pt_meff',fontdict={'fontsize':30,'fontweight':'light'})
ax.set_ylim(y_range[-1])
ax.set_xlabel('%s'%('t/a'))
ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
# ax.set_xticklabels([np.asarray(range(0,t_hlf,3))],fontdict=xylabel_font)
# ax.set_yticklabels(y_range[-1],fontdict=xylabel_font)
for P_indx in range(N_P):
    ax.errorbar(np.asarray(range(t_hlf)), Re_2pt_mean[0,-1,P_indx,:t_hlf], yerr=Re_2pt_err[0,-1,P_indx,:t_hlf], alpha=0.5, marker = 's', mfc='orange', capsize=3.5, capthick=1.5, label='P=(%d,%d,%d)'%(P[P_indx,0],P[P_indx,1],P[P_indx,2]), linestyle='none',elinewidth=2) # fmt = 'bs'
plt.legend(fontsize=18)
fig.savefig("%s/%s"%(save_path, 'C2pt_meff.png'))
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
ax.set_title('meff_2pt_ENV',fontdict={'fontsize':30,'fontweight':'light'})
ax.set_ylim(y_range[-2])
ax.set_xlabel('%s'%('N_ENV'))
ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
for P_indx in range(N_P):
    Re_2pt_mean_ENV = Re_2pt_mean[0,:,P_indx,12]
    Re_2pt_err_ENV = Re_2pt_err[0,:,P_indx,12]
    ax.errorbar(np.asarray(range(ENV_start, ENV_end+1, ENV_gap)), Re_2pt_mean_ENV, yerr=Re_2pt_err_ENV, alpha=0.5, marker = 's', mfc='orange', capsize=3.5, capthick=1.5, label='P=(%d,%d,%d)'%(P[P_indx,0],P[P_indx,1],P[P_indx,2]), linestyle='none',elinewidth=2) # fmt = 'bs'
plt.legend()
fig.savefig("%s/%s"%(save_path, 'meff_2pt_ENV.png'))
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
ax.set_title('C_2pt_ENV',fontdict={'fontsize':30,'fontweight':'light'})
ax.set_ylim(y_range[-3])
ax.set_xlabel('%s'%('N_ENV'))
ax.set_ylabel('%s'%('$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
C2pt_mean = np.zeros((N_tsep,N_ENV))
C2pt_err = np.zeros((N_tsep,N_ENV))
for t_sep_indx in range(N_tsep):
    C2pt_tsep = data_readed[-1,-1,:,0,:,(t_sep_indx+1)*t_sep_gap,0].T
    C2pt_ENV = C2pt_tsep / (ENV * Nt)
    C2pt_mean[t_sep_indx] = jackknife_ctr_err(C2pt_ENV)[0]
    C2pt_err[t_sep_indx] = jackknife_ctr_err(C2pt_ENV)[1]
    ax.errorbar(np.asarray(range(ENV_start, ENV_end+1, ENV_gap)), C2pt_mean[t_sep_indx], yerr=C2pt_err[t_sep_indx], alpha=0.5, marker = 's', mfc='orange', capsize=3.5, capthick=1.5, label='tsep%d'%(t_sep_array[t_sep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
    plt.legend()
fig.savefig("%s/%s"%(save_path, 'C_2pt_ENV.png'))
for data_indx in range(N_data-1):
    if data_indx==0:
        flavour = 'U'
    if data_indx==1:
        flavour = 'D'
    fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax.set_title('C_3pt_C_2pt_ENV_%s'%(flavour),fontdict={'fontsize':30,'fontweight':'light'})
    ax.set_ylim(y_range[data_indx])
    ax.set_xlabel('%s'%('N_ENV'))
    ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$/$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
    for t_sep_indx in range(N_tsep):
        ax.errorbar(np.asarray(range(ENV_start, ENV_end+1, ENV_gap)), ENV_y[t_sep_indx,data_indx,:,0].T, yerr=ENV_err[t_sep_indx,data_indx,:,0].T, alpha=0.5, marker = 's', mfc='orange', capsize=3.5, capthick=1.5, label='tsep%d'%(t_sep_array[t_sep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
        plt.legend(loc=2)
    fig.savefig("%s/%s"%(save_path, 'C_3pt_C_2pt_ENV_%s.png'%(flavour)))
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
ax.set_title('C_3pt_ENV',fontdict={'fontsize':30,'fontweight':'light'})
ax.set_ylim(y_range[-4])
ax.set_xlabel('%s'%('N_ENV'))
ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$')) # $E_{\mathrm{2pt}}$/Gev
C3pt_mean = np.zeros((N_tsep,N_ENV))
C3pt_err = np.zeros((N_tsep,N_ENV))
for t_sep_indx in range(N_tsep):
    C3pt = data_readed[t_sep_indx,0,:,0,:,:(t_sep_indx+1)*t_sep_gap,0] # ((t_sep_indx, N_ENV, Ncnfg, :(t_sep_indx+1)*t_sep_gap) # (N_tsep, N_data, N_ENV, N_P, Ncnfg, Nt, (re,im))
    C3pt_tmean = (np.mean(C3pt[:,:,1:(t_sep_indx+1)*t_sep_gap],axis=-1)).T
    C3pt_ENV = C3pt_tmean / (ENV * Nt)
    C3pt_mean[t_sep_indx] = jackknife_ctr_err(C3pt_ENV)[0]
    C3pt_err[t_sep_indx] = jackknife_ctr_err(C3pt_ENV)[1]
    
    ax.errorbar(np.asarray(range(ENV_start, ENV_end+1, ENV_gap)), C3pt_mean[t_sep_indx], yerr=C3pt_err[t_sep_indx], alpha=0.5, marker = 's', mfc='orange', capsize=3.5, capthick=1.5, label='C%dpt|tsep%d'%(3,t_sep_array[t_sep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
    plt.legend()
    
fig.savefig("%s/%s"%(save_path, 'C_3pt_ENV.png'))
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
    ax.errorbar(np.asarray(range(ENV_start, ENV_end+1, ENV_gap)), together_C3pt_C2pt_mean[t_sep_indx], yerr=together_C3pt_C2pt_err[t_sep_indx], alpha=0.5, marker = 's', mfc='orange', capsize=3.5, capthick=1.5, label='C%dpt|tsep%d'%(Cpt,t_sep_array[(t_sep_indx)%N_tsep]),linestyle='none',elinewidth=2) # fmt = 'bs'
    plt.legend()
    
fig.savefig("%s/%s"%(save_path, 'C_3pt_C_2pt_ENV.png'))
ed_plot = time.time()
print("plot png figure use time %.3f s"%(ed_plot-st_plot))
print("****************************** all complete use time %.3f s*********************************"%(ed_plot-st_io))
# lsqfit
# t_ary = np.asarray(range(t_sep+1))-int(t_sep/2)
# ini_prr = {'C0': '2(0.5)', 'C1': '0(0.5)', 'C3': '0(0.5)', 'C5': '0(0.5)', 'E0': '0.1(0.5)'}
# # fit_all = np.zeros((data_n,fit_n),dtype=classmethod)
# fit_parameter = np.zeros(3,dtype=float)
# def ft_mdls(t_dctnry, p):
#     mdls = {}
#     ts = t_dctnry['C3pt']
#     mdls['C3pt'] = (p['C0'] + p['C1']*(np.exp(-p['E0']*(ts - int(t_sep/2))*alttc/fm2GeV) + np.exp(-p['E0']*(ts + int(t_sep/2))*alttc/fm2GeV)) + p['C3']*np.exp(-p['E0']*t_sep*alttc/fm2GeV)) /(1 + p['C5']*np.exp(-p['E0']*t_sep*alttc/fm2GeV))
#     return mdls
# t_dctnry = {'C3pt': t_ary[:]}
# data_dctnry = {'C3pt': gv.gvar(Re_ratio_3pt_2pt_mean[0,-1,0], Re_ratio_3pt_2pt_cov[0,-1,0])}
# fit = lsqfit.nonlinear_fit(data=(t_dctnry, data_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) 
# fit_parameter[0] = (float)(fit.chi2/fit.dof)
# fit_parameter[1] = (float)(fit.Q)
# fit_parameter[2] = (float)(fit.logGBF)
# print(fit.format(True))
