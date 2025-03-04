import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import fileinput
import sympy as sp
import lsqfit
from iog_reader.iog_reader import iog_read
class read_input:
    def __init__(
        self,
        infile
        ) -> None:
        for line in infile:
            tmp=line.split()
            if(tmp[0]=='num_quark'):
                self.num_quark=int(tmp[1])
            if(tmp[0]=='Nt'):
                self.Nt=int(tmp[1])
            if(tmp[0]=='Nx'):
                self.Nx=int(tmp[1])
            if(tmp[0]=='alttc'):
                self.alttc=float(tmp[1])
            if(tmp[0]=='Ncnfg'):
                self.Ncnfg=int(tmp[1])
            if(tmp[0]=='gap'):
                self.gap=int(tmp[1])
            if(tmp[0]=='N_start'):
                self.N_start=int(tmp[1])
            if(tmp[0]=='link_max'):
                self.link_max=int(tmp[1])
            if(tmp[0]=='type'):
                self.type=tmp[1]
            if(tmp[0]=='read_type'):
                self.read_type=tmp[1]
            if(tmp[0]=='time_fold'):
                self.time_fold=int(tmp[1])
            # the P part
            if(tmp[0]=='Px_start'):
                self.Px_start=int(tmp[1])
            if(tmp[0]=='Py_start'):
                self.Py_start=int(tmp[1])
            if(tmp[0]=='Pz_start'):
                self.Pz_start=int(tmp[1])
            if(tmp[0]=='Px_end'):
                self.Px_end=int(tmp[1])
            if(tmp[0]=='Py_end'):
                self.Py_end=int(tmp[1])
            if(tmp[0]=='Pz_end'):
                self.Pz_end=int(tmp[1])
        
            # the num of t_sep
            if(tmp[0]=='t_sep_start'):
                self.t_sep_start=int(tmp[1])
            if(tmp[0]=='t_sep_end'):
                self.t_sep_end=int(tmp[1])
            if(tmp[0]=='t_sep_gap'):
                self.t_sep_gap=int(tmp[1])
        
            # the num of ENV
            if(tmp[0]=='ENV_start'):
                self.ENV_start=int(tmp[1])
            if(tmp[0]=='ENV_end'):
                self.ENV_end=int(tmp[1])
            if(tmp[0]=='ENV_gap'):
                self.ENV_gap=int(tmp[1])
            # the file path part
            if(tmp[0]=='data_quark_3pt_corr_path'):
                self.data_quark_3pt_corr_path=tmp[1]
            if(tmp[0]=='data_corr_2pt_path'):
                self.data_corr_2pt_path=tmp[1]
            if(tmp[0]=='iog_quark_3pt_corr_path'):
                self.iog_quark_3pt_corr_path=tmp[1]
            if(tmp[0]=='iog_corr_2pt_path'):
                self.iog_corr_2pt_path=tmp[1]
            if(tmp[0]=='save_path'):
                self.save_path=tmp[1]
            # plot parameter
            if(tmp[0]=='C2pt_type'):
                self.C2pt_type=tmp[1]
class data_analyse:
    def __init__(
        self,
        num_data:int,
        hadron:str,
        filepath:np.ndarray,
        alttc:float,
        Nx:int,
        Nt:int,
        P:np.ndarray,
        ENV:np.ndarray,
        N_start:int,
        gap:int,
        tsep:np.ndarray,
        save_path:str,
        Ncnfg_data:int=0,
        Ncnfg_iog:int=0,
        link_max:int=0,
        time_fold:bool=False,
        analyse_type:str='ratio',
        read_type:str='data',
        meff_type:str='cosh',
        ) -> None:
        self.fm2GeV = 0.1973
        self.num_data = num_data
        self.hadron = hadron
        self.alttc = alttc
        self.Nx = Nx
        self.Nt = Nt
        self.Nt_con = Nt
        self.N_ENV = int(ENV.shape[0])
        self.ENV = np.asarray(ENV)
        self.N_P = int(P.shape[0])
        self.P = np.asarray(P)
        self.N_tsep = int(tsep.shape[0])
        self.tsep = np.asarray(tsep)
        self.filepath = filepath
        self.link_max = link_max
        self.N_link = 2*link_max+1
        self.Ncnfg_data = Ncnfg_data
        self.Ncnfg_iog = Ncnfg_iog
        self.link = np.asarray(range(-link_max, link_max+1, 1))
        self.conf_iog = np.asarray(range(N_start, N_start+gap*Ncnfg_iog+1, gap))
        self.conf_data = np.asarray(range(N_start, N_start+gap*Ncnfg_data+1, gap))
        self.time_fold = time_fold
        self.analyse_type = analyse_type
        self.read_type = read_type
        self.meff_type = meff_type
        self.save_path = save_path
        self.marker_array = np.array(['s','*','+','x','p','h','v','X','D','P','H','o'])
        self.readed = {}
        self.meff_data_2pt = {}
        self.readed_mean = {}
        self.readed_jcknf = {}
        self.PDF_mean = {}
        self.PDF_err = {}
        self.PDF_cov = {}
        self.link_y = {}
        self.link_err = {}
        if self.read_type == 'data' or self.read_type == 'both':
            if self.analyse_type == 'ratio':
                for i in range(self.num_data - 1):
                    if i == 0 : flavour = 'U'
                    elif i == 1 : flavour = 'D'
                    self.read_data_3pt(self.filepath[i], flavour)
                self.read_data_2pt(self.filepath[self.num_data - 1])
            elif self.analyse_type == '2pt':
                if self.filepath.shape[0] == 1:
                    C2pt = 0
                elif self.num_data == 3 :
                    C2pt = 2
                else:
                    C2pt = 1
                self.read_data_2pt(self.filepath[C2pt])
        if self.read_type == 'iog' or self.read_type == 'both':
            if self.analyse_type == 'ratio':
                for i in range(1):
                    if i == 0 : flavour = 'U'
                    elif i == 1 : flavour = 'D'
                    self.read_iog_3pt(self.filepath[-2], flavour)
                self.read_iog_2pt(self.filepath[-1])
            elif self.analyse_type == '2pt':
                self.read_iog_2pt(self.filepath[-1])
        if self.read_type != 'iog' and self.read_type != 'both' and self.read_type != 'data':
            print('Unsupported file format. This class can sport iog and data.')
            exit()
        if time_fold == 1:
            self.Nt = Nt//2+1
        for i in range(len(self.readed.keys())):
            self.readed_mean[str(list(self.readed.keys())[i])] = np.mean(list(self.readed.values())[i][...,0], axis=-2)
            self.readed_jcknf[str(list(self.readed.keys())[i])] = self.jcknf_sample(list(self.readed.values())[i][...,0])
    def read_iog_3pt(self, filepath, flavour):
        readed = np.zeros((self.N_P, 1, self.N_tsep, self.N_link, self.Ncnfg_iog, self.Nt, 2), dtype = np.double)
        intrptr = ['ID']
        for i in range(1 * self.N_tsep * self.N_P * self.Ncnfg_iog):
            tsep_indx = (i) % self.N_tsep
            conf_indx = (i//(self.N_tsep)) % self.Ncnfg_iog
            P_indx = (i//(self.N_tsep * self.Ncnfg_iog)) % self.N_P
            data = iog_read(filepath%(self.Nx, self.Nt, self.P[P_indx,0], self.P[P_indx,1], self.P[P_indx,2], -1, self.conf_iog[conf_indx], self.tsep[tsep_indx], self.link_max), intrptr)
            readed[P_indx,0,tsep_indx,:,conf_indx,:,0] = np.append(data['Re'].to_numpy().reshape(self.N_link,16,self.Nt)[self.link_max+1:,8][::-1], data['Re'].to_numpy().reshape(self.N_link,16,self.Nt)[:self.link_max+1,8], axis=0)
            readed[P_indx,0,tsep_indx,:,conf_indx,:,1] = np.append(data['Im'].to_numpy().reshape(self.N_link,16,self.Nt)[self.link_max+1:,8][::-1], data['Im'].to_numpy().reshape(self.N_link,16,self.Nt)[:self.link_max+1,8], axis=0)
        if self.time_fold == True:
            readed = readed[...,1:self.Nt//2+2,:] - readed[...,self.Nt//2-1:,:][...,::-1,:]
        self.readed['readed_3pt_iog_%s'%(flavour)] = readed  
    def read_iog_2pt(self, filepath):
        readed = np.zeros((self.N_P, 1, self.Ncnfg_iog, self.Nt, 2), dtype = np.double)
        intrptr = ['ID']
        for i in range(1*self.N_P*self.Ncnfg_iog):
            conf_indx = (i)%self.Ncnfg_iog
            P_indx = (i//(self.Ncnfg_iog))%self.N_P
            data = iog_read(filepath%(self.Nx, self.Nt, self.P[P_indx,0], self.P[P_indx,1], self.P[P_indx,2], -1, self.conf_iog[conf_indx]), intrptr)
            readed[P_indx,0,conf_indx,:,0] = data['Re'].to_numpy()
            readed[P_indx,0,conf_indx,:,1] = data['Im'].to_numpy()
        if self.time_fold == True:
            readed = readed[...,:self.Nt//2+1,:]
        self.readed['readed_2pt_iog'] = readed
    def read_data_3pt(self, filepath, flavour):
        readed = np.zeros((self.N_P, self.N_ENV, self.N_tsep, self.N_link, self.Ncnfg_data, self.Nt, 3), dtype = np.double)
        for i in range(1 * self.N_link * self.N_tsep * self.N_ENV * self.N_P * self.Ncnfg_data):
            link_indx = (i)%self.N_link
            tsep_indx = (i//self.N_link)%self.N_tsep
            conf_indx = (i//(self.N_link*self.N_tsep))%self.Ncnfg_data
            ENV_indx = (i//(self.N_link*self.N_tsep*self.Ncnfg_data))%self.N_ENV
            P_indx = (i//(self.N_link*self.N_tsep*self.Ncnfg_data*self.N_ENV))%self.N_P
            data = open(filepath%(self.Nx, self.Nt, self.P[P_indx,0], self.P[P_indx,1], self.P[P_indx,2], self.ENV[ENV_indx], self.conf_data[conf_indx], self.tsep[tsep_indx], self.link[link_indx]), '+r')
            data_A = data.readlines()
            mid_data_B = np.array([item.replace('\n', '') for item in data_A])
            n_data = np.size(mid_data_B)-1
            mid_data_C = np.zeros(3)
            for t_indx in range(n_data):
                mid_data_C = mid_data_B[t_indx+1].split(' ')
                readed[P_indx,ENV_indx,tsep_indx,link_indx,conf_indx,t_indx,:] = [float(list) for list in mid_data_C] # self.P, self.ENV, tsep, link, self.Ncnfg_data, self.Nt, (list,re,im)
            data.close()
        if self.time_fold == True:
            readed = readed[...,1:self.Nt//2+2,:] - readed[...,self.Nt//2-1:,:][...,::-1,:]
        self.readed['readed_3pt_data_%s'%(flavour)] = readed[...,1:]
    def read_data_2pt(self,filepath):
        readed = np.zeros((self.N_P, self.N_ENV, self.Ncnfg_data, self.Nt, 3), dtype = np.double)
        for i in range(1 * self.N_ENV * self.N_P * self.Ncnfg_data):
            conf_indx = (i)%self.Ncnfg_data
            ENV_indx = (i//(self.Ncnfg_data))%self.N_ENV
            P_indx = (i//(self.Ncnfg_data*self.N_ENV))%self.N_P
            data = open(filepath%(self.Nx, self.Nt, self.P[P_indx,0], self.P[P_indx,1], self.P[P_indx,2], self.ENV[ENV_indx], self.conf_data[conf_indx]), '+r')
            data_A = data.readlines()
            mid_data_B = np.array([item.replace('\n', '') for item in data_A])
            n_data = np.size(mid_data_B)-1
            mid_data_C = np.zeros(3)
            for t_indx in range(n_data):
                mid_data_C = mid_data_B[t_indx+1].split(' ')
                readed[P_indx,ENV_indx,conf_indx,t_indx,:] = [float(list) for list in mid_data_C] # data, P, ENV, tsep, link, Ncnfg, Nt, (list,re,im)
            data.close()
        if self.time_fold == True:
            readed = readed[...,:self.Nt//2+1,:]
        self.readed['readed_2pt_data'] = readed[...,1:]
    def jcknf_sample(self, data):
        sum_dimension = np.asarray(np.shape(data))
        sum_dimension[-2] = 1
        n = data.shape[-2]
        data_sum = np.sum(data, axis = -2).reshape(sum_dimension)
        jcknf_sample = (data_sum - data)/(n-1)
        return jcknf_sample
    def jackknife_ctr_err(self, data): # data (Ncnfg, Nt)
        n = data.shape[0]
        Nt = data.shape[1]
        jcknf_data = np.zeros((2,Nt), dtype = np.double)
        jcknf_sample_data = self.jcknf_sample(data)
        jcknf_data_cntrl = np.mean(jcknf_sample_data, axis = 0)
        # jcknf_data_err_2 = np.sqrt(np.sum((data_mean-data_minus_state)**2,axis=0)*(n-1)/n)  # same as jcknf_data_err_2
        jcknf_data_err = np.std(jcknf_sample_data, axis = 0)*np.sqrt(n-1)
        return jcknf_data_cntrl,jcknf_data_err
    def meff_2pt(self, data_type:str): # data:1, P, ENV, 1, 1, Ncnfg, Nt, dtype=complex
        C2pt_mean = self.readed_mean['readed_2pt_%s'%(data_type)]
        C2pt_jcknf = self.readed_jcknf['readed_2pt_%s'%(data_type)]
        Ncnfg = C2pt_jcknf.shape[-2]
        if self.meff_type == 'log':
            data_mean = np.array(np.log(np.real(C2pt_mean[...,:-1])/np.real(C2pt_mean[...,1:]))) * (self.fm2GeV / self.alttc)
            data_log_sample = np.array(np.log(np.real(C2pt_jcknf[...,:-1])/np.real(C2pt_jcknf[...,1:]))) * (self.fm2GeV / self.alttc)
            data_err = np.std(data_log_sample, axis=-2)*np.sqrt(Ncnfg-1)
            # data_err = np.sqrt(np.sum((data_log_sample - data_mean)**2, axis=-2) * (Ncnfg-1) / Ncnfg)
        elif self.meff_type == 'cosh':
            data_mean = np.arccosh((C2pt_mean[...,2:] + C2pt_mean[...,:-2])/(2 * C2pt_mean[...,1:-1])) * (self.fm2GeV / self.alttc)
            data_cosh_sample = np.arccosh((C2pt_jcknf[...,2:] + C2pt_jcknf[...,:-2])/(2 * C2pt_jcknf[...,1:-1])) * (self.fm2GeV / self.alttc)
            # data_sample_ini = data_sample_ini.astype(np.float64)
            # t_ary = np.array(range(Nt-1))
            # ini = 0.2*np.ones_like(t_ary)
            # def eff_mass_eqn(c2pt):
            #     return lambda E0: (c2pt[:-1]/c2pt[1:]) - np.cosh(E0*(T_hlf-t_ary)) / np.cosh(E0*(T_hlf-(t_ary+1)))
            # def fndroot(eqnf,ini):
            #     sol = fsolve(eqnf,ini, xtol=1e-5)
            #     return sol
            # data_cosh_sample = np.array([fndroot(eff_mass_eqn(c2pt),ini) for c2pt in data_sample_ini[:,:]])
            # data_mean = np.mean(data_cosh_sample,axis=-2)
            data_err = np.sqrt(Ncnfg-1)*np.std(data_cosh_sample,axis=-2)
        self.meff_data_2pt['meff_2pt_%s_mean'%(data_type)] = data_mean
        self.meff_data_2pt['meff_2pt_%s_err'%(data_type)] = data_err
    def PDF(self, data_type:str, flavour:str, link_fold:bool = False): # data, P, ENV, tsep, link, Ncnfg, Nt, dtype=complex
        # the number of data
        if data_type == 'iog':
            N_ENV = 1
            Ncnfg = self.Ncnfg_iog
        elif data_type == 'data':
            N_ENV = self.N_ENV
            Ncnfg = self.Ncnfg_data
        if link_fold == True:
            N_link = self.link_max + 1
            readed_3pt = (
                self.readed['readed_3pt_%s_%s'%(data_type, flavour)][...,self.link_max:,:,:,0] +\
                self.readed['readed_3pt_%s_%s'%(data_type, flavour)][...,:self.link_max + 1,:,:,0][...,::-1,:,:]
                )/2
            self.N_link = N_link
            self.link_max_indx = 0
        elif link_fold == False :
            N_link = self.N_link
            readed_3pt = self.readed['readed_3pt_%s_%s'%(data_type, flavour)][...,0]
            self.link_max_indx = self.link_max
        readed_2pt = self.readed['readed_2pt_%s'%(data_type)][...,0]
        PDF_3pt_2pt_mean = np.zeros((self.N_P, N_ENV, self.N_tsep, N_link, self.Nt), dtype=np.double)
        PDF_3pt_2pt_sample = np.zeros(( self.N_P, N_ENV, self.N_tsep, N_link, Ncnfg, self.Nt), dtype=np.double)
        PDF_3pt_2pt = np.zeros(( self.N_P, N_ENV, self.N_tsep, N_link, Ncnfg, self.Nt), dtype=np.double)
        PDF_3pt_2pt_cov = np.zeros((self.N_P, N_ENV, self.N_tsep, N_link, self.Nt, self.Nt), dtype=np.double)
        # the 3pt/2pt part
        for tsep_indx in range(self.N_tsep):
            PDF_3pt_2pt[:,:,tsep_indx,:,:,:self.tsep[tsep_indx]+1] = readed_3pt[...,tsep_indx,:,:,:self.tsep[tsep_indx]+1] / readed_2pt[...,self.tsep[tsep_indx]].reshape(self.N_P, N_ENV, 1, Ncnfg, 1)
            PDF_3pt_2pt_sample = self.jcknf_sample(PDF_3pt_2pt)
            PDF_3pt_2pt_mean[:,:,tsep_indx,:,:self.tsep[tsep_indx]+1] = np.mean(PDF_3pt_2pt[:,:,tsep_indx,:,:,:self.tsep[tsep_indx]+1], axis=-2)
            #ratio cov
            # cov_mean_dimension = np.asarray(np.shape(PDF_3pt_2pt_sample))
            # cov_mean_dimension[-1] = 1
            # cov_matrix = PDF_3pt_2pt_sample[:,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1] - np.mean(PDF_3pt_2pt_sample[:,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1],axis=-1).reshape(cov_mean_dimension)
            # PDF_3pt_2pt_cov[:,:,:,tsep_indx,:,:tsep_array[tsep_indx]+1,:tsep_array[tsep_indx]+1] = (1/(self.Nt-1)) * np.transpose(cov_matrix,(0,1,2,3,5,4)) @ cov_matrix
        #ratio_err
        self.PDF_mean['PDF_%s_%s'%(data_type, flavour)] = PDF_3pt_2pt_mean
        self.PDF_err['PDF_%s_%s'%(data_type, flavour)] = np.std(PDF_3pt_2pt_sample, axis = -2)*np.sqrt(Ncnfg-1)
        self.PDF_cov['PDF_%s_%s'%(data_type, flavour)] = PDF_3pt_2pt_cov
        # PDF_3pt_2pt_data = np.array([PDF_3pt_2pt_mean,PDF_3pt_2pt_err]) #size = (2,data_1-1,t_sep+1)
    def link_analyse(self, data_type:str, flavour:str):
        if data_type == 'iog':
            N_ENV = 1
        elif data_type == 'data':
            N_ENV = self.N_ENV
    
        link_y = np.zeros((self.N_P, N_ENV, self.N_tsep, self.N_link), dtype=np.double)
        link_err = np.zeros((self.N_P, N_ENV, self.N_tsep, self.N_link), dtype=np.double)
        for tsep_indx in range(self.N_tsep):
            link_y[...,tsep_indx,:] = np.mean(self.PDF_mean['PDF_%s_%s'%(data_type, flavour)][:,:,tsep_indx,:,self.tsep[tsep_indx]//2-5:self.tsep[tsep_indx]//2+5], axis=-1)
            link_err[...,tsep_indx,:] = np.mean(self.PDF_err['PDF_%s_%s'%(data_type, flavour)][:,:,tsep_indx,:,self.tsep[tsep_indx]//2-5:self.tsep[tsep_indx]//2+5], axis=-1)
        self.link_y['link_%s_%s'%(data_type, flavour)] = link_y
        self.link_err['link_%s_%s'%(data_type, flavour)] = link_err
    def plot_meff_2pt(self, data_type, y_range, name:str=None, save_name:str=None):
        if not name : name = '%s_meff_%dx%d_%s_%s'%(self.hadron,self.Nx,self.Nt,self.meff_type,data_type)
        if not save_name : save_name = '%s_meff_%dx%d_%s_%s'%(self.hadron,self.Nx,self.Nt,self.meff_type,data_type)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title(name, fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('t/a'))
        ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
        # ax.set_xticklabels([np.asarray(range(0,t_hlf,3))],fontdict=xylabel_font)
        # ax.set_yticklabels(y_range[-1],fontdict=xylabel_font)
        for P_indx in range(self.N_P):
            N_range = self.meff_data_2pt['meff_2pt_%s_mean'%(data_type)][P_indx, -1].shape[0]
            ax.errorbar(np.arange(N_range), self.meff_data_2pt['meff_2pt_%s_mean'%(data_type)][P_indx, -1], yerr=self.meff_data_2pt['meff_2pt_%s_err'%(data_type)][P_indx, -1], alpha=0.5, marker = self.marker_array[P_indx], capsize=3.5, capthick=1.5, label='P=(%d,%d,%d)'%(self.P[P_indx,0],self.P[P_indx,1],self.P[P_indx,2]), linestyle='none',elinewidth=2) # fmt = 'bs'
        plt.legend(fontsize=18)
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
    def plot_meff_ENV(self, y_range,name:str=None, save_name:str=None):
        if not name : name = '%s_meff_%dx%d_%s_ENV.pdf'%(self.hadron, self.Nx, self.Nt, self.meff_type)
        if not save_name : save_name = '%s_meff_%dx%d_%s_ENV.pdf'%(self.hadron, self.Nx, self.Nt, self.meff_type)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title('%s_meff_%dx%d_%s_ENV'%(self.hadron, self.Nx, self.Nt, self.meff_type),fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('N_ENV'))
        ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
        for P_indx in range(self.N_P):
            Re_2pt_mean_ENV = self.meff_data_2pt['meff_2pt_%s_mean'%('data')][P_indx, :, 12]
            Re_2pt_err_ENV = self.meff_data_2pt['meff_2pt_%s_err'%('data')][P_indx, :, 12]
            ax.errorbar(self.ENV, Re_2pt_mean_ENV, yerr=Re_2pt_err_ENV, alpha=0.5, marker = self.marker_array[P_indx], capsize=3.5, capthick=1.5, label='P=(%d,%d,%d)'%(self.P[P_indx,0],self.P[P_indx,1],self.P[P_indx,2]), linestyle='none',elinewidth=2) # fmt = 'bs'
        plt.legend()
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
    def plot_C2pt_ENV(self, y_range,name:str=None, save_name:str=None):
        if not name : name = '%s_C2pt_%dx%d_ENV'%(self.hadron, self.Nx, self.Nt)
        if not save_name : save_name = '%s_C2pt_%dx%d_ENV'%(self.hadron, self.Nx, self.Nt)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title('%s_C2pt_%dx%d_%s_ENV'%(self.hadron, self.Nx, self.Nt, self.meff_type),fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('N_ENV'))
        ax.set_ylabel('%s'%('$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
        C2pt_mean = np.zeros((self.N_tsep, self.N_ENV))
        C2pt_err = np.zeros((self.N_tsep, self.N_ENV))
        for tsep_indx in range(self.N_tsep):
            C2pt_ENV = self.readed['readed_2pt_data'][0,:,:,self.tsep[tsep_indx],0].T / (self.ENV * self.Nt)
            C2pt_mean, C2pt_err = self.jackknife_ctr_err(C2pt_ENV)
            ax.errorbar(self.ENV, C2pt_mean, yerr=C2pt_err, alpha=0.5, marker = self.marker_array[tsep_indx], capsize=3.5, capthick=1.5, label='tsep%d'%(self.tsep[tsep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
            plt.legend()
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
    def plot_PDF_ENV(self, flavour, y_range, name:str=None, save_name:str=None):
        if not name : name = '%s_PDF_%dx%d_ENV_%s'%(self.hadron, self.Nx, self.Nt, flavour)
        if not save_name : save_name = '%s_PDF_%dx%d_ENV_%s'%(self.hadron, self.Nx, self.Nt, flavour)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title(name,fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('ENV'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$/$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
        for i in range(len(self.link_y.keys())):
            data_name = str(list(self.link_y.keys())[i])
            if data_name == 'link_iog_U': N_ENV_change = self.ENV[-1]; label = 'chroma'
            elif data_name == 'link_data_D': N_ENV_change = self.ENV; label = 'dis_reduse'
            elif data_name == 'link_data_U': N_ENV_change = self.ENV; label = 'dis_none'
            for tsep_indx in range(self.N_tsep):
                ax.errorbar(N_ENV_change, self.link_y[data_name][0,:,tsep_indx,self.link_max_indx], yerr=self.link_err[data_name][0,:,tsep_indx,self.link_max_indx], alpha=0.5, marker = self.marker_array[tsep_indx], capsize=3.5, capthick=1.5, label='%s'%(label), linestyle='none',elinewidth=2) # fmt = 'bs'
                plt.legend(loc=2)
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
    def plot_link_C3pt_C3pt(self, y_range, num:int=3, name:str=None, save_name:str=None):
        if not name : name = '%s_link_%dx%d_C3_C3.pdf'%(self.hadron, self.Nx, self.Nt)  
        if not save_name : save_name = '%s_link_%dx%d_C3_C3.pdf'%(self.hadron, self.Nx, self.Nt)  
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title(name,fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('Z'))
        ax.set_ylabel('%s'%('$C_{\mathrm{3pt}}$ / $C_{\mathrm{3pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        for i in range(len(self.link_y.keys())):
            data_name = str(list(self.link_y.keys())[i])
            if data_name == 'link_iog_U': N_ENV_change = range(1); ENV_array_change=np.asarray([-1]); data_type = 'chroma'; N_ENV = 1
            elif data_name == 'link_data_D': N_ENV_change = range(-num,0,1); ENV_array_change = self.ENV; data_type = 'dis_reduse'; N_ENV = self.N_ENV
            elif data_name == 'link_data_U': N_ENV_change = range(-num,0,1); ENV_array_change = self.ENV; data_type = 'dis_none'; N_ENV = self.N_ENV
            link_y_change = self.link_y[data_name] / self.link_y[data_name][...,self.link_max_indx].reshape(self.N_P ,N_ENV, self.N_tsep, 1)
            link_err_change = self.link_err[data_name] / self.link_y[data_name][...,self.link_max_indx].reshape(self.N_P, N_ENV, self.N_tsep, 1)
            for ENV_indx in N_ENV_change:
                ax.errorbar(range(link_y_change.shape[-1]), link_y_change[0,ENV_indx,0], yerr=link_err_change[0,ENV_indx,0], alpha=0.5, marker = self.marker_array[ENV_indx], capsize=3.5, capthick=1.5, label='%s-ENV%d'%(data_type,ENV_array_change[ENV_indx]),linestyle='none',elinewidth=2) # fmt = 'bs'
                plt.legend()
   
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
    def plot_link_ratio(self, y_range, num:int=3, name:str=None, save_name:str=None):
        if not name : name = '%s_link_%dx%d_ratio'%(self.hadron, self.Nx, self.Nt)
        if not save_name : save_name = '%s_link_%dx%d_ratio'%(self.hadron, self.Nx, self.Nt)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title(name,fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('Z'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        for i in range(len(self.link_y.keys())):
            data_name = str(list(self.link_y.keys())[i])
            link_y_change = self.link_y[data_name]
            link_err_change = self.link_err[data_name]
            if data_name == 'link_iog_U': N_ENV_change = range(1); ENV_array_change=np.asarray(['41472']); data_type = 'exact'
            elif data_name == 'link_data_D': N_ENV_change = range(-num,0,1); ENV_array_change = np.asarray([str(self.ENV[i]) for i in range(self.N_ENV)]); data_type = 'dis_reduse'; mark = -1
            elif data_name == 'link_data_U': N_ENV_change = range(-num,0,1); ENV_array_change = np.asarray([str(self.ENV[i]) for i in range(self.N_ENV)]); data_type = 'dis_none'; mark = -5
            if self.link_max_indx == self.link_max : normal = self.link_max
            else : normal = 0
            for ENV_indx in N_ENV_change:
                ax.errorbar(range(link_y_change.shape[-1]) - normal, link_y_change[0,ENV_indx,0], yerr=link_err_change[0,ENV_indx,0], alpha=0.5, marker = self.marker_array[ENV_indx], capsize=3.5, capthick=1.5, label='%s_ENV:%s'%(data_type,ENV_array_change[ENV_indx]),linestyle='none',elinewidth=2) # fmt = 'bs'
                plt.legend()
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
    def plot_link_indx_ENV(self, indx, y_range, name:str=None, save_name:str=None, normal:int=0, fold:int=0):
        if not name : name = '%s_link_indx%d_%dx%d_ratio'%(self.hadron, indx, self.Nx, self.Nt)
        if not save_name : save_name = '%s_link_indx%d_%dx%d_ratio'%(self.hadron, indx, self.Nx, self.Nt)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title(name, fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('N_ENV'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        for i in range(len(self.link_y.keys())):
            data_name = str(list(self.link_y.keys())[i])
            if data_name == 'link_iog_U': N_ENV_change = self.ENV[-1]; data_type = 'chroma';  N_ENV = 1
            elif data_name == 'link_data_D': N_ENV_change = self.ENV[:self.link_y[data_name].shape[1]]; data_type = 'dis_reduse'; N_ENV = self.N_ENV
            elif data_name == 'link_data_U': N_ENV_change = self.ENV[:self.link_y[data_name].shape[1]]; data_type = 'dis_none'; N_ENV = self.N_ENV
            if normal == 1:
                link_y_change = self.link_y[data_name] / self.link_y[data_name][...,self.link_max_indx].reshape(self.N_P,N_ENV,self.N_tsep,1)
                link_err_change = self.link_err[data_name] / self.link_y[data_name][...,self.link_max_indx].reshape(self.N_P,N_ENV,self.N_tsep,1)
            else:
                link_y_change = self.link_y[data_name]
                link_err_change = self.link_err[data_name]
            ax.errorbar(N_ENV_change, link_y_change[0,:,0, indx+self.link_max_indx], yerr=link_err_change[0,:,0, indx+self.link_max_indx], alpha=0.5, marker = self.marker_array[i], capsize=3.5, capthick=1.5, label='%s_indx%d'%(data_type, indx),linestyle='none',elinewidth=2) # fmt = 'bs'
            plt.legend()
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
    def plot_fit(self, y_range, flavour, X, Y, name:str=None, save_name:str=None, normal:int=0):
        if not name : name = '%s_PDF_%dx%d_ENV_%s_fit'%(self.hadron, self.Nx, self.Nt, flavour)
        if not save_name : save_name = '%s_PDF_%dx%d_ENV_%s_fit'%(self.hadron, self.Nx, self.Nt, flavour)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title(name,fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('N_ENV'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$/$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
        for i in range(1):
            data_name = 'link_data_U'; N_ENV_change = self.ENV; label = 'distillation'
            for tsep_indx in range(self.N_tsep):
                ax.errorbar(N_ENV_change, self.link_y[data_name][0,:,tsep_indx,self.link_max_indx], yerr=self.link_err[data_name][0,:,tsep_indx,self.link_max_indx], alpha=0.5, marker = self.marker_array[tsep_indx], capsize=3.5, capthick=1.5, label='%s_tsep%d'%(label,self.tsep[tsep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
                plt.legend(loc=2)
        for i in range(2):
            ax.errorbar(X, Y, yerr=np.zeros_like(X), alpha=0.5, marker = self.marker_array[tsep_indx], capsize=3.5, capthick=1.5, label='%s_tsep%d'%('fit',self.tsep[tsep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
def min_fun_descent(function, X:np.ndarray, X0:np.ndarray, type:str):
    alpha = 1e-12
    delta = 1e-12
    max_cycle_index = 10000
    cycle_index = 0
    br = sp.symbols('br')
    N = X.shape[0]
    d_fun_value = np.zeros(N)
    d_fun = np.asarray([sp.diff(function, X[n]) for n in range(N)])
    if type == 'multinomial' :
        X0_dic = sp.solve([d_fun[n] for n in range(N)],[X[n]for n in range(N)])
        X0 = [X0_dic[indx] for indx in X]
    d_fun_value = np.asarray([(float)(d_fun[n].subs({X[i]: X0[i] for i in range(N)}).evalf())  for n in range(N)])
    while (cycle_index <= max_cycle_index and sum(abs(d_value) for d_value in d_fun_value ) >= delta):
        cycle_index += 1
        X0_mid = X0 - d_fun_value * br
        # X0_mid = [ X0[n] - d_fun_value[n] * br  for n in range(N)] # * np.exp(abs(d_fun_value_2)-abs(dd_fun_value_2)) # * u
        beta = float(abs(sp.solve((function.subs({X[i]: X0_mid[i] for i in range(N)}).evalf()))[0]))
        X0 = X0 - d_fun_value * beta * alpha 
        print('beta:',beta)
        d_fun_value = np.asarray([(float)(d_fun[n].subs({X[i]: X0[i] for i in range(N)}).evalf()) for n in range(N)])
        function_value = (float)(function.subs({X[i]: X0[i] for i in range(N)}).evalf())
        print('cycle_index:%d'%(cycle_index))
        print('X0:',X0)
        print('d_fun_value:',d_fun_value)
        print('function_value:%f'%(function_value))
        print('\n')
    function_value = (float)(function.subs({X[i]: X0[i] for i in range(N)}).evalf())
    print(function_value)
    return X0, function_value
def min_fun_newton(function, X_1, X_2, X0_1, X0_2):
    max_cycle_index = 200
    alpha = 0.01
    delta = 0.0001
    cycle_index = 0
    d_fun_1 = sp.diff(function, X_1)[0]
    d_fun_2 = sp.diff(function, X_2)[0]
    dd_fun_1  = sp.diff(function, X_1, 2)[0]
    dd_fun_12 = sp.diff(function, X_1, X_2)[0]
    dd_fun_2  = sp.diff(function, X_2, 2)[0]
    d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf()) 
    d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    dd_fun_value_1  = (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
    dd_fun_value_2  = (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    dd_fun_value_12 = (float)(dd_fun_12.subs({X_1: X0_1, X_2: X0_2}).evalf())
    while (cycle_index <= max_cycle_index and abs(d_fun_value_1)+abs(d_fun_value_2) >= delta):
        cycle_index += 1
        G = np.vstack((d_fun_value_1,d_fun_value_2))
        H = np.vstack(([dd_fun_value_1,dd_fun_value_12],[dd_fun_value_12,dd_fun_value_2]))
        z_value = np.vstack((X0_1, X0_2))-np.linalg.inv(H)@G
        X0_1 = (float)(z_value[0])
        X0_2 = (float)(z_value[1])
        d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf()) 
        d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
        dd_fun_value_1  = (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
        dd_fun_value_2  = (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
        dd_fun_value_12 = (float)(dd_fun_12.subs({X_1: X0_1, X_2: X0_2}).evalf())
        print('cycle_index:%d\n X0_1:%f X0_2:%f\n d_fun_value_1:%f  d_fun_value_2:%f\n dd_fun_value_1:%f  dd_fun_value_2:%f\n'%(cycle_index,X0_1,X0_2,d_fun_value_1,d_fun_value_2, dd_fun_value_1, dd_fun_value_2))
    function_value = (float)(function[0].subs({X_1: X0_1, X_2: X0_2}).evalf())
    parameter = np.array([X0_1, X0_2, function_value])
    return parameter
def c_square(data, function, X_1, X_2, X0_1, X0_2):
    n = data.shape[0]
    t = data.shape[1]
    c_square_ni = 0
    c_square_ni = np.zeros((n,3))
    cov_m_A = np.cov(data.T)
    cov_m_inv = np.linalg.inv(cov_m_A)
    c_square_fun = ((data[0,:].reshape(t,1)-function).T)@cov_m_inv@(data[0,:].reshape(t,1)-function)
    c_square_ni[0] = min_fun_newton(c_square_fun, X_1, X_2, X0_1, X0_2)
    for ni in range(1,n):
        c_square_fun = ((data[ni,:].reshape(t,1)-function).T)@cov_m_inv@(data[ni,:].reshape(t,1)-function)
        c_square_ni[ni] = min_fun_newton(c_square_fun, X_1, X_2, c_square_ni[ni-1, 0], c_square_ni[ni-1, 1])
    c_square_mean = np.sum(c_square_ni, axis = 0)/n
    return c_square_mean
