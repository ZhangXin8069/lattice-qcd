#!/public/home/xinghy/anaconda3-2023.03/envs/cupy114/bin/python
import numpy as np
import cupy as cp
import os
import fileinput
from gamma_matrix_cupy import *
from input_output import *
from opt_einsum import contract
import time
infile=fileinput.input()
for line in infile:
	tmp=line.split()
	if(tmp[0]=='Nt'):
		Nt=int(tmp[1])
	if(tmp[0]=='Nx'):
		Nx=int(tmp[1])
	if(tmp[0]=='conf_id'):
		conf_id=tmp[1]
	if(tmp[0]=='Nev'): #number of eigenvectors in perambulators
		Nev=int(tmp[1])
	if(tmp[0]=='Nev1'):  #number of eigenvectors used in contraction
		Nev1=int(tmp[1])
	if(tmp[0]=='NevVVV'): #number of eigenvector in VVV data
		NevVVV=int(tmp[1])
	if(tmp[0]=='tsource_interval'):
		tsource_interval=int(tmp[1])
	if(tmp[0]=='tsource_start'):
		tsource_start=int(tmp[1])
	if(tmp[0]=='tsep_min'):
		tsep_min=int(tmp[1])
	if(tmp[0]=='tsep_max'):
		tsep_max=int(tmp[1])
	if(tmp[0]=='peram_u_dir'):
		peram_u_dir=tmp[1]
	if(tmp[0]=='peram_c_dir'):
		peram_c_dir=tmp[1]
	if(tmp[0]=='VVV_dir'):
		VVV_dir=tmp[1]
	if(tmp[0]=='corr_dir'):
		corr_dir=tmp[1]
#mom to be calculated for the baryon twopt
mom_baryon=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,0,1],[-1,0,-1],[0,1,1],[0,-1,-1],[1,-1,0],[-1,1,0],[1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2]])
#mom_baryon=np.array([[0,0,0],[0,0,1],[0,1,1],[1,1,1],[0,0,2]])
mom_baryon1=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,0,1],[-1,0,-1],[0,1,1],[0,-1,-1],[1,-1,0],[-1,1,0],[1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1]]) # for Sig_c
number_of_mom=mom_baryon.shape[0]
number_of_mom1=mom_baryon1.shape[0]
gsink_baryon=gamma(7)
gsource_baryon=gamma(7)
VVV=np.zeros((number_of_mom,Nt,Nev1,Nev1,Nev1),dtype=complex)
for _n in range(number_of_mom):
	print(_n)
	st=time.time()
	VVV[_n] = readin_VVV_cpu(VVV_dir, NevVVV, Nev1, Nt, conf_id, mom_baryon[_n,0], mom_baryon[_n,1], mom_baryon[_n,2])
	ed=time.time()
	print("read VVV %d done, time used: %.6f s" %(_n, ed-st))
twopt_Lamc=cp.zeros((number_of_mom, number_of_mom, Nt,Nt,2,2),dtype=complex)
twopt_1_Sigc=cp.zeros((number_of_mom1, number_of_mom1, Nt,Nt,2,2),dtype=complex)
twopt_2_Sigc=cp.zeros((number_of_mom1, number_of_mom1, Nt,Nt,2,2),dtype=complex)
for t_source in range(tsource_start,Nt,tsource_interval):
	st0 = time.time()
	st = time.time()
	peram_u=readin_peram(peram_u_dir, conf_id, Nt, Nev, Nev1, t_source)
	peram_c=readin_peram(peram_c_dir, conf_id, Nt, Nev, Nev1, t_source)
	ed = time.time()
	print("read peram_u done, time used: %.3f s" %(ed-st))
	VVV_source=cp.array(VVV[:,t_source])
	
	for t_sink in range(0,Nt):
		if((t_sink-t_source+Nt)%Nt<tsep_min or (t_sink-t_source+Nt)%Nt>tsep_max):
			continue
		VVV_sink=cp.array(VVV[:,t_sink])
############   Nucleon 2pt  ################
#diagram 1
		st=time.time()
		twopt_Lamc[:, :, t_sink, t_source]=contract("vabc, afkp, bglq, chmr, kl, pq, wfgh -> vwmr", VVV_sink, peram_u[t_sink], peram_u[t_sink], peram_c[t_sink,:,:,0:2,0:2], gsink_baryon, gsource_baryon, cp.conj(VVV_source))
		twopt_1_Sigc[:, :, t_sink, t_source]=contract("vabc, afkp, bglq, chmr, kl, pq, wfgh -> vwmr", VVV_sink[0:number_of_mom1], peram_u[t_sink], peram_c[t_sink], peram_u[t_sink,:,:,0:2,0:2], gsink_baryon, gsource_baryon, cp.conj(VVV_source[0:number_of_mom1]))
		twopt_2_Sigc[:, :, t_sink, t_source]=contract("vabc, ahkr, bglq, cfmp, kl, pq, wfgh -> vwmr", VVV_sink[0:number_of_mom1], peram_u[t_sink,:,:,:,0:2], peram_c[t_sink], peram_u[t_sink,:,:,0:2,:], gsink_baryon, gsource_baryon, cp.conj(VVV_source[0:number_of_mom1]))
		ed=time.time()
		print("Lamc Sigc 2pt t_source: %d, t_sink: %d contraction done, %.6f s" %(t_source, t_sink, ed-st))
#diagram 2	
	del peram_u, peram_c
	cp._default_memory_pool.free_all_blocks()
corr=cp.asnumpy(twopt_Lamc)
np.savez("%s/corr_Lamc_conf%s.npz" %(corr_dir, conf_id), corr=corr, mom=mom_baryon)
corr=cp.asnumpy(twopt_1_Sigc)
np.savez("%s/corr_Sigc_fig1_conf%s.npz" %(corr_dir, conf_id), corr=corr, mom=mom_baryon1)
corr=cp.asnumpy(twopt_2_Sigc)
np.savez("%s/corr_Sigc_fig2_conf%s.npz" %(corr_dir, conf_id), corr=corr, mom=mom_baryon1)
