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
mom=np.zeros((33,3),dtype=int)
i=0
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
	if(tmp[0]=='number_of_mom'):
		number_of_mom=int(tmp[1])
	if(len(tmp)==3):
		mom[i,0]=int(tmp[0])
		mom[i,1]=int(tmp[1])
		mom[i,2]=int(tmp[2])
		i=i+1
#mom to be calculated for the baryon twopt
#mom=np.array([[0,0,0],[0,0,1],[0,1,1],[1,1,1],[0,0,2]])
#number_of_mom=mom.shape[0]
#gsink_baryon=gamma(7)
#gsource_baryon=gamma(7)
gsink_baryon=gamma(13)
gsource_baryon=gamma(13)
VVV=np.zeros((number_of_mom,Nt,Nev1,Nev1,Nev1),dtype=complex)
for _n in range(number_of_mom):
	print(_n)
	st=time.time()
	VVV[_n] = readin_VVV_cpu(VVV_dir, NevVVV, Nev1, Nt, conf_id, mom[_n,0], mom[_n,1], mom[_n,2])
#	for _t in range(Nt):
#		VVV[_n, _t] = readin_VVV_cpu_t(VVV_dir, _t, NevVVV, Nev1, conf_id, mom[_n,0], mom[_n,1], mom[_n,2])
	ed=time.time()
	print("read VVV %d done, time used: %.6f s" %(_n, ed-st))
twopt_1_N=cp.zeros((number_of_mom,Nt,Nt,2,2),dtype=complex)
twopt_2_N=cp.zeros((number_of_mom, Nt,Nt,2,2),dtype=complex)
twopt_1_Lamc=cp.zeros((number_of_mom, Nt,Nt,2,2),dtype=complex)
twopt_1_Sigc=cp.zeros((number_of_mom, Nt,Nt,2,2),dtype=complex)
twopt_2_Sigc=cp.zeros((number_of_mom, Nt,Nt,2,2),dtype=complex)
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
		twopt_1_N[:, t_sink, t_source]=contract("vabc, afkp, bglq, chmr, kl, pq, vfgh -> vmr", VVV_sink, peram_u[t_sink], peram_u[t_sink], peram_u[t_sink,:,:,0:2,0:2], gsink_baryon, gsource_baryon, cp.conj(VVV_source))
		twopt_2_N[:, t_sink, t_source]=contract("vabc, ahkr, bglq, cfmp, kl, pq, vfgh -> vmr", VVV_sink, peram_u[t_sink,:,:,:,0:2], peram_u[t_sink], peram_u[t_sink,:,:,0:2,:], gsink_baryon, gsource_baryon, cp.conj(VVV_source))
		twopt_1_Lamc[:, t_sink, t_source]=contract("vabc, afkp, bglq, chmr, kl, pq, vfgh -> vmr", VVV_sink, peram_u[t_sink], peram_u[t_sink], peram_c[t_sink,:,:,0:2,0:2], gsink_baryon, gsource_baryon, cp.conj(VVV_source))
		twopt_1_Sigc[:, t_sink, t_source]=contract("vabc, afkp, bglq, chmr, kl, pq, vfgh -> vmr", VVV_sink[0:number_of_mom], peram_u[t_sink], peram_c[t_sink], peram_u[t_sink,:,:,0:2,0:2], gsink_baryon, gsource_baryon, cp.conj(VVV_source[0:number_of_mom]))
		twopt_2_Sigc[:, t_sink, t_source]=contract("vabc, ahkr, bglq, cfmp, kl, pq, vfgh -> vmr", VVV_sink[0:number_of_mom], peram_u[t_sink,:,:,:,0:2], peram_c[t_sink], peram_u[t_sink,:,:,0:2,:], gsink_baryon, gsource_baryon, cp.conj(VVV_source[0:number_of_mom]))
		ed=time.time()
		print("Lamc Sigc 2pt t_source: %d, t_sink: %d contraction done, %.6f s" %(t_source, t_sink, ed-st))
#diagram 2	
	del peram_u, peram_c
	cp._default_memory_pool.free_all_blocks()
twopt_N=twopt_1_N[:,:,:,0,0]+twopt_1_N[:,:,:,1,1]-twopt_2_N[:,:,:,0,0]-twopt_2_N[:,:,:,1,1]
twopt_Lamc=twopt_1_Lamc[:,:,:,0,0]+twopt_1_Lamc[:,:,:,1,1]
twopt_Sigc=twopt_1_Sigc[:,:,:,0,0]+twopt_1_Sigc[:,:,:,1,1]-twopt_2_Sigc[:,:,:,0,0]-twopt_2_Sigc[:,:,:,1,1]
twopt_N=cp.asnumpy(twopt_N)
twopt_Lamc=cp.asnumpy(twopt_Lamc)
twopt_Sigc=cp.asnumpy(twopt_Sigc)
np.savez("%s/baryon.conf%s.npz" %(corr_dir, conf_id), corr_N = twopt_N, corr_Lamc=twopt_Lamc, corr_Sigc=twopt_Sigc, mom=mom)
