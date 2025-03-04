#!/public/home/xinghy/anaconda3-2023.03/envs/cupy114/bin/python
import numpy as np
import os
import fileinput
from gamma_matrix_cpu import *
from input_output_cpu import *
from opt_einsum import contract
from functions import *
import time
import multiprocessing as mp
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
	if(tmp[0]=='NevVdV'): #number of eigenvector in VdV data
		NevVdV=int(tmp[1])
	if(tmp[0]=='tsource_interval'):
		tsource_interval=int(tmp[1])
	if(tmp[0]=='tsource_start'):
		tsource_start=int(tmp[1])
	# if(tmp[0]=='tsep_min'):
	# 	tsep_min=int(tmp[1])
	# if(tmp[0]=='tsep_max'):
	# 	tsep_max=int(tmp[1])
	if(tmp[0]=='peram_u_dir'):
		peram_u_dir=tmp[1]
	if(tmp[0]=='peram_c_dir'):
		peram_c_dir=tmp[1]
	if(tmp[0]=='peram_s_dir'):
		peram_s_dir=tmp[1]
	if(tmp[0]=='VdV_dir'):
		VdV_dir=tmp[1]
	if(tmp[0]=='corr_dir'):
		corr_dir=tmp[1]
	if(tmp[0]=='number_of_processes'):
		number_of_processes=int(tmp[1])
#mom to be calculated for the baryon twopt
# mom=np.array([[0,0,0],[0,0,1],[0,1,1],[1,1,1],[0,0,2]])
mom=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,0,1],[-1,0,-1],[0,1,1],[0,-1,-1],[1,-1,0],[-1,1,0],[1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2]])
number_of_mom=mom.shape[0]
gamma_meson=np.array([gamma(5), gamma(1), gamma(2), gamma(3)])
# gamma_meson=np.array([gamma(3)@gamma(5)])   #K1
number_of_gamma=gamma_meson.shape[0]
VdV=np.zeros((number_of_mom,Nt,Nev1,Nev1),dtype=complex)
for _n in range(1,number_of_mom):
	print(_n)
	st=time.time()
	VdV[_n,:,:,:] = readin_VdV_all(VdV_dir, NevVdV, Nev1, Nt, conf_id, mom[_n,0],mom[_n,1], mom[_n,2])
	ed=time.time()
	print("read VdV %d done, time used: %.6f s" %(_n, ed-st))
for _t in range(Nt):
	VdV[0, _t]=np.identity(Nev1,dtype=complex)
# twopt_K=np.zeros((number_of_mom, number_of_mom, number_of_gamma, number_of_gamma, Nt,Nt),dtype=complex)
# twopt_Etac=np.zeros((number_of_mom, number_of_mom, number_of_gamma, number_of_gamma, Nt,Nt),dtype=complex)
# Pi
stt=time.time()
all_args=[[conf_id, tsource, Nt, Nev, Nev1, peram_u_dir, VdV, mom, gamma_meson] for tsource in range(tsource_start,Nt,tsource_interval)]
with mp.Pool(processes = number_of_processes) as p:
	results = p.starmap(meson_2pt_self1, all_args)
twopt_Pi=np.array(results).transpose(1,2,0,3)
edt=time.time()
print("twopt_Pi contraction done, %.6f s" %(edt-stt))
np.savez("%s/Pi/corr_uu_conf%s.npz" %(corr_dir, conf_id), corr=twopt_Pi, mom=mom, gamma=gamma_meson)
# K
stt=time.time()
all_args=[[conf_id, tsource, Nt, Nev, Nev1, peram_s_dir, peram_u_dir, VdV, mom, gamma_meson] for tsource in range(tsource_start,Nt,tsource_interval)]
with mp.Pool(processes = number_of_processes) as p:
	results = p.starmap(meson_2pt_self2, all_args)
twopt_K=np.array(results).transpose(1,2,0,3)
edt=time.time()
print("twopt_K contraction done, %.6f s" %(edt-stt))
np.savez("%s/K/corr_su_conf%s.npz" %(corr_dir, conf_id), corr=twopt_K, mom=mom, gamma=gamma_meson)
# Etac
stt=time.time()
all_args=[[conf_id, tsource, Nt, Nev, Nev1, peram_c_dir, VdV, mom, gamma_meson] for tsource in range(tsource_start,Nt,tsource_interval)]
with mp.Pool(processes = number_of_processes) as p:
	results = p.starmap(meson_2pt_self1, all_args)
twopt_Etac=np.array(results).transpose(1,2,0,3)
edt=time.time()
print("twopt_Etac contraction done, %.6f s" %(edt-stt))
np.savez("%s/Etac/corr_cc_conf%s.npz" %(corr_dir, conf_id), corr=twopt_Etac, mom=mom, gamma=gamma_meson)
# D
stt=time.time()
all_args=[[conf_id, tsource, Nt, Nev, Nev1, peram_u_dir, peram_c_dir, VdV, mom, gamma_meson] for tsource in range(tsource_start,Nt,tsource_interval)]
with mp.Pool(processes = number_of_processes) as p:
	results = p.starmap(meson_2pt_self2, all_args)
twopt_D=np.array(results).transpose(1,2,0,3)
edt=time.time()
print("twopt_D contraction done, %.6f s" %(edt-stt))
np.savez("%s/D/corr_uc_conf%s.npz" %(corr_dir, conf_id), corr=twopt_D, mom=mom, gamma=gamma_meson)
# Ds
stt=time.time()
all_args=[[conf_id, tsource, Nt, Nev, Nev1, peram_s_dir, peram_c_dir, VdV, mom, gamma_meson] for tsource in range(tsource_start,Nt,tsource_interval)]
with mp.Pool(processes = number_of_processes) as p:
	results = p.starmap(meson_2pt_self2, all_args)
twopt_Ds=np.array(results).transpose(1,2,0,3)
edt=time.time()
print("twopt_Ds contraction done, %.6f s" %(edt-stt))
np.savez("%s/Ds/corr_sc_conf%s.npz" %(corr_dir, conf_id), corr=twopt_Ds, mom=mom, gamma=gamma_meson)