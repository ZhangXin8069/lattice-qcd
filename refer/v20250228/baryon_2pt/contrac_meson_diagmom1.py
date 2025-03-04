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
	if(tmp[0]=='NevVdV'): #number of eigenvector in VdV data
		NevVdV=int(tmp[1])
	if(tmp[0]=='tsource_interval'):
		tsource_interval=int(tmp[1])
	if(tmp[0]=='tsource_start'):
		tsource_start=int(tmp[1])
	if(tmp[0]=='peram_u_dir'):
		peram_u_dir=tmp[1]
	if(tmp[0]=='peram_c_dir'):
		peram_c_dir=tmp[1]
	if(tmp[0]=='VdV_dir'):
		VdV_dir=tmp[1]
	if(tmp[0]=='corr_dir'):
		corr_dir=tmp[1]
#mom to be calculated for the baryon twopt
mom=cp.array([[0,0,0]])
number_of_mom=mom.shape[0]
gamma_meson=cp.array([gamma(5), gamma(1), gamma(2), gamma(3)])
number_of_gamma=4
gsink_meson=cp.zeros((number_of_gamma,4,4), dtype=complex)
gsource_meson=cp.zeros((number_of_gamma,4,4), dtype=complex)
for _n in range(number_of_gamma):
	gsink_meson[_n] = gamma(5)@gamma_meson[_n]
	gsource_meson[_n] = gamma_meson[_n]@gamma(5)
VdV=cp.zeros((number_of_mom,Nt,Nev1,Nev1),dtype=complex)
for _n in range(1,number_of_mom):
	print(_n)
	st=time.time()
	VdV[_n] = readin_VdV(VdV_dir, NevVdV, Nev1, Nt, conf_id, mom[_n,0],mom[_n,1], mom[_n,2])
	ed=time.time()
	print("read VdV %d done, time used: %.6f s" %(_n, ed-st))
for _t in range(Nt):
	VdV[0, _t]=cp.identity(Nev1,dtype=complex)
twopt_D=cp.zeros((number_of_mom, number_of_gamma, Nt,Nt),dtype=complex)
twopt_Etac=cp.zeros((number_of_mom, number_of_gamma, Nt,Nt),dtype=complex)
twopt_D_sum=cp.zeros((number_of_mom, number_of_gamma, 1, Nt),dtype=complex)
twopt_Etac_sum=cp.zeros((number_of_mom, number_of_gamma, 1, Nt),dtype=complex)
for t_source in range(tsource_start,Nt,tsource_interval):
	st0 = time.time()
	st = time.time()
	peram_u=readin_peram(peram_u_dir, conf_id, Nt, Nev, Nev1, t_source)
	ed = time.time()
	print("read peram_u done, time used: %.3f s" %(ed-st))
	st = time.time()
	peram_c=readin_peram(peram_c_dir, conf_id, Nt, Nev, Nev1, t_source)
	ed = time.time()
	print("read peram_c done, time used: %.3f s" %(ed-st))
	st=time.time()
	twopt_D[:, :,:,t_source]=contract("wade, adins, aejot, xno, xts, wij -> wxa", VdV, cp.conj(peram_c), peram_u, gsink_meson, gsource_meson, cp.conj(VdV[:, t_source]))
	twopt_Etac[:, :,:,t_source]=contract("wade, adins, aejot, xno, xts, wij -> wxa", VdV, cp.conj(peram_c), peram_c, gsink_meson, gsource_meson, cp.conj(VdV[:, t_source]))
	ed=time.time()
	print("twopt t_source: %d, contraction done, %.6f s" %(t_source, ed-st))
	for t_sink in range(Nt):
		twopt_D_sum[:,:,0,(t_sink-t_source+Nt)%Nt]=twopt_D_sum[:,:,0,(t_sink-t_source+Nt)%Nt]+ twopt_D[:,:,t_sink, t_source] 
		twopt_Etac_sum[:,:,0,(t_sink-t_source+Nt)%Nt]=twopt_Etac_sum[:,:,0,(t_sink-t_source+Nt)%Nt]+twopt_Etac[:,:,t_sink, t_source] 
corr=cp.asnumpy(twopt_D)
np.savez("%s/corr_D_conf%s.npz" %(corr_dir, conf_id), corr=corr,mom=mom,gamma=gamma_meson)
corr=cp.asnumpy(twopt_Etac)
np.savez("%s/corr_Etac_conf%s.npz" %(corr_dir, conf_id), corr=corr,mom=mom,gamma=gamma_meson)
twopt_D_sum=cp.asnumpy(twopt_D_sum)
twopt_Etac_sum=cp.asnumpy(twopt_Etac_sum)
for _i in range(number_of_mom):
	write_data_ascii(twopt_D_sum[_i,0], Nt, Nx, "%s/corr_D_Px%iPy%iPz%i_conf%s.dat"%(corr_dir,mom[_i,0],mom[_i,1],mom[_i,2],conf_id))
	write_data_ascii(twopt_Etac_sum[_i,0], Nt, Nx, "%s/corr_Etac_Px%iPy%iPz%i_conf%s.dat"%(corr_dir,mom[_i,0],mom[_i,1],mom[_i,2],conf_id))
	for _k in range(1,4):
		write_data_ascii(twopt_D_sum[_i,_k], Nt, Nx, "%s/corr_Dstar_gamma%i_Px%iPy%iPz%i_conf%s.dat"%(corr_dir,_k,mom[_i,0],mom[_i,1],mom[_i,2],conf_id))
		write_data_ascii(twopt_Etac_sum[_i,_k], Nt, Nx, "%s/corr_JPsi_gamma%i_Px%iPy%iPz%i_conf%s.dat"%(corr_dir,_k,mom[_i,0],mom[_i,1],mom[_i,2],conf_id))
