import numpy as np
import cupy as cp
import os
import fileinput
from gamma_matrix_cupy import *
from input_output import *
from opt_einsum import contract
import time
infile=fileinput.input()
mom=np.zeros((20,3),dtype=int)
i=0
for line in infile:
	print(1)
	tmp=line.split()
	if(tmp[0]=='Nt'):
		Nt=int(tmp[1])
		print("Nt = ",Nt)
	if(tmp[0]=='Nx'):
		Nx=int(tmp[1])
	if(tmp[0]=='conf_id'):
		conf_id=tmp[1]
	if(tmp[0]=='Nev'): #number of eigenvectors in perambulators
		Nev=int(tmp[1])
	if(tmp[0]=='Nev1'):  #number of eigenvectors used in contraction
		Nev1=int(tmp[1])
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
	if(tmp[0]=='eig_dir'):
		eig_dir=tmp[1]
	if(tmp[0]=='corr_dir'):
		corr_dir=tmp[1]
	if(tmp[0]=='number_of_mom'):
		number_of_mom=int(tmp[1])
	if(len(tmp)==3):
		mom[i][0]=int(tmp[0])
		mom[i][1]=int(tmp[1])
		mom[i][2]=int(tmp[2])
		i=i+1
#mom to be calculated for the baryon twopt
gsink_baryon=gamma(7)
gsource_baryon=gamma(7)
VVV=np.zeros((number_of_mom,Nt,Nev1,Nev1,Nev1),dtype=complex)
twopt_1_N=cp.zeros((number_of_mom, Nt,Nt,2,2),dtype=complex)
twopt_2_N=cp.zeros((number_of_mom, Nt,Nt,2,2),dtype=complex)
#twopt_N=cp.zeros((number_of_mom,number_of_mom,Nt,Nt,2,2), dtype=complex)
phase=cp.zeros((number_of_mom, Nx*Nx*Nx), dtype=complex)
epsi=cp.array([[0,1,2],[0,2,1],[1,2,0],[1,0,2],[2,0,1],[2,1,0]])
sign=cp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
st=time.time()
for _n in range(number_of_mom):
	Px=mom[_n,0]
	Py=mom[_n,1]
	Pz=mom[_n,2]
	for z in range(Nx):
		for y in range(Nx):
			for x in range(Nx):
				phase[_n, z*Nx*Nx+y*Nx+x]=cp.exp(-1j*2.0*cp.pi*(Px*x+Py*y+Pz*z)/Nx)
ed=time.time()
print("phase done in %.6f s" %(ed-st))
for t in range(0,Nt):
	print("t:%d" %t)
	st=time.time()
	V=readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx)
	ed=time.time()
	print("read in done in %.3f s"%(ed-st))
	for _n in range(number_of_mom):
		Px=mom[_n,0]
		Py=mom[_n,1]
		Pz=mom[_n,2]
		st=time.time()
		VVV_temp=cp.zeros((Nev1,Nev1,Nev1), dtype=complex)
		for k in range(6):
			for ix in range(Nx):
				VVV_temp=VVV_temp+sign[k]*contract("i,ai,bi,ci->abc", phase[_n, ix*Nx*Nx:(ix+1)*Nx*Nx], V[:,ix*Nx*Nx:(ix+1)*Nx*Nx,epsi[k,0]],V[:,ix*Nx*Nx:(ix+1)*Nx*Nx,epsi[k,1]], V[:,ix*Nx*Nx:(ix+1)*Nx*Nx,epsi[k,2]])
		ed=time.time()
		print("VVV done in %.3f s"%(ed-st))
		VVV[_n,t]=cp.asnumpy(VVV_temp)
		del VVV_temp
		cp._default_memory_pool.free_all_blocks()
	del V
	cp._default_memory_pool.free_all_blocks()
for t_source in range(tsource_start,Nt,tsource_interval):
	st0 = time.time()
	st = time.time()
	peram_u=readin_peram(peram_u_dir, conf_id, Nt, Nev, Nev1, t_source)
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
		twopt_1_N[:,t_sink, t_source]=contract("vabc, afkp, bglq, chmr, kl, pq, vfgh -> vmr", VVV_sink, peram_u[t_sink], peram_u[t_sink], peram_u[t_sink,:,:,0:2,0:2], gsink_baryon, gsource_baryon, cp.conj(VVV_source))
		twopt_2_N[:,t_sink, t_source]=contract("vabc, ahkr, bglq, cfmp, kl, pq, vfgh -> vmr", VVV_sink, peram_u[t_sink,:,:,:,0:2], peram_u[t_sink], peram_u[t_sink,:,:,0:2,:], gsink_baryon, gsource_baryon, cp.conj(VVV_source))
		ed=time.time()
		print("N 2pt digram1 & 2 t_source: %d, t_sink: %d contraction done, %.6f s" %(t_source, t_sink, ed-st))
#diagram 2	
	del peram_u
	cp._default_memory_pool.free_all_blocks()
twopt_N=twopt_1_N[:,:,:,0,0]+twopt_1_N[:,:,:,1,1]-twopt_2_N[:,:,:,0,0]-twopt_2_N[:,:,:,1,1]
twopt_N=cp.asnumpy(twopt_N)
twopt_sum=np.zeros((number_of_mom,1,Nt),dtype=complex)
for t_source in range(tsource_start,Nt,tsource_interval):
	for t_sink in range(Nt):
		if(t_sink<t_source):
			twopt_N[:,t_sink,t_source] = -1.0*twopt_N[:,t_sink,t_source]
		twopt_sum[:,0,(t_sink-t_source+Nt)%Nt] = twopt_sum[:,0,(t_sink-t_source+Nt)%Nt] + twopt_N[:,t_sink,t_source]
for _i in range(number_of_mom):
	write_data_ascii(twopt_sum[_i], Nt, Nx, "%s/corr_N_Px%iPy%iPz%i_conf%s.dat"%(corr_dir,mom[_i,0],mom[_i,1],mom[_i,2],conf_id))
