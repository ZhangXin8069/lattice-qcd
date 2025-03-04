#!/public/home/xinghy/anaconda3/bin/python
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
	if(tmp[0]=='peram_u_dir'):
		peram_u_dir=tmp[1]
	if(tmp[0]=='peram_c_dir'):
		peram_c_dir=tmp[1]
	if(tmp[0]=='VVV_dir'):
		VVV_dir=tmp[1]
	if(tmp[0]=='corr_dir'):
		corr_dir=tmp[1]
	if(tmp[0]=='number_of_processes'):
		number_of_processes=int(tmp[1])
#mom to be calculated for the baryon twopt
mom=np.array([[0,0,0]])
number_of_mom=mom.shape[0]
gamma_meson=np.array([gamma(5)])
number_of_gamma=gamma_meson.shape[0]
VVV=np.zeros((number_of_mom,Nt,Nev1,Nev1,Nev1),dtype=complex)
for _n in range(number_of_mom):
	print(_n)
	st=time.time()
	VVV[_n,:,:,:] = readin_VVV(VVV_dir, Nev, Nev1, Nt, conf_id, mom[_n,0],mom[_n,1], mom[_n,2])
	ed=time.time()
	print("read VVV %d done, time used: %.6f s" %(_n, ed-st))
twopt=np.zeros((number_of_mom, number_of_mom, number_of_gamma, number_of_gamma, Nt,Nt),dtype=complex)
stt=time.time()
all_args=[[conf_id, tsource, Nt, Nev, Nev1, peram_u_dir, peram_u_dir, VVV, mom, gamma_meson] for tsource in range(tsource_start,Nt,tsource_interval)]
with mp.Pool(processes = number_of_processes) as p:
	results1 = p.starmap(baryon_2pt_multi, all_args)
print(np.array(results1).shape)
twopt1=np.array(results1).transpose(1,2,3,0,4)
edt=time.time()
print("twopt contraction done, %.6f s" %(edt-stt))
np.save("%s/corr1_conf%s.npy" %(corr_dir, conf_id), twopt1[0])
np.save("%s/corr2_conf%s.npy" %(corr_dir, conf_id), twopt1[1])
