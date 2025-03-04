#!/public/home/xinghy/anaconda3-2023.03/envs/cupy114/bin/python
import numpy as np
import os
from input_output_cpu import *
import fileinput
import time
from os.path import exists
infile=fileinput.input()
for line in infile:
	tmp=line.split()
	if(tmp[0]=='Nt'):
		Nt=int(tmp[1])
	if(tmp[0]=='Nx'):
		Nx=int(tmp[1])
	if(tmp[0]=='conf_start'):
		conf_start=int(tmp[1])
	if(tmp[0]=='conf_end'):
		conf_end=int(tmp[1])
	if(tmp[0]=='conf_inter'):
		conf_inter=int(tmp[1])
	if(tmp[0]=='tsource_interval'):
		tsource_interval=int(tmp[1])
	if(tmp[0]=='tsource_start'):
		tsource_start=int(tmp[1])	
	if(tmp[0]=='input_dir'):
		input_dir=tmp[1]
	if(tmp[0]=='output_dir'):
		output_dir=tmp[1]
number_of_tsource=Nt/tsource_interval
mom=np.array([[0,0,0],[0,0,1],[0,1,1],[1,1,1],[0,0,2]])
number_of_mom=mom.shape[0]
nsample=0
for conf in range(conf_start, conf_end+conf_inter, conf_inter):
	if exists("%s/corr_D_conf%s.npz" %(input_dir, conf)):
		nsample=nsample+1
print(nsample)
#corr_D=np.zeros((nsample,number_of_mom,4,Nt,Nt),dtype=complex)
#corr_Etac=np.zeros((nsample,number_of_mom,4,Nt,Nt),dtype=complex)
corr_D_sum=np.zeros((nsample,number_of_mom,4,Nt),dtype=complex)
corr_Etac_sum=np.zeros((nsample,number_of_mom,4,Nt),dtype=complex)
count=0
for conf in range(conf_start, conf_end+conf_inter, conf_inter):
	if exists("%s/corr_D_conf%s.npz" %(input_dir, conf)):
		print(conf)
		temp_D = np.load("%s/corr_D_conf%s.npz" %(input_dir, conf))['corr']	
		print(temp_D.shape)
		temp_Etac = np.load("%s/corr_Etac_conf%s.npz" %(input_dir, conf))['corr']	
		for _i in range(number_of_mom):
			for _j in range(4): # four gamma matrices
				for tsink in range(Nt):
					for tsource in range(tsource_start,Nt,tsource_interval):
						corr_D_sum[count, _i,_j,(tsink-tsource+Nt)%Nt] = corr_D_sum[count, _i,_j,(tsink-tsource+Nt)%Nt] + temp_D[_i,_j, tsink, tsource]
						corr_Etac_sum[count, _i,_j,(tsink-tsource+Nt)%Nt] = corr_Etac_sum[count, _i,_j,(tsink-tsource+Nt)%Nt] + temp_Etac[_i,_j,tsink, tsource]
		count=count+1
number_of_source=Nt/tsource_interval
corr_D_sum=corr_D_sum/number_of_source
corr_Etac_sum=corr_Etac_sum/number_of_source
for _i in range(number_of_mom):
	write_data_ascii(corr_D_sum[:,_i,0], Nt, Nx, "%s/corr_D_Px%iPy%iPz%i.dat"%(output_dir,mom[_i,0],mom[_i,1],mom[_i,2]))
	write_data_ascii(corr_Etac_sum[:,_i,0], Nt, Nx, "%s/corr_Etac_Px%iPy%iPz%i.dat"%(output_dir,mom[_i,0],mom[_i,1],mom[_i,2]))
	for _j in {1,2,3}:
		write_data_ascii(corr_D_sum[:,_i,_j], Nt, Nx, "%s/corr_Dstar_gamma%i_Px%iPy%iPz%i.dat"%(output_dir,_j,mom[_i,0],mom[_i,1],mom[_i,2]))
		write_data_ascii(corr_Etac_sum[:,_i,_j], Nt, Nx, "%s/corr_JPsi_gamma%i_Px%iPy%iPz%i.dat"%(output_dir,_j,mom[_i,0],mom[_i,1],mom[_i,2]))
