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
nsample=0
for conf in range(conf_start, conf_end+conf_inter, conf_inter):
	if exists("%s/baryon.conf%s.npz" %(input_dir, conf)):
		nsample=nsample+1
print(nsample)
mom = np.load('%s/baryon.conf%s.npz' %(input_dir, conf_start))['mom']
number_of_mom = mom.shape[0]
corr_N=np.zeros((number_of_mom,nsample,Nt),dtype=complex)
corr_Lamc=np.zeros((number_of_mom,nsample,Nt),dtype=complex)
corr_Sigc=np.zeros((number_of_mom,nsample,Nt),dtype=complex)
count=0
for conf in range(conf_start, conf_end+conf_inter, conf_inter):
	if exists("%s/baryon.conf%s.npz" %(input_dir, conf)):
		print(conf)
		data=np.load("%s/baryon.conf%s.npz" %(input_dir, conf))
		temp_N = data['corr_N']
		temp_Lamc = data['corr_Lamc']
		temp_Sigc = data['corr_Sigc']
		for t_sink in range(Nt):
			for t_source in range(Nt):
				if(t_sink < t_source):
					temp_N[:,t_sink,t_source] = -1.0*temp_N[:,t_sink,t_source]
					temp_Lamc[:,t_sink,t_source] = -1.0*temp_Lamc[:,t_sink,t_source]
					temp_Sigc[:,t_sink,t_source] = -1.0*temp_Sigc[:,t_sink,t_source]
		for t_sink in range(Nt):
			for t_source in range(Nt):
				corr_N[:, count, (t_sink-t_source)%Nt] = corr_N[:, count, (t_sink-t_source)%Nt] + temp_N[:,t_sink, t_source]
				corr_Lamc[:, count, (t_sink-t_source)%Nt] = corr_Lamc[:, count, (t_sink-t_source)%Nt] + temp_Lamc[:,t_sink, t_source]
				corr_Sigc[:, count, (t_sink-t_source)%Nt] = corr_Sigc[:, count, (t_sink-t_source)%Nt] + temp_Sigc[:,t_sink, t_source]
 
		count=count+1
corr_N=corr_N/number_of_tsource
corr_Lamc=corr_Lamc/number_of_tsource
corr_Sigc=corr_Sigc/number_of_tsource
for _i in range(number_of_mom):
	write_data_ascii(corr_N[_i], Nt, Nx, "%s/corr_N_Px%iPy%iPz%i.dat"%(output_dir,mom[_i,0],mom[_i,1],mom[_i,2]))
	write_data_ascii(corr_Lamc[_i], Nt, Nx, "%s/corr_Lamc_Px%iPy%iPz%i.dat"%(output_dir,mom[_i,0],mom[_i,1],mom[_i,2]))
	write_data_ascii(corr_Sigc[_i,], Nt, Nx, "%s/corr_Sigc_Px%iPy%iPz%i.dat"%(output_dir,mom[_i,0],mom[_i,1],mom[_i,2]))
#sum over different mom direction
mom2=np.zeros(number_of_mom, dtype=int)
for _i in range(number_of_mom):
        mom2[_i] = mom[_i,0]**2 + mom[_i,1]**2 + mom[_i,2]**2
p2list,ip2=np.unique(mom2,return_counts=True)
number_of_p2=p2list.shape[0]
j=0
for _n in range(number_of_p2):
        corr_N_momave=np.zeros((nsample,Nt),dtype=complex)
        corr_Lamc_momave=np.zeros((nsample,Nt),dtype=complex)
        corr_Sigc_momave=np.zeros((nsample,Nt),dtype=complex)
        for _i in range(ip2[_n]):
                corr_N_momave = corr_N_momave + corr_N[_i+j]
                corr_Lamc_momave = corr_Lamc_momave + corr_Lamc[_i+j]
                corr_Sigc_momave = corr_Sigc_momave + corr_Sigc[_i+j]
        j=j+ip2[_n]
        corr_N_momave=corr_N_momave/ip2[_n]
        corr_Lamc_momave=corr_Lamc_momave/ip2[_n]
        corr_Sigc_momave=corr_Sigc_momave/ip2[_n]
        write_data_ascii(corr_N_momave, Nt, Nx, "%s/ave/corr_N_P%i.dat"%(output_dir,p2list[_n]))
        write_data_ascii(corr_Lamc_momave, Nt, Nx, "%s/ave/corr_Lamc_P%i.dat"%(output_dir,p2list[_n]))
        write_data_ascii(corr_Sigc_momave, Nt, Nx, "%s/ave/corr_Sigc_P%i.dat"%(output_dir,p2list[_n]))
