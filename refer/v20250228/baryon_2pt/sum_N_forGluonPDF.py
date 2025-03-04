#!/public/home/xinghy/anaconda3-2023.03/envs/cupy114/bin/python
import numpy as np
from os.path import exists
conf_start=11000
conf_end=32500
conf_inter=50
Nt=64
mom_baryon=np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4]])
number_of_mom=mom_baryon.shape[0]
corr_dir="/public/home/liuming/LapH/data/gluonPDF/N_2pt/beta6.72_mu-0.1850_ms-0.1700_L48x144/Nev200/Nosum"
output_dir="/public/home/liuming/LapH/data/gluonPDF/N_2pt/beta6.72_mu-0.1850_ms-0.1700_L48x144/Nev200"
for conf in range(conf_start, conf_end+conf_inter, conf_inter):
	for n in range(number_of_mom):
		twopt_N=np.zeros((Nt,Nt,2,2),dtype=complex)
		twopt_N_sum=np.zeros((Nt,Nt), dtype=complex)
		if exists("%s/N_2pt_pp_Px%iPy%iPz%i.conf%i.npz" %(corr_dir, mom_baryon[n,0],mom_baryon[n,1],mom_baryon[n,2], conf)):
			print(conf)
			twopt_N=np.load("%s/N_2pt_pp_Px%iPy%iPz%i.conf%i.npz" %(corr_dir, mom_baryon[n,0],mom_baryon[n,1],mom_baryon[n,2], conf))['corr']
			twopt_N_sum = twopt_N[:,:,0,0] + twopt_N[:,:,1,1]
			print(twopt_N) 
			np.savez("%s/N_2pt_pp_Px%dPy%dPz%d.conf%d.npz" %(output_dir,mom_baryon[n,0],mom_baryon[n,1],mom_baryon[n,2],conf), twopt_N_sum)
