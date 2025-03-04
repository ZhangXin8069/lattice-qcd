#!/public/home/xinghy/anaconda3/bin/python
import numpy as np
import os
from gamma_matrix_cpu import *
from opt_einsum import contract
import time
from input_output_cpu import *
def meson_2pt_2flav(conf_id, tsource, Nt, Nev, Nev1, peram1_dir, peram2_dir, VdV, mom, gamma_meson):
	number_of_mom=mom.shape[0]
	number_of_gamma=gamma_meson.shape[0]
	gamma_sink=np.zeros((number_of_gamma, 4, 4),dtype=complex)
	gamma_source=np.zeros((number_of_gamma, 4 ,4), dtype=complex)
	for _n in range(number_of_gamma):
		gamma_sink[_n]=gamma(5)@gamma_meson[_n]
		gamma_source[_n]=gamma_meson[_n]@gamma(5)
	if(number_of_mom!=VdV.shape[0]):
		print("momentum setting wrong.")
		exit(0)
	peram1=readin_peram(peram1_dir, conf_id, Nt, Nev, Nev1, tsource)
	peram2=readin_peram(peram2_dir, conf_id, Nt, Nev, Nev1, tsource)
	corr=np.zeros((number_of_mom, number_of_mom, number_of_gamma, number_of_gamma, Nt),dtype=complex)
	for _mom_sink in range(number_of_mom):
		print("mom #: %d" %_mom_sink)
		st=time.time()	
		corr[_mom_sink, :, :, :, :] = contract("ade, adins, aejot, xno, yts, wij -> wxya", VdV[_mom_sink], np.conj(peram1), peram2, gamma_sink, gamma_source, np.conj(VdV[:, tsource]))
		ed=time.time()
		print("twopt_ mom: %d, contraction done, %.6f s" %(_mom_sink, ed-st))
	return corr
	
def meson_2pt_1flav(conf_id, tsource, Nt, Nev, Nev1, peram1_dir, VdV, mom, gamma_meson):
	number_of_mom=mom.shape[0]
	number_of_gamma=gamma_meson.shape[0]
	gamma_sink=np.zeros((number_of_gamma, 4, 4),dtype=complex)
	gamma_source=np.zeros((number_of_gamma, 4 ,4), dtype=complex)
	for _n in range(number_of_gamma):
		gamma_sink[_n]=gamma(5)@gamma_meson[_n]
		gamma_source[_n]=gamma_meson[_n]@gamma(5)
	if(number_of_mom!=VdV.shape[0]):
		print("momentum setting wrong.")
		exit(0)
	peram1=readin_peram(peram1_dir, conf_id, Nt, Nev, Nev1, tsource)
	corr=np.zeros((number_of_mom, number_of_mom, number_of_gamma, number_of_gamma, Nt),dtype=complex)
	for _mom_sink in range(number_of_mom):
		print("mom #: %d" %_mom_sink)
		st=time.time()	
		corr[_mom_sink, :, :, :, :] = contract("ade, adins, aejot, xno, yts, wij -> wxya", VdV[_mom_sink], np.conj(peram1), peram1, gamma_sink, gamma_source, np.conj(VdV[:, tsource]))
		ed=time.time()
		print("twopt_ mom: %d, contraction done, %.6f s" %(_mom_sink, ed-st))
	return corr
	
def meson_2pt_self1(conf_id, tsource, Nt, Nev, Nev1, peram1_dir, VdV, mom, gamma_meson):
	number_of_mom=mom.shape[0]
	number_of_gamma=gamma_meson.shape[0]
	gamma_sink=np.zeros((number_of_gamma, 4, 4),dtype=complex)
	gamma_source=np.zeros((number_of_gamma, 4 ,4), dtype=complex)
	for _n in range(number_of_gamma):
		gamma_sink[_n]=gamma(5)@gamma_meson[_n]
		gamma_source[_n]=gamma_meson[_n]@gamma(5)
	if(number_of_mom!=VdV.shape[0]):
		print("momentum setting wrong.")
		exit(0)
	peram1=readin_peram(peram1_dir, conf_id, Nt, Nev, Nev1, tsource)
	# peram2=readin_peram(peram2_dir, conf_id, Nt, Nev, Nev1, tsource)
	corr=np.zeros((number_of_mom, number_of_gamma, Nt),dtype=complex)
	for _mom_sink in range(number_of_mom):
		print("mom #: %d" %_mom_sink)
		st=time.time()
		corr[_mom_sink, :, :] = contract("ade, adins, aejot, xno, xts, ij -> xa", VdV[_mom_sink], np.conj(peram1), peram1, gamma_sink, gamma_source, np.conj(VdV[_mom_sink, tsource]))
		ed=time.time()
		print("twopt_ mom: %d, contraction done, %.6f s" %(_mom_sink, ed-st))
	return corr
def meson_2pt_self2(conf_id, tsource, Nt, Nev, Nev1, peram1_dir, peram2_dir, VdV, mom, gamma_meson):
	number_of_mom=mom.shape[0]
	number_of_gamma=gamma_meson.shape[0]
	gamma_sink=np.zeros((number_of_gamma, 4, 4),dtype=complex)
	gamma_source=np.zeros((number_of_gamma, 4 ,4), dtype=complex)
	for _n in range(number_of_gamma):
		gamma_sink[_n]=gamma(5)@gamma_meson[_n]
		gamma_source[_n]=gamma_meson[_n]@gamma(5)
	if(number_of_mom!=VdV.shape[0]):
		print("momentum setting wrong.")
		exit(0)
	peram1=readin_peram(peram1_dir, conf_id, Nt, Nev, Nev1, tsource)
	peram2=readin_peram(peram2_dir, conf_id, Nt, Nev, Nev1, tsource)
	corr=np.zeros((number_of_mom, number_of_gamma, Nt),dtype=complex)
	for _mom_sink in range(number_of_mom):
		print("mom #: %d" %_mom_sink)
		st=time.time()
		corr[_mom_sink, :, :] = contract("ade, adins, aejot, xno, xts, ij -> xa", VdV[_mom_sink], np.conj(peram1), peram2, gamma_sink, gamma_source, np.conj(VdV[_mom_sink, tsource]))
		ed=time.time()
		print("twopt_ mom: %d, contraction done, %.6f s" %(_mom_sink, ed-st))
	return corr
def baryon_2pt_straight(conf_id, tsource, Nt, Nev, Nev1, peram1_dir, peram2_dir, peram3_dir, VVV, mom, gamma_baryon):
	number_of_mom=mom.shape[0]
	number_of_gamma=gamma_baryon.shape[0]
	if(number_of_mom!=VVV.shape[0]):
		print("momentum setting wrong.")
		exit(0)
	peram1=readin_peram(peram1_dir, conf_id, Nt, Nev, Nev1, tsource)
	peram2=readin_peram(peram2_dir, conf_id, Nt, Nev, Nev1, tsource)
	peram3=readin_peram(peram3_dir, conf_id, Nt, Nev, Nev1, tsource)
	corr=np.zeros((number_of_mom, number_of_gamma, Nt,4,4),dtype=complex)
	corr_pp=np.zeros((number_of_mom, number_of_gamma, Nt),dtype=complex)
	corr_pm=np.zeros((number_of_mom, number_of_gamma, Nt),dtype=complex)
	for _mom_sink in range(number_of_mom):
		print("mom #: %d" %_mom_sink)
		st=time.time()
		CGperamCG=contract("xgh,tbehk,xjk->xtbegj", gamma(10)@gamma_baryon, peram2, gamma(10)@gamma_baryon)
		for tsink in range(Nt):
			corr[_mom_sink, :, tsink] = contract("abc,adgj,xbegj,cfil,def->xil", \
									VVV[_mom_sink, tsink], peram1[tsink], CGperamCG[:,tsink],  peram3[tsink], np.conj(VVV[_mom_sink, tsource]))
		matrix_pplus = 0.5 * (gamma(0) + gamma(4))  # positive parity projection
		matrix_pminus = 0.5 * (gamma(0) - gamma(4))  # negative parity projection
		corr_pp=contract("li,mxtil->mxt",matrix_pplus,corr)
		corr_pm=contract("li,mxtil->mxt",matrix_pminus,corr)
		
		ed=time.time()
		print("twopt_ mom: %d, contraction done, %.6f s" %(_mom_sink, ed-st))
	return corr_pp,corr_pm
def baryon_2pt_multi(conf_id, tsource, Nt, Nev, Nev1, peram1_dir, peram2_dir, VVV, mom, gamma_baryon):
	number_of_mom=mom.shape[0]
	number_of_gamma=gamma_baryon.shape[0]
	Cgamma=np.zeros_like(gamma_baryon)
	for _i in range(number_of_gamma):
		Cgamma[_i]=gamma(10)@gamma_baryon[_i]
	if(number_of_mom!=VVV.shape[0]):
		print("momentum setting wrong.")
		exit(0)
	peram1=readin_peram(peram1_dir, conf_id, Nt, Nev, Nev1, tsource)
	if peram1_dir==peram1_dir:
		peram2=peram1
	else:
		peram2=readin_peram(peram2_dir, conf_id, Nt, Nev, Nev1, tsource)
	corr=np.zeros((number_of_mom, number_of_gamma, Nt,4,4),dtype=complex)
	corr_pp=np.zeros((number_of_mom, number_of_gamma, Nt),dtype=complex)
	corr_pm=np.zeros((number_of_mom, number_of_gamma, Nt),dtype=complex)
	for _mom_sink in range(number_of_mom):
		print("mom #: %d" %_mom_sink)
		st=time.time()
		CGperamCG=contract("xgh,tbehk,xjk->xtbegj", Cgamma, peram2, Cgamma)
		for tsink in range(Nt):
			corr[_mom_sink, :, tsink] = \
				contract("abc,adgj,xbegj,cfil,def->xil", VVV[_mom_sink, tsink], peram1[tsink], CGperamCG[:,tsink], peram1[tsink], np.conj(VVV[_mom_sink, tsource])) - \
        		contract("abc,afgl,xbegj,cdij,def->xil", VVV[_mom_sink, tsink], peram1[tsink], CGperamCG[:,tsink], peram1[tsink], np.conj(VVV[_mom_sink, tsource])) 
		matrix_pplus = 0.5 * (gamma(0) + gamma(4))  # positive parity projection
		matrix_pminus = 0.5 * (gamma(0) - gamma(4))  # negative parity projection
		corr_pp=contract("li,mxtil->mxt",matrix_pplus,corr)
		corr_pm=contract("li,mxtil->mxt",matrix_pminus,corr)
		
		ed=time.time()
		print("twopt_ mom: %d, contraction done, %.6f s" %(_mom_sink, ed-st))
	return corr_pp,corr_pm