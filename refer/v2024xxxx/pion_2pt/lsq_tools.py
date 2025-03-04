import numpy as np
import time
import lsqfit
import gvar as gv
def read_dat_file(file_prefix,file_subfix="",one=True):
    import glob
    if one == True:
        file_list = glob.glob(f"{file_prefix}{file_subfix}")
    else:
        file_list = glob.glob(f"{file_prefix}*{file_subfix}")
    sorted_file_list = sorted(file_list, key=lambda x: int(''.join(filter(str.isdigit, x))))
    #print(sorted_file_list)
    all_data=[]
    for file in sorted_file_list:
        file_data = open(F"{file}","rb")
        real = []
        for line in file_data:
            tmp = line.split()
            real.append(float(tmp[1]))
        Nt = int(real[0]) 
        del real[0]
        all_data.append(real)
        file_data.close()
    all_data=np.array(all_data)
    #print("N=", all_data.shape[0])
    if one==True:
        N = all_data[0].shape[0]//Nt
        contract=np.zeros((N,Nt))
        for i in range(N):
            contract[i]=all_data[0,int(Nt*i):int(Nt*(i+1))]
        if N!=1:
            return contract
        else:
            return all_data[0]
    else:
        return all_data
def conf_list(filepath):
    file_data = open(filepath,"rb")
    conf_list = []
    for line in file_data:
        tmp = line.split()
        conf_list.append(int(tmp[0]))
    file_data.close()
    return np.array(conf_list)
def jackknife(corr):
    jacksample=corr.shape[0]
    jack_corr = (np.sum(corr,0)-corr)/(jacksample-1)
    return jack_corr
def bootstrap(source, nbsamples):
	source_shape = source.shape
	nsample = source_shape[0]
	boots_shape = list(source_shape)
	boots_shape[0] = nbsamples
	boots_shape = tuple(boots_shape)
	
	np.random.seed(1227)
	boot = np.zeros(boots_shape, dtype=float)
	boot[0] = np.mean(source, axis=0)
	for _i in range(1, nbsamples):
		_rnd = np.random.randint(0, nsample, size=nsample)
		_sum = np.zeros_like(boot[0])
		for _r in range(0, nsample):
			_sum = _sum + source[_rnd[_r]]
		boot[_i] = _sum / nsample
	return boot
def cov_M(contract):
    # contract.shape[0] is nsamples 
    cov=contract - np.mean(contract,0)
    cov=np.matmul(cov.T, cov) / contract.shape[0]
    return cov   
def lsq_fit(contract,start,beta,Eexp,method,shape,efunc):
    N=10  #contract.shape[0]
    Nt=contract.shape[1]
    M_ij=cov_M(contract)
    pr_arr_corr = np.zeros([N, 3])
    if shape=="twostate":
        pr_arr_corr = np.zeros([N, 4])
    T = np.arange(Nt)
    if method == "p0":
        print("fit with p0 method")
        print("")
        p0 = {}
        p0["a0"] = 0.3
        p0["E0"] = Eexp
        if shape=="twostate":
            p0["dE"] = 0.5
    else:
        print("fit with prior method")
        print("")
        prior = {}
        prior["a0"] = gv.gvar(0.3, 1e10)
        prior["E0"] = gv.gvar(Eexp, 2)
        if shape=="twostate":
            prior["dE"] = gv.gvar(0.5, 1e10)
        
    for fid in range(N):
        gv_corr = gv.gvar(contract[fid], M_ij)
        if method == "p0":
            fit = lsqfit.nonlinear_fit(
                data=(T[start :  beta + 1], gv_corr[start :  beta + 1]),
                fcn=efunc,
                svdcut=1e-3,
                p0=p0,
            )
        else:
            fit = lsqfit.nonlinear_fit(
                data=(T[start : beta + 1], gv_corr[start :  beta + 1]),
                fcn=efunc,
                svdcut=1e-3,
                prior=prior,
            )
            
        pr_corr = fit.p
        pr_arr_corr[fid][0] = pr_corr["E0"].mean
        pr_arr_corr[fid][1] = pr_corr["a0"].mean
        pr_arr_corr[fid][2] = fit.chi2 / (fit.dof) #-2
        if shape=="twostate":
            pr_arr_corr[fid][3] = pr_corr["dE"].mean
        
        
        #print("id=",fid, "E=",fit.p["E0"].mean, "chis = ", fit.chi2/ (fit.dof+1))
            
    return pr_arr_corr