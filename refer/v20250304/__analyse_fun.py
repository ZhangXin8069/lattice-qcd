import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import fileinput
import sympy as sy
import lsqfit
from iog_reader.iog_reader import iog_read
def read_iog(filepath, Nx, Nt, P, ENV, N_start, gap, Ncnfg, tsep_array, link_max, type):
    N_ENV = ENV.shape[0]
    N_P = P.shape[0]
    N_data = filepath.shape[0]
    N_link = 2*link_max+1
    N_tsep = tsep_array.shape[0]
    conf_array = np.asarray(range(N_start, N_start+gap*Ncnfg+1, gap))
    link_array = np.asarray(range(-link_max, link_max+1, 1))
    data_readed_3pt = np.zeros((N_data, N_P, N_ENV, N_tsep, N_link, Ncnfg, Nt, 2), dtype = np.double)
    intrptr = ['ID','OP0','OP1']
    if type == '3pt' or type == 'ratio':
        for i in range((N_data-1)*N_tsep*N_ENV*N_P*Ncnfg):
            tsep_indx = (i)%N_tsep
            conf_indx = (i//(N_tsep))%Ncnfg
            ENV_indx = (i//(N_tsep*Ncnfg))%N_ENV
            P_indx = (i//(N_tsep*Ncnfg*N_ENV))%N_P
            data_indx = (i//(N_tsep*Ncnfg*N_ENV*N_P))%N_data
            data = iog_read(filepath[data_indx]%(Nx, Nt, P[P_indx,0], P[P_indx,1], P[P_indx,2], ENV[ENV_indx], conf_array[conf_indx], tsep_array[tsep_indx], link_max), intrptr)
            data_readed_3pt[data_indx,P_indx,ENV_indx,tsep_indx,:,conf_indx,:,0] = np.append(data['Re'].to_numpy().reshape(N_link,16,Nt)[link_max+1:,8][::-1],data['Re'].to_numpy().reshape(N_link,16,Nt)[:link_max+1,8],axis=0)
            data_readed_3pt[data_indx,P_indx,ENV_indx,tsep_indx,:,conf_indx,:,1] = np.append(data['Im'].to_numpy().reshape(N_link,16,Nt)[link_max+1:,8][::-1],data['Im'].to_numpy().reshape(N_link,16,Nt)[:link_max+1,8],axis=0)
    if type == '2pt' or type == 'ratio':
        for i in range(1*N_ENV*N_P*Ncnfg):
            conf_indx = (i)%Ncnfg
            ENV_indx = (i//(Ncnfg))%N_ENV
            P_indx = (i//(Ncnfg*N_ENV))%N_P
            data = iog_read(filepath[-1]%(Nx, Nt, P[P_indx,0], P[P_indx,1], P[P_indx,2], ENV[ENV_indx], conf_array[conf_indx]), intrptr)
            data_readed_3pt[-1,P_indx,ENV_indx,0,0,conf_indx,:,0] = data['Re'].to_numpy()
            data_readed_3pt[-1,P_indx,ENV_indx,0,0,conf_indx,:,1] = data['Im'].to_numpy()
    return data_readed_3pt
def read_data(filepath, Nx, Nt, P, ENV, N_start, gap, Ncnfg, tsep_array, link_max, type):
    N_ENV = ENV.shape[0]
    N_P = P.shape[0]
    N_data = filepath.shape[0]
    N_link = 2*link_max+1
    N_tsep = tsep_array.shape[0]
    conf_array = np.asarray(range(N_start, N_start+gap*Ncnfg+1, gap))
    link_array = np.asarray(range(-link_max, link_max+1, 1))
    data_readed_3pt = np.zeros((N_data, N_P, N_ENV, N_tsep, N_link, Ncnfg, Nt, 3), dtype = np.double)
    if type == '3pt' or type == 'ratio':
        for i in range((N_data-1)*N_link*N_tsep*N_ENV*N_P*Ncnfg):
            link_indx = (i)%N_link
            tsep_indx = (i//N_link)%N_tsep
            conf_indx = (i//(N_link*N_tsep))%Ncnfg
            ENV_indx = (i//(N_link*N_tsep*Ncnfg))%N_ENV
            P_indx = (i//(N_link*N_tsep*Ncnfg*N_ENV))%N_P
            data_indx = (i//(N_link*N_tsep*Ncnfg*N_ENV*N_P))%N_data
            data = open(filepath[data_indx]%(Nx, Nt, P[P_indx,0], P[P_indx,1], P[P_indx,2], ENV[ENV_indx], conf_array[conf_indx], tsep_array[tsep_indx], link_array[link_indx]), '+r')
            data_A = data.readlines()
            mid_data_B = np.array([item.replace('\n', '') for item in data_A])
            n_data = np.size(mid_data_B)-1
            mid_data_C = np.zeros(3)
            for t_indx in range(n_data):
                mid_data_C = mid_data_B[t_indx+1].split(' ')
                data_readed_3pt[data_indx,P_indx,ENV_indx,tsep_indx,link_indx,conf_indx,t_indx,:] = [float(list) for list in mid_data_C] # data, P, ENV, tsep, link, Ncnfg, Nt, (list,re,im)
    if type == '2pt' or type == 'ratio':
        for i in range(1*N_ENV*N_P*Ncnfg):
            conf_indx = (i)%Ncnfg
            ENV_indx = (i//(Ncnfg))%N_ENV
            P_indx = (i//(Ncnfg*N_ENV))%N_P
            data = open(filepath[-1]%(Nx, Nt, P[P_indx,0], P[P_indx,1], P[P_indx,2], ENV[ENV_indx], conf_array[conf_indx]), '+r')
            data_A = data.readlines()
            mid_data_B = np.array([item.replace('\n', '') for item in data_A])
            n_data = np.size(mid_data_B)-1
            mid_data_C = np.zeros(3)
            for t_indx in range(n_data):
                mid_data_C = mid_data_B[t_indx+1].split(' ')
                data_readed_3pt[-1,P_indx,ENV_indx,0,0,conf_indx,t_indx,:] = [float(list) for list in mid_data_C] # data, P, ENV, tsep, link, Ncnfg, Nt, (list,re,im)
    return data_readed_3pt[...,1:] # data, P, ENV, tsep, link, Ncnfg, Nt, (list,re,im)
def jcknf_sample(data):
    sum_dimension = np.asarray(np.shape(data))
    sum_dimension[-2] = 1
    n = data.shape[-2]
    data_sum = np.sum(data, axis = -2).reshape(sum_dimension)
    jcknf_sample = (data_sum - data)/(n-1)
    return jcknf_sample
def jackknife_ctr_err(data): # data (Ncnfg, Nt)
    n = data.shape[0]
    Nt = data.shape[1]
    jcknf_data = np.zeros((2,Nt), dtype = np.double)
    jcknf_sample_data = jcknf_sample(data)
    jcknf_data_cntrl = np.mean(jcknf_sample_data, axis = 0)
    # jcknf_data_err_2 = np.sqrt(np.sum((data_mean-data_minus_state)**2,axis=0)*(n-1)/n)  # same as jcknf_data_err_2
    jcknf_data_err = np.std(jcknf_sample_data, axis = 0)*np.sqrt(n-1)
    jcknf_data = np.array([[jcknf_data_cntrl],[jcknf_data_err]])
    return jcknf_data
def meff_2pt(data,dtype): # data:1, P, ENV, 1, 1, Ncnfg, Nt, dtype=complex
    Ncnfg = data.shape[-2]
    Nt = data.shape[-1]
    T_hlf = int(Nt/2)
    data_mean_ini = np.mean(data,axis=-2) # data:1, P, ENV, 1, 1, Nt, dtype=complex
    data_sample_ini = jcknf_sample(data) # data:1, P, ENV, 1, 1, Ncnfg, Nt, dtype=complex
    if (dtype=='log'):
        data_mean = np.array(np.log(np.real(data_mean_ini[...,:-1])/np.real(data_mean_ini[...,1:])))
        data_log_sample = np.array(np.log(np.real(data_sample_ini[...,:-1])/np.real(data_sample_ini[...,1:])))
        data_err = np.sqrt(np.sum((data_log_sample - data_mean)**2, axis=-2) * (Ncnfg-1) / Ncnfg)
    else:
        if(dtype=='cosh'):
            data_sample_ini = data_sample_ini.astype(np.float64)
            t_ary = np.array(range(Nt-1))
            ini = 0.2*np.ones_like(t_ary)
            def eff_mass_eqn(c2pt):
                return lambda E0: (c2pt[:-1]/c2pt[1:]) - np.cosh(E0*(Nt/2-t_ary)) / np.cosh(E0*(Nt/2-(t_ary+1)))
            def fndroot(eqnf,ini):
                sol = fsolve(eqnf,ini, xtol=1e-5)
                return sol
            data_cosh_sample = np.array([fndroot(eff_mass_eqn(c2pt),ini) for c2pt in data_sample_ini[:,:]])
            data_mean = np.mean(data_cosh_sample,axis=-2)
            data_err = np.sqrt(Ncnfg-1)*np.std(data_cosh_sample,axis=-2)
    meff_data_2pt = np.array([[data_mean],[data_err]])
    return meff_data_2pt
def PDF_3pt_2pt(data,tsep_array): # data, P, ENV, tsep, link, Ncnfg, Nt, dtype=complex
    # the number of data
    N_data = data.shape[0]
    N_P = data.shape[1]
    N_ENV = data.shape[2]
    N_tsep = data.shape[3]
    N_link = data.shape[4]
    Ncnfg = data.shape[5]
    Nt = data.shape[6]
    PDF_3pt_2pt_mean = np.zeros((N_data-1, N_P, N_ENV, N_tsep, N_link, Nt),dtype=np.complex128)
    PDF_3pt_2pt_err = np.zeros((N_data-1, N_P, N_ENV, N_tsep, N_link, Nt),dtype=np.complex128)
    PDF_3pt_2pt_cov = np.zeros((N_data-1, N_P, N_ENV, N_tsep, N_link, Nt, Nt), dtype=complex)
    
    jcknf_data_sample = jcknf_sample(data) # data, P, ENV, tsep, link, Ncnfg, Nt, dtype=complex
    jcknf_data_mean = np.mean(jcknf_data_sample,axis=-2) # data, P, ENV, tsep, link, Nt, dtype=complex
    
    # the 3pt/2pt part
    # ratio_mean
    PDF_3pt_2pt_sample = np.zeros((N_data-1, N_P, N_ENV, N_tsep, N_link, Ncnfg, Nt))
    for tsep_indx in range(N_tsep):
        PDF_3pt_2pt_sample[:,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1] = jcknf_data_sample[:-1,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1] / jcknf_data_sample[-1,:,:,0,0,:,tsep_array[tsep_indx]].reshape(1,N_P,N_ENV,1,Ncnfg,1)
        PDF_3pt_2pt_mean[:,:,:,tsep_indx,:,:tsep_array[tsep_indx]+1] = jcknf_data_mean[:-1,:,:,tsep_indx,:,:tsep_array[tsep_indx]+1] / jcknf_data_mean[-1,:,:,0,0,tsep_array[tsep_indx]].reshape(1,N_P,N_ENV,1,1)
        #ratio cov
        # cov_mean_dimension = np.asarray(np.shape(PDF_3pt_2pt_sample))
        # cov_mean_dimension[-1] = 1
        # cov_matrix = PDF_3pt_2pt_sample[:,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1] - np.mean(PDF_3pt_2pt_sample[:,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1],axis=-1).reshape(cov_mean_dimension)
        # PDF_3pt_2pt_cov[:,:,:,tsep_indx,:,:tsep_array[tsep_indx]+1,:tsep_array[tsep_indx]+1] = (1/(Nt-1)) * np.transpose(cov_matrix,(0,1,2,3,5,4)) @ cov_matrix
    #ratio_err
    PDF_3pt_2pt_err = np.std(PDF_3pt_2pt_sample, axis = -2)*np.sqrt(Ncnfg-1)
        
    PDF_3pt_2pt_data = np.array([PDF_3pt_2pt_mean,PDF_3pt_2pt_err]) #size = (2,data_1-1,t_sep+1)
    
    return PDF_3pt_2pt_data, PDF_3pt_2pt_cov
def average_for_back(data):
    
    n = data.shape[0]
    t = data.shape[1]
    t_half = int(t/2)
    average_for  = data[:, 0:t_half]
    average_back = data[:,t_half:][:,::-1] 
    data_average_for_back = (average_for + average_back)/2  
    
    return data_average_for_back
def average_for_min_back(data):
    
    n = data.shape[0]
    t = data.shape[1]
    t_half = int(t/2)
    average_for  = data[:, 0:t_half]
    average_back = data[:,t_half:][:,::-1] 
    data_average_for_back = (average_for - average_back)/2  
    
    return data_average_for_back
def min_fun_descent(function, X_1, X_2, X0_1, X0_2):
    alpha = 0.1
    delta = 0.01
    max_cycle_index = 250
    cycle_index = 0
    # u = sy.symbols('u')
    # d_fun = sy.zeros(n)
    # d_fun_value = np.zeros(n)
    d_fun_1 = sy.diff(function, X_1)#[0]
    d_fun_2 = sy.diff(function, X_2)#[0]
    
    # dd_fun_1 = sy.diff(function, X_1, 2)[0]
    # dd_fun_2 = sy.diff(function, X_2, 2)[0]
    
    d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())    # d_fun[i] = (float)(d_fun[i].subs(X, X0).evalf())
    d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    
    dd_fun_value_1 =0# (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
    dd_fun_value_2 =0# (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    while (cycle_index <= max_cycle_index and abs(d_fun_value_1)+abs(d_fun_value_2) >= delta):
        
        cycle_index += 1
        
        X0_1 = X0_1 - d_fun_value_1 * alpha# * (abs(d_fun_value_1)/abs(dd_fun_value_1)) # * np.exp(abs(d_fun_value_1)-abs(dd_fun_value_1)) # * u
        X0_2 = X0_2 - d_fun_value_2 * alpha# * (abs(d_fun_value_2)/abs(dd_fun_value_2)) # * np.exp(abs(d_fun_value_2)-abs(dd_fun_value_2)) # * u
        
        # function_up = function[0].subs({X_1:X0_1_up, X_2:X0_2_up})
        # d_function_up = sy.diff(function_up, u)
        # u_value = sy.solve(d_function_up, u)[0]
        print('cycle_index:%d\n X0_1:%f X0_2:%f\n d_fun_value_1:%f  d_fun_value_2:%f\n dd_fun_value_1:%f  dd_fun_value_2:%f\n'%(cycle_index,X0_1,X0_2,d_fun_value_1,d_fun_value_2, dd_fun_value_1, dd_fun_value_2))
        d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
        d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
        
        # dd_fun_value_1 = (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
        # dd_fun_value_2 = (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
        # X0_1 = (float)(X0_1 + u_value * d_fun_value_1)
        # X0_2 = (float)(X0_2 + u_value * d_fun_value_2)
        
    function_value = (float)(function[0].subs({X_1: X0_1, X_2: X0_2}).evalf())
    parameter = np.array([X0_1, X0_2, function_value])
    # print(parameter)
    return parameter
def min_fun_newton(function, X_1, X_2, X0_1, X0_2):
    max_cycle_index = 200
    alpha = 0.01
    delta = 0.0001
    cycle_index = 0
    
    d_fun_1 = sy.diff(function, X_1)[0]
    d_fun_2 = sy.diff(function, X_2)[0]
    
    dd_fun_1  = sy.diff(function, X_1, 2)[0]
    dd_fun_12 = sy.diff(function, X_1, X_2)[0]
    dd_fun_2  = sy.diff(function, X_2, 2)[0]
    
    d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf()) 
    d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    
    dd_fun_value_1  = (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
    dd_fun_value_2  = (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    dd_fun_value_12 = (float)(dd_fun_12.subs({X_1: X0_1, X_2: X0_2}).evalf())
    
    while (cycle_index <= max_cycle_index and abs(d_fun_value_1)+abs(d_fun_value_2) >= delta):
        cycle_index += 1
        
        G = np.vstack((d_fun_value_1,d_fun_value_2))
        H = np.vstack(([dd_fun_value_1,dd_fun_value_12],[dd_fun_value_12,dd_fun_value_2]))
        
        
        z_value = np.vstack((X0_1, X0_2))-np.linalg.inv(H)@G
        X0_1 = (float)(z_value[0])
        X0_2 = (float)(z_value[1])
        
        d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf()) 
        d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    
        dd_fun_value_1  = (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
        dd_fun_value_2  = (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
        dd_fun_value_12 = (float)(dd_fun_12.subs({X_1: X0_1, X_2: X0_2}).evalf())
        
        print('cycle_index:%d\n X0_1:%f X0_2:%f\n d_fun_value_1:%f  d_fun_value_2:%f\n dd_fun_value_1:%f  dd_fun_value_2:%f\n'%(cycle_index,X0_1,X0_2,d_fun_value_1,d_fun_value_2, dd_fun_value_1, dd_fun_value_2))
        
    function_value = (float)(function[0].subs({X_1: X0_1, X_2: X0_2}).evalf())
    parameter = np.array([X0_1, X0_2, function_value])
    return parameter
def c_square(data, function, X_1, X_2, X0_1, X0_2):
    n = data.shape[0]
    t = data.shape[1]
    c_square_ni = 0
    c_square_ni = np.zeros((n,3))
    
    cov_m_A = np.cov(data.T)
    cov_m_inv = np.linalg.inv(cov_m_A)
    
    c_square_fun = ((data[0,:].reshape(t,1)-function).T)@cov_m_inv@(data[0,:].reshape(t,1)-function)    
    c_square_ni[0] = min_fun_newton(c_square_fun, X_1, X_2, X0_1, X0_2)
    
    for ni in range(1,n):
        c_square_fun = ((data[ni,:].reshape(t,1)-function).T)@cov_m_inv@(data[ni,:].reshape(t,1)-function)    
        c_square_ni[ni] = min_fun_newton(c_square_fun, X_1, X_2, c_square_ni[ni-1, 0], c_square_ni[ni-1, 1])
    
    c_square_mean = np.sum(c_square_ni, axis = 0)/n
    return c_square_mean
# Para = min_fun(fit_function, )
# def dispersion_fit()
# def plt():
    # figure = plt.figure()
    # axes = plt.axes(projection = '3d')
    # x = np.arange(0,2,0.01)
    # y = np.arange(0,1.5,0.01)
    # z=np.zeros((x.size,y.size))
    # for i in range(x.size):
    #     for j in range(y.size):
            
    #         z[i,j] = (float)(c_square_fun[0].subs({X_1: x[i], X_2: y[j]}).evalf())
    # x,y = np.meshgrid(x,y)
    # axes.plot_surface(x,y,z.T,cmap='rainbow')
    # plt.savefig("./c_square.png")