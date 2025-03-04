import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import fileinput
import sympy as sy
import lsqfit
def read_data(N_start, gap, Ncnfg, ENV, P, Nx, Nt, filepath, t_sep):
    N_ENV = ENV.shape[0]
    N_P = P.shape[0]
    N_data = filepath.shape[0]
    data_readed = np.zeros((N_data, N_ENV, N_P, Ncnfg, Nt, 3), dtype = np.double)
    # the type of data
    for i in range(N_data):
        # the number of ENV
        for j in range(N_ENV):
            # the number of P
            for k in range(N_P):
                # conf id
                for l in range(N_start,N_start+gap*Ncnfg,gap):
                    if i < N_data-1:
                        data = open(filepath[i]%(P[k,0], P[k,1], P[k,2], ENV[j], Nx, Nt, l, t_sep), '+r')
                    else:
                        data = open(filepath[i]%(P[k,0], P[k,1], P[k,2], ENV[j], Nx, Nt, l), '+r')
                    data_A = data.readlines()
                    mid_data_B = np.array([item.replace('\n', '') for item in data_A])
                    n_data = np.size(mid_data_B)-1
                    mid_data_C = np.zeros(3)
                    for o in range(n_data):
                        mid_data_C = mid_data_B[o+1].split(' ')
                        data_readed[i,j,k,int((l-N_start)/gap),o,:] = [float(list) for list in mid_data_C] # data_0, data_1, Ncnfg, Nt,(list,re,im)
    return data_readed[:,:,:,:,:,1:] # N_data, N_ENV, N_P, Ncnfg, Nt, (re,im)
def jcknf_sample(data): # data (Ncnfg, Nt)
    n = data.shape[0]
    data_sum = np.sum(data, axis = 0)
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
def meff_2pt(data,dtype): # data(Ncnfg,Nt)  dtype = (log, cosh)
    Ncnfg = data.shape[0]
    Nt = data.shape[1]
    T_hlf = int(Nt/2)
    data_mean_ini = np.mean(data,axis=0)
    data_sample_ini = jcknf_sample(data)
    if (dtype=='log'):
        data_mean = np.array(np.log(np.real(data_mean_ini[:-1])/np.real(data_mean_ini[1:])))
        data_log_sample = np.array(np.log(np.real(data_sample_ini[:,:-1])/np.real(data_sample_ini[:,1:])))
        data_err = np.sqrt(np.sum((data_mean - data_log_sample)**2, axis=0) * (Ncnfg-1) / Ncnfg)
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
            data_mean = np.mean(data_cosh_sample,axis=0)
            data_err = np.sqrt(Ncnfg-1)*np.std(data_cosh_sample,axis=0)
    meff_data_2pt = np.array([[data_mean],[data_err]])
    return meff_data_2pt
def PDF_3pt_2pt(data,t_sep): # data (data_1, Ncnfg, Nt, complex)
    # the type of data
    data_1 = data.shape[0]
    Ncnfg = data.shape[1]
    Nt = data.shape[2]
    
    jcknf_A_sample = np.zeros((data_1, Ncnfg, t_sep+1),dtype=complex)
    jcknf_A_mean = np.zeros((data_1,t_sep+1),dtype=complex)
    for i in range(data_1): # except the last element to do jcknf
        jcknf_A_sample[i,:,:] = jcknf_sample(data[i,:,:t_sep+1])
        jcknf_A_mean[i,:] = np.mean(jcknf_A_sample[i,:,:], axis=0)
    
    # the 3pt/2pt part
    # ratio_mean
    PDF_3pt_2pt_sample = np.zeros((data_1-1,Ncnfg,t_sep+1),dtype=complex)
    PDF_3pt_2pt_mean = jcknf_A_mean[:-1,:] / jcknf_A_mean[-1,t_sep]
    for j in range(Ncnfg):
        PDF_3pt_2pt_sample[:,j,:] = jcknf_A_sample[:-1,j,:] / jcknf_A_sample[-1,j,t_sep]
        
    #ratio_err_cov
    PDF_3pt_2pt_err = np.zeros((data_1-1,t_sep+1),dtype=complex)
    PDF_3pt_2pt_cov = np.zeros((data_1-1,t_sep+1,t_sep+1), dtype=complex)
    for k in range(data_1-1):
        PDF_3pt_2pt_err[k] = np.std(PDF_3pt_2pt_sample[k,:,:], axis = 0)*np.sqrt(Ncnfg-1)
        PDF_3pt_2pt_cov[k] = np.cov(PDF_3pt_2pt_sample[k,:,:].T)*(Ncnfg-1)
        
    PDF_3pt_2pt_data = np.array([[PDF_3pt_2pt_mean],[PDF_3pt_2pt_err]]) #size = (2,data_1-1,t_sep+1)
    
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