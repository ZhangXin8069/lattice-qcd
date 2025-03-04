# We apply Jackknife sampling to $C_i$, calculate the effective and make a fit of 2pt
import pandas as pd
import numpy as np
import gvar as gv
import lsqfit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('pdf')
# ltS_GeV=0.197/0.0962
ltS_GeV=0.197/0.108
# ltS_GeV=1
# Read the useful data
# We then need to extract column 7 and reshape it to a matrix ($C_i$ matrix) such that each row represents a time series for observations under the corresponding configuration.
def read2pt(name):
    dat_ori = pd.read_table(name, delim_whitespace = True, header = None)
    # Nconf = len(set(dat_ori[0])) # number of configurations
    Nconf = 25 # number of configurations
    dat = dat_ori[1].to_numpy().reshape((Nconf, 72)) # rows representing each observation, columns representing time steps
    return dat
# Resampling is taken by extracting the $i$-th row and calculating the average along each column.
def Jsamp (ci):
    ci_resamp = np.zeros_like(ci)
    for nrow in range(ci.shape[0]):
        cols_extracted = np.concatenate((ci[:nrow], ci[nrow + 1:]), axis = 0)
        ci_resamp[nrow, :] = np.average(cols_extracted, axis = 0)
    return ci_resamp
# Calculat the effecgtive mass
def eff_m_per(ci):
    Mass = ltS_GeV * np.log(ci[:, :-1]/ci[:, 1:])
    return Mass
# Calculat the central value and errors without correaltion
def cen_err(a, jackYes):
    N_j = a.shape[0]
    if(jackYes=='jacknife'):
         return np.mean(a,axis=0), np.std(a,axis=0)*np.sqrt(N_j-1)
    else:
         return np.mean(a,axis=0), np.std(a,axis=0)/np.sqrt(N_j-1)
# calcualte the covriant matrix, some change with respect to my orignal code.
def Covmatrix(data,jackYes):
    nf, nt = data.shape
    ave = np.broadcast_to(np.mean(data,0), (nf, nt))
    cov = data - ave
    cov = np.matmul(cov.T, cov) / nf  #利用了矩阵乘法把组态求和了
    if(jackYes== 'jacknife'):
        cov=cov*(nf-1)
    return cov
#### plot the effective mass
def plot_eff(mass_cen, mass_err,mass_cen_Jsamp, mass_err_Jsamp):
# Here, xmin and xmax means the plot range, not fit range
    xmin=1
    xmax=72
    # (fig,ax) = plt.subplots(nrows=1, ncols=1, sharex=True,figsize =(8,4)) # original!
    (fig,ax) = plt.subplots(nrows=1, ncols=1, sharex=True) # changed!
    x_range=np.arange(0,len(mass_cen))
    # ax.errorbar(x_range[xmin:xmax]-0.1,mass_cen[xmin:xmax],mass_err[xmin:xmax],fmt='None',ms=3.,color='brown',ecolor='brown',label=r'$Orignal$')
    # ax.errorbar(x_range[xmin:xmax],mass_cen_Jsamp[xmin:xmax],mass_err_Jsamp[xmin:xmax],fmt='None',ms=3.,color='b',ecolor='b',label=r'$Jacknife$')
    ax.errorbar(x_range[xmin:xmax],mass_cen_Jsamp[xmin:xmax],mass_err_Jsamp[xmin:xmax],fmt='o'+'r', markersize=4, capsize=4,label=r'$Jackknife$')
    plt.minorticks_on()
    # ax.axhline(y=0.31, c="r", ls="--",lw=1, label=r'$m=0.31{\mathrm{GeV}}$')
    plt.legend(loc='upper right')
    plt.xlabel(r'Time / Lattice Unit')
    plt.ylabel(r'Mass / GeV')
    ax.set_ylim(0.,0.5) # original!
    ax.set_xlim(0,20) # original!
    pp = PdfPages("effetive_mass_nucl.pdf")
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.close()
    return 0
#### plot and fit the 2pt
def plot_fit_2pt(c2pt_cen, c2pt_err,c2pt_cov):
    xmin=4
    xmax=15
    plotxmin=1 #(Added by H.X.)
    plotxmax=72 #(Added by H.X.)
    # prior = {'c0':gv.gvar(0.02, 0.5),'m0':gv.gvar(0.3,1.),'c1':gv.gvar(1., 100.),'deltam':gv.gvar(0.5,10.)}
    prior = {'c0':gv.gvar(372151.79738314, 100),'m0':gv.gvar(0.3, 1.)}
    xfit = np.arange(xmin,xmax)
    plotx= np.arange(plotxmin,plotxmax)
    ## different errors in the fit
#    yfit= gv.gvar(c2pt_cen[xmin:xmax],c2pt_err[xmin:xmax])
    # yfit= gv.gvar(c2pt_cen[xmin:xmax],np.sqrt(c2pt_cov[xmin:xmax,xmin:xmax].diagonal()))
    yfit= gv.gvar(c2pt_cen[xmin:xmax],c2pt_cov[xmin:xmax,xmin:xmax])
    def fcn2pt(x, p):                        # fit function of x and parameters p
        # ans = p['c0']*(np.exp(-x* p['m0']/ltS_GeV))*(1+p['c1']*(np.exp(-x* p['deltam']/ltS_GeV)))
        ans = p['c0']*(np.exp(-x* p['m0']/ltS_GeV))
        return ans
    fit = lsqfit.nonlinear_fit(data=(xfit, yfit),svdcut=1e-3,prior=prior, fcn=fcn2pt)
    print(fit.format(maxline=True))
    fitted_result=fcn2pt(xfit,fit.p);
    fitted_cen = np.array([ fitted_result[i].mean  for i in range(0,len(fitted_result))])
    fitted_err = np.array([ fitted_result[i].sdev  for i in range(0,len(fitted_result))])
### We plot the 2pt and fitted results
    (fig,ax) = plt.subplots(nrows=1, ncols=1, sharex=True)
    # ax.errorbar(xfit,c2pt_cen[xmin:xmax],c2pt_err[xmin:xmax],fmt='None',ms=3.,color='brown',ecolor='brown',label=r'$Data$')
    ax.errorbar(plotx,c2pt_cen[plotxmin:plotxmax],c2pt_err[plotxmin:plotxmax],fmt='o'+'r', markersize=4, capsize=4,label=r'$Jackknife$') #(Changed by H.X.)
    ax.errorbar(xfit,fitted_cen,fitted_err,fmt='b',ms=3.,color='b',ecolor='b',label=r'$Fitting$')
    ax.fill_between(xfit,fitted_cen+fitted_err,fitted_cen+(-1)*fitted_err,facecolor='b',alpha=0.4)
    plt.minorticks_on()
    plt.legend(loc='upper right')
    plt.xlabel(r'Time / Lattice Unit')
    plt.ylabel(r'$C_2$')
    plt.yscale("log")
    pp = PdfPages("c2.pdf")
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.close()
    return 0
def main():
    c2pt = read2pt("pion_gamma15_p0_t0_1.dat")
    c2pt_Jsamp = Jsamp(c2pt)
    eff_m_Jsamp = eff_m_per(c2pt_Jsamp)
    eff_m = eff_m_per(c2pt)
    ### effective mass
    mass_cen, mass_err = cen_err(eff_m,'No');
    mass_cen_Jsamp, mass_err_Jsamp = cen_err(eff_m_Jsamp,'jacknife');
    print(mass_cen)
    plot_eff(mass_cen, mass_err,mass_cen_Jsamp, mass_err_Jsamp);
    ### calculate  2pt and fit 2pt
    c2pt_cen, c2pt_err =cen_err(c2pt_Jsamp,'jacknife');
    print(c2pt_Jsamp.shape);
    c2pt_cov = Covmatrix(c2pt_Jsamp,'jacknife');
    plot_fit_2pt(c2pt_cen, c2pt_err,c2pt_cov);
    print(c2pt_cen)
    return 0
if __name__ == "__main__":
    main()
    print("end")
