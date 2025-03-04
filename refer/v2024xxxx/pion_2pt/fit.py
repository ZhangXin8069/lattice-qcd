import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import time
import lsqfit
import gvar as gv
from prettytable import PrettyTable
import sys
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
# data loading读取数据
data_all = np.load(r"/public/home/zhangxin/lattice-qcd/laph/diJpsi/single/corr/corr_hc_mom0.npy")
Nsamples=data_all.shape[0]
print("Nsamples=",Nsamples)
print("Nt=",data_all.shape[1])
Nt0=96   # length of correlation function
nbsamples = 2000   # bootstrap number 571
# particle information
state="F32P30"
object= r"$\chi_{c0}$" #r"$J/\psi$ $p=0$"  #"Pion" #     r"$\eta_c$"
# fitting method select and input判断用那种方法
jack=False#True#  # choose jackknife or bootstrap
if jack==True:
    boot_corr = jackknife(data_all)
else:
    boot_corr = bootstrap(data_all , nbsamples)
#拟合方法
shape = "onestate"  # choose the number of exponential of fitting function(拟合时只用一个cosh)
exp_E=1.3         # initial of energy levels(拟合的期望值,即提前预估拟合的结果大概是多少)
#平台所在区间
start_t=20          # left side of fitting section 
end_t=40            # right side of fitting section
deltat=  1           # how long can left side move(右端固定,不断移动平台左端,拟合10次)(因为右端误差大,对结果贡献小)
select_t=20        # final start timeslice
# plot input
ymin=1.3
ymax=1.5
########################################   begin fitting work
print(F"fit job starts!!!!")
st0 = time.time()
contract = boot_corr[:,:Nt0//2]/boot_corr[0,0]  # make the first data equal to 1 to make the fitting easier
N = contract.shape[0]  # your final number of samples
print("N=", N)
# meson effective mass
meson_mass =np.arccosh((np.roll(contract, -1, 1) + np.roll(contract, 1, 1)) / (2 * contract)) 
# baryon effective mass
# meson_mass =np.log(np.roll(contract, 1, 1) / (contract)) 
erros = np.std(meson_mass, 0)
if jack==True:
    erros=erros*np.sqrt(Nsamples-1) 
# covariance matrix
M_ij = contract - np.mean(contract, 0)
M_ij = np.matmul(M_ij.T, M_ij) 
if jack==True:
    M_ij = M_ij*(Nsamples-1)/Nsamples
    print("jack!!")
else:
    M_ij = M_ij/nbsamples
    print("boot!!")
# fitting function
def efunc(x, p):
    if shape == "onestate":
        return p["a0"]/np.exp(p["E0"]*(Nt0//2)) * np.cosh(-p["E0"] * (x-Nt0//2))
    elif shape =="twostate":
        return p["a0"] * np.exp(-p["E0"] * x) + (p["a1"]) * np.exp(-(p["E0"]+p["dE"]**2) * x)
pr_arr_corr = np.zeros([N, 2])
T = np.arange(Nt0//2)
chi_square = np.zeros([1, deltat])
mass_mean = np.zeros([1, deltat])
error_mean = np.zeros([1, deltat])
save_E = np.zeros([deltat,N])
for t1 in range(start_t, start_t + deltat):
    st1=time.time()
    print("left:", t1, "right:",  end_t)
    # p0 is the initial parameters
    p0 = {}
    p0["a0"] = 0.5
    p0["E0"] = exp_E
    if shape=="twostate":
        p0["dE"] = 0.5
        p0["a1"] = 0.5
    gv_corr = gv.gvar(np.mean(contract,0), M_ij) # average data with covariance matrix
    # fitting of anverage
    fit_mean = lsqfit.nonlinear_fit(data=(T[t1 : end_t + 1], gv_corr[t1 : end_t + 1]), fcn=efunc,p0=p0)
    chi_mean=fit_mean.chi2/fit_mean.dof
    # fiting of all samples
    for id in range(N):
        gv_corr = gv.gvar(contract[id], M_ij) # the id of data to be fitted
        fit = lsqfit.nonlinear_fit(data=(T[t1 : end_t + 1], gv_corr[t1 : end_t + 1]), fcn=efunc,p0=p0)
        pr_corr = fit.p  # parameters with error
        pr_arr_corr[id][0] = pr_corr["E0"].mean  # parameter center value to be saved
        pr_arr_corr[id][1] = pr_corr["a0"].mean 
        save_E[t1 - start_t,id] = pr_corr["E0"].mean
        
    mass_mean[0][t1 - start_t] = np.mean(pr_arr_corr, 0)[0]
    chi_square[0][t1 - start_t] = chi_mean
    print(fit.p, "chi2/dof = ", chi_mean) 
    error_mean[0][t1 - start_t] = np.std(pr_arr_corr, 0)[0]
    if jack==True:
        error_mean[0][t1 - start_t]=error_mean[0][t1 - start_t]*np.sqrt(Nsamples-1)
    
    ed1=time.time()
    print(f"{t1} to {end_t} fit done, time used : %.3f s"%(ed1-st1))
    
###########################################   picture
    end = end_t
    E_mean = mass_mean[0][t1 - start_t]  
    E_error = error_mean[0][t1 - start_t] 
    fig = plt.figure()    
    plt.errorbar(x=T,y=np.mean(meson_mass,0),yerr=erros,ecolor="cornflowerblue",linestyle="none",\
        mec="cornflowerblue",marker="o",alpha=0.7,markerfacecolor="none",capsize=2,capthick=1,label="Meff")
    plt.legend(loc='upper right')
    left, bottom, width, height = (t1, (E_mean - E_error), end - t1, 2 * E_error)
    rect = mpatches.Rectangle((left, bottom),width,height,alpha=0.4,facecolor="red",label="fitting")
    plt.gca().add_patch(rect)
    plt.text(10, (ymax-ymin)*0.8+ymin, f'$\chi^2/d.o.f.$ = {chi_mean}', fontsize=9)
    plt.text(10, (ymax-ymin)*0.9+ymin, f'fit $E$ = {E_mean}', fontsize=9)
    plt.text(10, (ymax-ymin)*0.85+ymin, f'error = {E_error}', fontsize=9)
    plt.legend(loc='upper right') 
    plt.title("%s %s" % (state, object))
    plt.xlabel("t")
    plt.ylabel("%s aE" % (object))
    plt.xlim(5, Nt0//2)
    plt.ylim(ymin, ymax)
    plt.savefig(f"/public/home/zhangxin/lattice-qcd/test.jpg",dpi=400)
### --------------pretty table
print("mass=", mass_mean)
print("error=", error_mean)
twostatefit = PrettyTable(
    ["start_t timeslice(end=" + str(end_t) + ")", "chi2/dof", "mass mean", "fitting error"]
)
for i in range(deltat):
    twostatefit.add_row(
        [
            np.arange(deltat)[i] + start_t,
            chi_square[0][i],
            mass_mean[0][i],
            error_mean[0][i],
        ]
    )
print(twostatefit)
fig, (ax1, ax2) = plt.subplots(2, 1)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax1.errorbar(x=np.arange(deltat) + start_t, y=mass_mean[0], yerr=error_mean[0], ecolor="red", linestyle="none",
    mec="red", marker="o", alpha=0.7,markerfacecolor="none",capsize=2, capthick=1,)
plotx=np.arange(deltat) + start_t
selectid=select_t-start_t
mean=mass_mean[0,selectid]
err=error_mean[0,selectid]
ax1.fill_between([plotx[0]-0.5,plotx[-1]+0.5],[mean+err,mean+err],\
    [mean-err,mean-err],alpha=0.2,color="red")
ax2.scatter(x=plotx,y=chi_square[0],color="cornflowerblue")
ax2.scatter(x=selectid+start_t,y=chi_square[0,selectid],color="brown")
ax2.text(selectid+start_t,chi_square[0,selectid]+0.25,f"{chi_square[0,selectid]:0.2f}",ha='center')
ax1.text(selectid+start_t,mass_mean[0,selectid]+error_mean[0,selectid]*1.2,\
            f"{mass_mean[0,selectid]:0.5f}"+"("+"%0.0f"%(error_mean[0,selectid]*100000)+")",ha='center')
ax2.plot(plotx,np.ones_like(plotx),color="grey",alpha=0.7,linestyle="--")
auto_ymin=np.mean(mass_mean[0,selectid])-error_mean[0,selectid]*5
auto_ymax=np.mean(mass_mean[0,selectid])+error_mean[0,selectid]*5
ax2.set_xlabel("start t")
ax1.set_ylabel("$fit$ $M_{eff}$")
ax2.set_ylabel("$\chi^2/d.o.f.$")
ax1.set_ylim(auto_ymin, auto_ymax)
ax2.set_ylim(0, 3)   
Xset=[]
for i in plotx:
    Xset.append(f"{i}")
ax2.set_xticks(plotx,Xset)
ax1.set_xticks(plotx,Xset)
Yset=[]
for i in np.arange(0,3,0.5):
    Yset.append(f"{i}")
ax2.set_yticks(np.arange(0,3,0.5),Yset)
plt.subplots_adjust(hspace=0)
ax1.set_title("%s %s $M_{eff}$ and $\chi^2$(end t=%s)" % (state, object, end_t))
plt.savefig("/public/home/zhangxin/lattice-qcd/test1.jpg",dpi=400)
                
ed00=time.time()
print(f"fit job done, time used : %.3f s"%(ed00-st0))
print("")
print("")
        
# np.save("/public/home/liuxr/fit/fit_pion/fit_pion_date/fitE.npy",save_E)
