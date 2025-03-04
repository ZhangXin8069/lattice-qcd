import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import time
import lsqfit
import gvar as gv
from prettytable import PrettyTable
from lsq_tools import *
import glob
Nt0=96
nbsamples = 3000
Lt = 96
state="F48P30"
object= "$\Xi_{cc}$"
jack=False#True#
shape = "onestate"
SAVEdataandgraph=True# False#          #used to control whether save data and graph
SAVEchijpg= True#False#                #used to control whether save chi graph
effbandinchi=True# False#              #used to control whether plot fit band in chi graph
TEST=     False# True#                 #used to control whether test
Ptest= None# 4  #                      #used to control which momentum
deltat=  13               
if TEST==True:
    SAVEdataandgraph=False
else:
    if SAVEdataandgraph==True:
        deltat=  1 
        SAVEchijpg=False
    elif SAVEdataandgraph==False:
        deltat=  10 
        SAVEchijpg=True
if state=="F32P30":
    conf_name="beta6.41_mu-0.2295_ms-0.2050_L32x96"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12,1e-12])
    id=np.arange(590)*50+1000
    #print(id)
    Nev=100
    Eexp=np.array([1.47,1.48,1.52,1.54,1.56])
    beta=[36,39,31,31,31]
    ymax=1.55#0.5#1.2  #1.2 #
    ymin=1.45#0.35#0.8  #0.8 #
    start1=[18,21,17,19,21]#[20,18,17,16,14]
    if deltat!=1:
        start=[15,15,15,15,15]
    else:
        start=start1
    Nsamples = 567 #371
    Nmom = 5
elif state=="F48P30":
    conf_name="beta6.41_mu-0.2295_ms-0.2050_L48x96"
    #svdcuts=np.array([2.5e-3,5e-3,6e-3,2e-2])
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12,1e-12])
    Nev=200
    id=np.arange(201)*20+2700   
    Eexp=np.array([1.47,1.49,1.52,1.54,1.56])
    beta=[35,31,30,31,31]
    ymax=1.51#1.05#1.2  #1.2 #
    ymin=1.45#0.9#0.8  #0.8 #
    start1=[17,17,18,19,19]#[15,18,18,16,15]
    if deltat!=1:
        start=[15,15,15,15,15]
    else:
        #start=[15,18,15,15,15]
        start=start1
    Nsamples = 201  
    Nmom = 5
if state=="F48P21":
    conf_name="beta6.41_mu-0.2320_ms-0.2050_L48x96"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12,1e-12])
    id=np.arange(222)*20+1620
    Nev=200
    #print(id)
    Eexp=np.array([1.47,1.48,1.49,1.50,1.51])
    beta=[27,29,29,27,28]
    ymax=1.00#1.2  #1.2 #
    ymin=0.85#0.8  #0.8 #
    start1=[16,16,14,14,16]#[20,18,17,16,14]
    if deltat!=1:
        start=[10,10,10,10,12]
    else:
        start=start1
    Nsamples = 567 #371
    Nmom = 5
 
#data_all_mom=np.zeros([Nsamples,4,96],"<c16")
#id=np.zeros(450,"int")
# id[0:194]=np.arange(1000,10700,50)
# id[194:222]=np.arange(14300,15700,50)
# id[222:225]=np.arange(16550,16700,50)
data_all_mom=np.zeros([Nmom,Nsamples,96],"<c16")
a=read_dat_file(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/result/{conf_name}/eigen200/000/corr_Xicc_pp_Px0Py0Pz0.conf",".dat",False)
data_all_mom[0]=a
a=read_dat_file(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/result/{conf_name}/eigen200/001/corr_Xicc_pp_Px0Py0Pz1.conf",".dat",False)
data_all_mom[1]=a
a=read_dat_file(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/result/{conf_name}/eigen200/011/corr_Xicc_pp_Px0Py1Pz1.conf",".dat",False)
data_all_mom[2]=a
a=read_dat_file(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/result/{conf_name}/eigen200/111/corr_Xicc_pp_Px1Py1Pz1.conf",".dat",False)
data_all_mom[3]=a
a=read_dat_file(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/result/{conf_name}/eigen200/002/corr_Xicc_pp_Px0Py0Pz2.conf",".dat",False)
data_all_mom[4]=a
print("!!!",data_all_mom.shape)
data_all_mom=data_all_mom.transpose(1,0,2)
Nsamples=data_all_mom.shape[0]
print(data_all_mom.shape)
print("Nsamples=",Nsamples)  
if jack==True:
    boot_corr = jackknife(data_all_mom)
else:
    boot_corr = bootstrap(data_all_mom , nbsamples)
Nt = 40
if Ptest!=None:
    Pall=[Ptest]
else:
    Pall=np.arange(Nmom)
for p in Pall:
    print(F"momentum {p} fit job starts!!!!")
    st0 = time.time()
    contract = boot_corr[:, p,:Nt]
    print(contract.shape)
    #print(contract)
    N = contract.shape[0]  # your final number of samples
    print("N=", N)
    contract_sum = np.sum(contract, 0)
    sum_average = np.mean(contract, 0)
    # meson_mass_average = np.arccosh((np.roll(sum_average, -1) + np.roll(sum_average, 1)) / (2 * sum_average)) #/ Na* 0.1974
    # meson_mass_jk =np.arccosh((np.roll(contract, -1, 1) + np.roll(contract, 1, 1)) / (2 * contract)) #/ Na* 0.1974
    meson_mass_average = np.log(np.roll(sum_average, 1) / (sum_average)) #/ Na* 0.1974
    meson_mass_jk =np.log(np.roll(contract, 1, 1) / (contract)) #/ Na* 0.1974
    jkerros = np.std(meson_mass_jk, 0)
    if jack==True:
        jkerros=jkerros*np.sqrt(Nsamples-1) 
    # ------------------------fitting
    M_ij = contract - sum_average
    M_ij = np.matmul(M_ij.T, M_ij) 
    if jack==True:
        M_ij = M_ij*(Nsamples-1)/Nsamples
        print("jack!!")
    else:
        M_ij = M_ij/nbsamples
        print("boot!!")
    M_dig = np.tril(M_ij)
    M_dig = np.triu(M_dig)
    def efunc(x, p):
        t00=0
        if shape == "onestate":
            return p["a0"] * np.exp(-p["E0"] * (x-t00))
        else:
            return p["a0"] * np.exp(-p["E0"] * (x-t00)) + (p["a1"]) * np.exp(-(p["E0"]+p["dE"]**2)* (x -t00))
    if TEST==True or SAVEchijpg==True:
        N=150
    pr_arr_corr = np.zeros([N, 2])
    T = np.arange(Nt)
    chi_square = np.zeros([1, deltat])
    mass_mean = np.zeros([1, deltat])
    error_mean = np.zeros([1, deltat])
    save_E = np.zeros([deltat,N])
    save_chi = np.zeros([deltat,N])
    save_a = np.zeros([deltat,N])
    for t2 in range(1):  # right side
        for t1 in range(start[p], start[p] + deltat):  # left side
            st1=time.time()
            print("t1=", t1, "t2=", t2 + beta[p])
            
            p0 = {}
            p0["a0"] = 0.5
            p0["E0"] = Eexp[p]
            if shape=="twostate":
                p0["dE"] = 0.5
                p0["a1"] = 0.5
            # prior = {}
            # prior["a0"] = gv.gvar(0.3, 1e10)
            # prior["E0"] = gv.gvar(Eexp[p], 2)
            #if shape=="twostate":
                # prior["dE"] = gv.gvar(0.5, 1e10)
            
            gv_corr = gv.gvar(np.mean(contract,0), M_ij)
            fit_mean = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta[p] + 1], gv_corr[t1 : t2 + beta[p] + 1]), fcn=efunc,svdcut=1e-12,p0=p0)
            chi_mean=fit_mean.chi2/fit_mean.dof
            for fid in range(N):
                gv_corr = gv.gvar(contract[fid], M_ij)
                fit = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta[p] + 1], gv_corr[t1 : t2 + beta[p] + 1]), fcn=efunc,svdcut=1e-12,p0=p0)
                pr_corr = fit.p
                pr_arr_corr[fid][0] = pr_corr["E0"].mean
                pr_arr_corr[fid][1] = pr_corr["a0"].mean
                
                #print("id=",fid, "E=",fit.p["E0"].mean, "chis = ", fit.chi2/ (fit.dof+1))
                save_E[t1 - start[p],fid] = pr_corr["E0"].mean
                save_a[t1 - start[p],fid] = pr_corr["a0"].mean
                save_chi[t1 - start[p],fid] = fit.chi2/ (fit.dof)
            
            pr_aver_corr = np.mean(pr_arr_corr, 0)
            print(fit.p, "chis = ", chi_mean)  #-2
            print("dof",fit.dof)
            mass_mean[t2][t1 - start[p]] = pr_aver_corr[0]
            chi_square[t2][t1 - start[p]] = chi_mean
            
            error_mean[t2][t1 - start[p]] = np.std(pr_arr_corr, 0)[0]
            if jack==True:
                error_mean[t2][t1 - start[p]]=error_mean[t2][t1 - start[p]]*np.sqrt(Nsamples-1)
            
            ed1=time.time()
            print(f"{t1} to {beta[p]} fit done, time used : %.3f s"%(ed1-st1))
            
            ##-----------------------picture
            end = beta[p]
            E_mean = mass_mean[0][t1 - start[p]] #/ Na * 0.1974
            E_error = error_mean[0][t1 - start[p]] #/ Na * 0.1974
            # Nt=int(Nt/2+1)
            T = np.arange(Nt)
            fig = plt.figure()
            # plt.plot([t1,t1],[0,10],color='black',alpha=0.3)
            
            plt.errorbar(
                x=T,
                y=meson_mass_average,
                yerr=jkerros,
                ecolor="cornflowerblue",
                linestyle="none",
                mec="cornflowerblue",
                marker="o",
                alpha=0.7,
                markerfacecolor="none",
                capsize=2,
                capthick=1,
                label="Meff",
            )
            plt.legend(loc='upper right')
            left, bottom, width, height = (t1, (E_mean - E_error), end - t1, 2 * E_error)
            rect = mpatches.Rectangle(
                (left, bottom),
                width,
                height,
                # fill=False,
                alpha=0.4,
                facecolor="red",
                label="fitting",
            )
            plt.gca().add_patch(rect)
            if p >2:
                plt.text(10, (ymax-ymin)*0.25+ymin, f'fit $E$ = {E_mean}', fontsize=9)
                plt.text(10, (ymax-ymin)*0.2+ymin, f'error = {E_error}', fontsize=9) 
                plt.text(10, (ymax-ymin)*0.15+ymin, f'$\chi^2/d.o.f.$ = {chi_mean}', fontsize=9)                               
            else:
                plt.text(10, (ymax-ymin)*0.8+ymin, f'$\chi^2/d.o.f.$ = {chi_mean}', fontsize=9)
                plt.text(10, (ymax-ymin)*0.9+ymin, f'fit $E$ = {E_mean}', fontsize=9)
                plt.text(10, (ymax-ymin)*0.85+ymin, f'error = {E_error}', fontsize=9)
            plt.legend(loc='upper right') 
            plt.title("%s %s |P|$^2$=%s" % (state, object,p))
            plt.xlabel("t")
            plt.ylabel("%s aE" % (object))
            #plt.ylabel("%s P%s%s%s Meff/Gev" % (object, px, py, pz))
            plt.xlim(5, Nt)
            plt.ylim(ymin, ymax)
            if SAVEdataandgraph==True:
                plt.savefig(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/diLambdac/oneparticle/{state}/Xicc/effmass/Xicc_eff_P{p}.pdf", format='pdf',dpi=400)
                #plt.savefig(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/picture/{conf_name}/P{p}_t{t1}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}.jpg",dpi=400) #,dpi=200
            if TEST==True:
                plt.savefig(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/test.jpg",dpi=400) #,dpi=200
                #plt.show()
        ### --------------pretty table
        print("mass=", mass_mean)
        print("error=", error_mean)
        twostatefit = PrettyTable(
            ["start timeslice(end=" + str(beta[p]) + ")", "chi2/dof", "mass mean", "fitting error"]
        )
        for i in range(deltat):
            twostatefit.add_row(
                [
                    np.arange(deltat)[i] + start[p],
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
        #mass_mean = mass_mean #/ Na * 0.1974
        #error_mean = error_mean #/ Na * 0.1974
        ax1.errorbar(
            x=np.arange(deltat) + start[p],
            y=mass_mean[0],
            yerr=error_mean[0],
            ecolor="red",
            linestyle="none",
            mec="red",
            marker="o",
            alpha=0.7,
            markerfacecolor="none",
            capsize=2,
            capthick=1,
            #label="%s $|P|^2$=%s $M_{eff}$" % (object,p),
        )
        plotx=np.arange(deltat) + start[p]
        if effbandinchi==True:
            selectid=start1[p]-start[p]
            mean=mass_mean[0,selectid]
            err=error_mean[0,selectid]
            ax1.fill_between([plotx[0]-0.5,plotx[-1]+0.5],[mean+err,mean+err],\
                [mean-err,mean-err],alpha=0.2,color="red")
        ax2.scatter(x=plotx,y=chi_square[0],color="cornflowerblue")
        ax2.scatter(x=selectid+start[p],y=chi_square[0,selectid],color="brown")
        ax2.text(selectid+start[p],chi_square[0,selectid]+0.25,f"{chi_square[0,selectid]:0.2f}",ha='center')
        ax1.text(selectid+start[p],mass_mean[0,selectid]+error_mean[0,selectid]*1.2,\
                 f"{mass_mean[0,selectid]:0.4f}"+"("+"%0.0f"%(error_mean[0,selectid]*10000)+")",ha='center')
        ax2.plot(plotx,np.ones_like(plotx),color="grey",alpha=0.7,linestyle="--")
        auto_ymin=np.mean(mass_mean[0,selectid])-error_mean[0,selectid]*5
        auto_ymax=np.mean(mass_mean[0,selectid])+error_mean[0,selectid]*5
        ax2.set_xlabel("start t")
        ax1.set_ylabel("$fit$ $M_{eff}$")
        ax2.set_ylabel("$\chi^2/d.o.f.$")
        ax1.set_ylim(auto_ymin, auto_ymax)
        ax2.set_ylim(0, 3)   
        #ax1.legend(loc=1)
        #ax2.legend(loc=1)
        Xset=[]
        for i in plotx:
            Xset.append(f"{i}")
        ax2.set_xticks(plotx,Xset)
        Yset=[]
        for i in np.arange(0,3,0.5):
            Yset.append(f"{i}")
        ax2.set_yticks(np.arange(0,3,0.5),Yset)
        plt.subplots_adjust(hspace=0)
        ax1.set_title("%s %s $|P|^2$=%s $M_{eff}$ and $\chi^2$(end t=%s)" % (state, object, p, beta[p]))
        if SAVEchijpg==True:
            plt.savefig(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/diLambdac/oneparticle/{state}/Xicc/chi/Xicc_fitchi_P{p}.pdf", format='pdf',dpi=400)
            #plt.savefig(f"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/picture/{conf_name}/chi_Meff_P{p}_{start[p]}_{beta[p]}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}.jpg",dpi=400)
        if TEST==True:
            plt.savefig("/public/home/zhangxin/lattice-lqcd/meson_run1110/test1.jpg",dpi=400)
                    
    ed00=time.time()
    print(f"all p={p} job done, time used : %.3f s"%(ed00-st0))
    print("")
    print("")
            
    # if SAVEdataandgraph==True:
        # np.save(F"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/fit_result/{conf_name}/p{p}_{start[p]}_{beta[p]}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}_one_E.npy",save_E)
        # np.save(F"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/fit_result/{conf_name}/param_a/p{p}_{start[p]}_{beta[p]}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}_one_a.npy",save_a)
        # np.save(F"/public/home/zhangxin/lattice-lqcd/meson_run1110/laph/Xicc/fit_result/{conf_name}/param_chi/p{p}_{start[p]}_{beta[p]}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}_one_chi.npy",save_chi)
    
    
