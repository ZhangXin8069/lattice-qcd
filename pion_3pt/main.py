import numpy as np
import matplotlib.pyplot as plt
# Ncnfg=20
# t_3pt=0
# t_2pt=12
# _3pt=np.load('_3pt.npy')[0,0,0,...]
# _2pt=np.load('_2pt.npy')[0,0,...]
# print("_3pt.shape:",_3pt.shape)
# print("_2pt.shape:",_2pt.shape)
# err_3pt=np.std(_3pt,axis=1)
# print("err_3pt.shape:",err_3pt.shape)
# data_3pt=np.mean(_3pt,axis=1)
# data_2pt=np.mean(_2pt,axis=0)
# print("data_3pt.shape",data_3pt.shape)
# print("data_2pt.shape",data_2pt.shape)
# _data_3pt=data_3pt[:,t_3pt]
# _data_2pt=data_2pt[t_2pt]
# _err_3pt=err_3pt[:,t_3pt]
# print("_data_3pt.shape",_data_3pt.shape)
# print("_data_2pt.shape",_data_2pt.shape)
# print("_err_3pt.shape",_err_3pt.shape)
# Y=_data_3pt/_data_2pt
# X=np.arange(len(Y))-(len(Y)-1)/2
# ERR=_err_3pt/_data_2pt
# print("Y.shape",Y.shape)
# print("ERR.shape",ERR.shape)
# print("X.shape",X.shape)
# print("Y:",Y)
# print("ERR:",ERR)
# print("X:",X)
# plt.rcParams.update({'font.size':25})
# fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
# fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
# ax.set_xlabel('Z')
# ax.set_ylabel('R = $C_{\mathrm{3pt}}$/$C_{\mathrm{2pt}}$')
# ax.errorbar(X,Y,yerr=ERR,fmt='o', alpha=0.5, capsize=3.5, capthick=1.5, label='Ncnfg='+str(Ncnfg))
# ax.legend()
# plt.title('pion 3pt')
# plt.savefig('pion_3pt.png')

Ncnfg=52
t_3pt=0
t_2pt=35
_3pt=np.load('_3pt.npy')[0,0,0,...]
_2pt=np.load('_2pt.npy')[0,0,...]
print("_3pt.shape:",_3pt.shape)
print("_2pt.shape:",_2pt.shape)
err_3pt=np.std(_3pt,axis=1)
print("err_3pt.shape:",err_3pt.shape)
data_3pt=np.mean(_3pt,axis=1)
data_2pt=np.mean(_2pt,axis=0)
print("data_3pt.shape",data_3pt.shape)
print("data_2pt.shape",data_2pt.shape)
_data_3pt=data_3pt[:,t_3pt]
_data_2pt=data_2pt[t_2pt]
_err_3pt=err_3pt[:,t_3pt]
print("_data_3pt.shape",_data_3pt.shape)
print("_data_2pt.shape",_data_2pt.shape)
print("_err_3pt.shape",_err_3pt.shape)
Y=_data_3pt/_data_2pt
X=np.arange(len(Y))-(len(Y)-1)/2
ERR=_err_3pt/_data_2pt
print("Y.shape",Y.shape)
print("ERR.shape",ERR.shape)
print("X.shape",X.shape)
print("Y:",Y)
print("ERR:",ERR)
print("X:",X)
plt.rcParams.update({'font.size':25})
fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
ax.set_xlabel('Z')
ax.set_ylabel('R = $C_{\mathrm{3pt}}$/$C_{\mathrm{2pt}}$')
ax.errorbar(X,Y,yerr=ERR,fmt='o', alpha=0.5, capsize=3.5, capthick=1.5, label='Ncnfg='+str(Ncnfg))
ax.legend()
plt.title('pion 3pt')
plt.savefig('pion_3pt.png')