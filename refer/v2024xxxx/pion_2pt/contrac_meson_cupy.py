import numpy as np
import cupy as cp
import math
import os
import fileinput
from gamma_matrix_cupy_DR import *
from input_output_4_cupy import *
from opt_einsum import contract
import time
print("Job started")
Nc=3
# ------------------------------------------------------------------------------
infile=fileinput.input()
for line in infile:
  tmp=line.split()
  if(tmp[0]=='Nt'):
    Nt=int(tmp[1])
  if(tmp[0]=='Nx'):
    Nx=int(tmp[1])
  if(tmp[0]=='conf_id'):
    conf_id=tmp[1]
  if(tmp[0]=='Nev1'):  #number of eigenvectors used in contraction
    Nev1=int(tmp[1])
  if(tmp[0]=='Px'):
    Px=int(tmp[1])
  if(tmp[0]=='Py'):
    Py=int(tmp[1])
  if(tmp[0]=='Pz'):
    Pz=int(tmp[1])
  if(tmp[0]=='peram_u_dir'):
    peram_u_dir=tmp[1]
  if(tmp[0]=='eigen_dir'):
    eig_dir=tmp[1]
  if(tmp[0]=='corr_pion_dir'):
    corr_pion_dir=tmp[1]
  if(tmp[0]=='VdV_dir'):
    VdV_dir=tmp[1]
# ------------------------------------------------------------------------------
def VDV_assemble(eig_dir, t,  Nev1, conf_id, Nx, Px, Py, Pz):
    Mom = np.array([Pz,Py,Px])
    
    exp_diag = np.zeros(Nx*Nx*Nx*Nc, dtype=complex)
    for z in range(0,Nx):
        for y in range(0,Nx):
            for x in range(0,Nx):
                Pos = np.array([z,y,x])
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3] = np.exp( -np.dot(Mom,Pos)*2*math.pi*1j/Nx )
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 1] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 2] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]
    eigvecs_cupy = readin_eigvecs_device(eig_dir, t,Nev1,conf_id, Nx)
    VDV_cupy = cp.zeros((Nev1, Nev1), dtype=complex)
    exp_diag_cupy=cp.asarray(exp_diag)
    VDV_cupy=cp.matmul((cp.conj(cp.transpose(eigvecs_cupy)) * exp_diag_cupy), eigvecs_cupy) 
    
    return VDV_cupy
# ------------------------------------------------------------------------------
corr_pion=np.zeros((1,Nt), dtype=complex)
st_vdv = time.time()
if Px==0 and Py==0 and Pz==0 :
    VdV_sink=cp.zeros((Nt, Nev1, Nev1), dtype=complex)
    for t in range(0,Nt):
        VdV_sink[t]=cp.identity(Nev1)
else :
    #VdV_sink=readin_VdV_all_device(VdV_dir,  Nev1, Nt, conf_id, Px, Py, Pz)
    VdV_sink = np.zeros([Nt,Nev1,Nev1],dtype=complex)
    for t in range(Nt):
      VdV_sink[t]=VDV_assemble(eig_dir, t,  Nev1, conf_id, Nx, Px, Py, Pz)
    
ed_vdv = time.time()
print("VDV done, time used: %.3f s" %(ed_vdv-st_vdv))
corr_pion_matrix=cp.zeros((Nt,Nt), dtype=complex)
G5Gx_sink = gamma(0)
GxG5_source = gamma(0)
for t_source in range(0,Nt):
  
    st0 = time.time()
    st = time.time()
    peram_u=readin_peram_device(peram_u_dir, conf_id, Nt, Nev1, t_source)
    # peram_uGxG5=contract("znsag,st->zntag", peram_u, GxG5_source)
    ed = time.time()
    print("read peram_u done, time used: %.3f s" %(ed-st))
    VdV_source = cp.conj(VdV_sink[t_source].T)
    st = time.time()
    # corr_pion_matrix[t_source] = contract("tli,eb,tbcij,cf,teflk,jk->t",  VdV_sink, G5Gx_sink, peram_u, GxG5_source, cp.conj(peram_u), VdV_source )
    corr_pion_matrix[t_source] = contract("tli,tecij,teclk,jk->t",  VdV_sink, peram_u, cp.conj(peram_u), VdV_source )
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()
    ed = time.time()
    print("contraction done, time used: %.3f s" %(ed-st))
    
    del peram_u
    
    ed0 = time.time()
    print("****************all complete for t_source %d, time used: %.3f s****************" %(t_source, ed0-st0))
corr_pion_matrix=cp.asnumpy(corr_pion_matrix)
for t_source in range(0,Nt):
    for t_sink in range(0,Nt):
        corr_pion[0,(t_sink-t_source+Nt)%Nt] = corr_pion[0,(t_sink-t_source+Nt)%Nt] + corr_pion_matrix[t_source,t_sink]
write_data_ascii(corr_pion, Nt, Nx, "%s/corr_uu_gamma5_Px%sPy%sPz%s_conf%s.dat"%(corr_pion_dir,Px,Py,Pz,conf_id))