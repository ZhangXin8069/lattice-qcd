#!/beegfs/home/liuming/software/install/python/bin/python3
import numpy as np
import os
#identity
g0=np.zeros((4,4),dtype=complex)
g0[0,0]=1.0+0.0*1j         #  1  0  0  0
g0[1,1]=1.0+0.0*1j         #  0  1  0  0
g0[2,2]=1.0+0.0*1j         #  0  0  1  0
g0[3,3]=1.0+0.0*1j         #  0  0  0  1 
#gamma1
g1=np.zeros((4,4),dtype=complex)
g1[0,3]=0.0+1.0*1j         #  0  0  0  i
g1[1,2]=0.0+1.0*1j         #  0  0  i  0
g1[2,1]=0.0-1.0*1j         #  0 -i  0  0
g1[3,0]=0.0-1.0*1j         # -i  0  0  0
#gamma2
g2=np.zeros((4,4),dtype=complex)
g2[0,3]=-1.0+0.0*1j        #  0  0  0 -1
g2[1,2]=1.0+0.0*1j         #  0  0  1  0
g2[2,1]=1.0+0.0*1j         #  0  1  0  0
g2[3,0]=-1.0+0.0*1j        # -1  0  0  0
#gamma3
g3=np.zeros((4,4),dtype=complex)
g3[0,2]=0.0+1.0*1j         #  0  0  i  0
g3[1,3]=0.0-1.0*1j         #  0  0  0 -i
g3[2,0]=0.0-1.0*1j         # -i  0  0  0
g3[3,1]=0.0+1.0*1j         #  0  i  0  0
#gamma4
g4=np.zeros((4,4),dtype=complex)
g4[0,2]=1.0+0.0*1j         #  0  0  1  0
g4[1,3]=1.0+0.0*1j         #  0  0  0  1
g4[2,0]=1.0+0.0*1j         #  1  0  0  0
g4[3,1]=1.0+0.0*1j         #  0  1  0  0
#gamma5
g5=np.zeros((4,4),dtype=complex)
g5[0,0]=1.0+0.0*1j         #  1  0  0  0
g5[1,1]=1.0+0.0*1j         #  0  1  0  0
g5[2,2]=-1.0+0.0*1j        #  0  0 -1  0
g5[3,3]=-1.0+0.0*1j        #  0  0  0 -1
def gamma(i):
	if i==0: #identity
		return g0
		
	elif i==1: #gamma1
		return g1
		
	elif i==2: #gamma2
		return g2
	elif i==3: #gamma3
		return g3
	
	elif i==4: #gamma4
		return g4
	
	elif i==5: #gamma5
		return g5
	elif i==6: #-gamma1*gamma4*gamma5 (gamma2*gamma3)
		return np.matmul(g2,g3)
		
	elif i==7: #-gamma2*gamma4*gamma5 (gamma3*gamma1)
		return np.matmul(g3,g1)
 
	elif i==8: #-gamma3*gamma4*gamma5 (gamma1*gamma2)
		return np.matmul(g1,g2)
 
	elif i==9: #gamma1*gamma4
		return np.matmul(g1,g4)
 
	elif i==10: #gamma2*gamma4
		return np.matmul(g2,g4)
 
	elif i==11: #gamma3*gamma4
		return np.matmul(g3,g4)
 
	elif i==12: #gamma1*gamma5
		return np.matmul(g1,g5)
 
	elif i==13: #gamma2*gamma5
		return np.matmul(g2,g5)
 
	elif i==14: #gamma3*gamma5
		return np.matmul(g3,g5)
 
	elif i==15: #gamma4*gamma5
		return np.matmul(g4,g5)
#C*gammai, Cgamma4, Cgamma5, Cgamma5*gammai for Tcc contraction, C=i g2 g4
	elif i==-1: #C*gamma1
		return 1j*g2@g4@g1
	elif i==-2: #C*gamma2
		return -1j*g4
	elif i==-3: #C*gamma3
		return 1j*g2@g4@g3
	elif i==-4: #C*gamma4
		return 1j*g2
	elif i==-5: #gamma5*C / C*gamma5
		return 1j*g5@g2@g4
	elif i==-6: #gamma5*gamma1*C
		return 1j*g5@g1@g2@g4
	elif i==-7: #gamma5*gamma2*C
		return 1j*g5@g4
	elif i==-8: #gamma5*gamma3*C
		return 1j*g5@g3@g2@g4
	
	elif i==-11: #gamma1*C
		return 1j*g1@g2@g4
	elif i==-22: #gamma2*C
		return 1j*g4
	elif i==-33: #gamma3*C
		return 1j*g3@g2@g4
	elif i==-44: #gamma4*C
		return -1j*g2
	elif i==-66: #C*gamma5*gamma1
		return 1j*g2@g4@g5@g1
	elif i==-77: #C*gamma5*gamma2
		return 1j*g4@g5
	elif i==-88: #C*gamma5*gamma3
		return 1j*g2@g4@g5@g3
	else:
		print("wrong gamma index")
		os.sys.exit(-3)
def gamma_index(g):
	value=np.zeros((4),dtype=complex)
	row=np.zeros((4),dtype=int)
	col=np.zeros((4),dtype=int)
	count=0
	for i in range(4):
		for j in range(4):
			if(np.abs(g[i,j]) != 0.0):
				value[count]=g[i,j]
				row[count]=i
				col[count]=j
				count=count+1
	return value, row, col
 
