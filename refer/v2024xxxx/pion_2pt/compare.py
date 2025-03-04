import numpy as np
# your two data path to be compared
datapath1="/public/home/changx/meson_run1110/result/2.dat"
datapath2="/public/home/changx/meson_run1110/result/1.dat"
###################################################
# if data saved in '.dat' form, please use below to load:
file1 = open(F"{datapath1}","rb")
real1 = []
imaginary1=[]
for line in file1:
    tmp = line.split()
    real1.append(float(tmp[1]))
    imaginary1.append(float(tmp[2]))
del real1[0]
del imaginary1[0]
file1.close()
data1=np.array(real1)+np.array(imaginary1)*1j
print(data1)
file2 = open(F"{datapath2}","rb")
real2 = []
imaginary2=[]
for line in file2:
    tmp = line.split()
    real2.append(float(tmp[1]))
    imaginary2.append(float(tmp[2]))
del real2[0]
del imaginary2[0]
file2.close()
data2=np.array(real2)+np.array(imaginary2)*1j
print(data2)
###################################################
# if data saved in '.npy' form, please use below to load:
# data1=np.load(datapath1)
# data2=np.load(datapath2)
###################################################
norm_square_sum=np.sum(abs((data1-data2)/data1)**2)
print("norm_square_sum=",norm_square_sum)