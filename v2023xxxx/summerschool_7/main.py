import matplotlib.pyplot as plt
import cupy as np
# import cupy as cp
import math


gamma_5 = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, -1],]
)
# for k in range(10,49):
length = 39
c_sum = np.zeros([72], dtype='complex128')
for k in range(10, 10+length):
    n = "{}".format(k)
    n = n + "000"
    Green = []
    print(n)
    for i in range(0, 4):
        p = "{}".format(18*i)
        while len(p) < 2:
            p = '0' + p
        Green = np.load("/public/home/user7/summer_school/coulomb_wall_propagator/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_" +
                        n+"_hyp0_gfixed3.light.tsrc_"+p+".npy")
        # print(Green.shape)
        # x,t,β,β',a,a'  β',α' -> x,t,β,a,a',α'
        Green_gamma_u = np.einsum(
            "tzyxijkl,jm->tzyxiklm", Green, gamma_5, optimize=True)
        # print(Green_gamma_u.shape)
        # x,t,σ,σ',a,a'  σ',α' -> x,t,σ,α',a,a'
        Green_1 = np.einsum("tzyxijkl,jm->tzyximkl",
                            Green, gamma_5, optimize=True)
        # x,t,σ,α',a,a'  α,σ -> x,t,α,α',a,a'
        Green_1 = np.einsum("tzyxijkl,mi->tzyxmjkl",
                            Green_1, gamma_5, optimize=True)
        # x,t,β,a,a',α'  β,α -> x,t,a,a',α',α
        Green_gamma_u = np.einsum(
            "tzyxijkl,mi->tzyxjklm", Green_gamma_u, gamma_5, optimize=True)
        # x,t,a,a',α',α  x,t,α,α',a,a'
        c = np.einsum("tzyxijkl,tzyxlkij->t", Green_gamma_u,
                      Green_1.conjugate(), optimize=True)
        # print(c)
        # print(c.shape)
        c_sum += np.roll(c, -18*i)/4/length
        # print(math.log10(c.real))+
    print(c_sum)
