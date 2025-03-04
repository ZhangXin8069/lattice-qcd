import numpy as np
from opt_einsum import contract
import cupy as cp
gamma_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
gamma_1 = np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]])
gamma_2 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
gamma_3 = np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]])
gamma_4 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
gamma_5 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
C = gamma_2 @ gamma_4
P = (gamma_0 + gamma_4) / 2
epsilon = np.zeros((3, 3, 3), np.int32)
for a in range(3):
    b, c = (a + 1) % 3, (a + 2) % 3
    epsilon[a, b, c] = 1
    epsilon[a, c, b] = -1
cfg = 10000
pion_numpy = np.zeros((4, 72), np.complex128)
proton_numpy = np.zeros((4, 72), np.complex128)
pion_cupy = cp.zeros((4, 72), cp.complex128)
proton_cupy = cp.zeros((4, 72), cp.complex128)
for tidx, tsrc in enumerate(range(0, 72, 72)):
    propag_numpy = np.load(f"/public/home/jiangxy/summer_school/coulomb_wall_propagator/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{cfg}_hyp0_gfixed3.light.tsrc_{tsrc:02d}.npy")
    pion_numpy[tidx] = contract(
        "tzyxijab,jk,tzyxlkab,li->t",
        propag_numpy,
        gamma_5 @ gamma_5,
        propag_numpy.conj(),
        gamma_5 @ gamma_5,
    )
for tidx, tsrc in enumerate(range(0, 72, 72)):
    propag_cupy = np.load(f"/public/home/jiangxy/summer_school/coulomb_wall_propagator/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{cfg}_hyp0_gfixed3.light.tsrc_{tsrc:02d}.npy")
    pion_cupy[tidx] = contract(
        "tzyxijab,jk,tzyxlkab,li->t",
        propag_cupy,
        gamma_5 @ gamma_5,
        propag_cupy.conj(),
        gamma_5 @ gamma_5,
    )
print(np.arange(12))
print(np.roll(np.arange(12), -1, 0))
print(np.roll(np.arange(12), -4, 0))
for tidx, tsrc in enumerate(range(0, 72, 18)):
    pion_numpy[tidx] = np.roll(pion_numpy[tidx], -tsrc, 0)
pion_numpy_tsrc_mean = pion_numpy.mean(0)
for tidx, tsrc in enumerate(range(0, 72, 18)):
    pion_cupy[tidx] = cp.roll(pion_cupy[tidx], -tsrc, 0)
pion_cupy_tsrc_mean = pion_cupy.mean(0)
print(pion_numpy[0])
print("cupy:")
print(pion_cupy[0])