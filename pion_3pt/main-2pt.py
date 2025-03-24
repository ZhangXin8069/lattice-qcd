from include import *
import numpy as np
# filepath = np.array([
#     './beta6.20_mu-0.2770_ms-0.2400_L%dx%d/iog/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog',
#                      ])
# analyse = data_analyse(
#     num_data=1,
#     hadron='pion',
#     filepath=filepath,
#     alttc=0.1053,
#     Nx=24,
#     Nt=72,
#     P=np.asarray([[0,0,0]]),
#     ENV=np.asarray(range(-1,0,1)),
#     N_start=10000,
#     gap=50,
#     Ncnfg_data=0,
#     Ncnfg_iog=20,
#     tsep=np.asarray([10]),
#     time_fold=0,
#     save_path='./',
#     link_max=10,
#     analyse_type='2pt',
#     meff_type='cosh',
#     read_type='iog',
# )
# meff_range = [0,1.0]
# analyse.meff_2pt('iog')
# print(analyse.meff_data_2pt)
# analyse.plot_meff_2pt('iog', meff_range)
# print('complete')

filepath = np.array([
    './beta6.20_mu-0.2770_ms-0.2400_L%dx%d/sush_iog/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog',
                     ])
analyse = data_analyse(
    num_data=1,
    hadron='pion',
    filepath=filepath,
    alttc=0.1053,
    Nx=24,
    Nt=72,
    P=np.asarray([[0,0,0]]),
    ENV=np.asarray(range(-1,0,1)),
    N_start=10050,
    gap=50,
    Ncnfg_data=0,
    Ncnfg_iog=52,
    tsep=np.asarray([36]),
    time_fold=0,
    save_path='./',
    link_max=10,
    analyse_type='2pt',
    meff_type='cosh',
    read_type='iog',
)
meff_range = [0,1.0]
analyse.meff_2pt('iog')
print(analyse.meff_data_2pt)
analyse.plot_meff_2pt('iog', meff_range)
print('complete')
