from include import *
import numpy as np
# filepath = np.array([
#     './beta6.20_mu-0.2770_ms-0.2400_L%dx%d/iog/pion_U_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d_conserved_true.iog', 
#     './beta6.20_mu-0.2770_ms-0.2400_L%dx%d/iog/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog',
#                      ])
# analyse = data_analyse(
#     num_data=2,
#     hadron='pion',
#     filepath=filepath,
#     alttc=0.1053,
#     Nx=24,
#     Nt=72,
#     time_fold=True,
#     P=np.asarray([[0,0,0]]),
#     ENV=np.asarray(range(-1,0,1)),
#     N_start=10000,
#     gap=50,
#     Ncnfg_data=20,
#     Ncnfg_iog=20,
#     tsep=np.asarray([10]),
#     link_max=10,
#     save_path='./',
#     analyse_type='ratio',
#     meff_type='cosh',# for 2pt
#     read_type='iog',
# )
# np.save('_2pt.npy', analyse.readed_jcknf['readed_2pt_iog'])
# np.save('_3pt.npy', analyse.readed_jcknf['readed_3pt_iog_U'])
# print('complete')
filepath = np.array([
    './beta6.20_mu-0.2770_ms-0.2400_L%dx%d/sush_iog/pion_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d.iog', 
    './beta6.20_mu-0.2770_ms-0.2400_L%dx%d/sush_iog/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog',
                     ])
analyse = data_analyse(
    num_data=2,
    hadron='pion',
    filepath=filepath,
    alttc=0.1053,
    Nx=24,
    Nt=72,
    time_fold=True,
    P=np.asarray([[0,0,0]]),
    ENV=np.asarray(range(-1,0,1)),
    N_start=10050,
    gap=50,
    Ncnfg_data=0,
    Ncnfg_iog=52,
    tsep=np.asarray([36]),
    link_max=10,
    save_path='./',
    analyse_type='ratio',
    meff_type='cosh',# for 2pt
    read_type='iog',
)
np.save('_2pt.npy', analyse.readed_jcknf['readed_2pt_iog'])
np.save('_3pt.npy', analyse.readed_jcknf['readed_3pt_iog_U'])
print('complete')
