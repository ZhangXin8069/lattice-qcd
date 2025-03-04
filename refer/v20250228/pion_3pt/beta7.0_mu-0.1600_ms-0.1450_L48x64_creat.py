from creat_chroma import *
import sys
hadron = str(sys.argv[1])
conf_id = str(sys.argv[2])
tsep=str(sys.argv[3])
quark_mass=str(sys.argv[4])
chroma = creat_chroma_ini(
    lattice_size = [48,48,48,64],
    hadron = hadron,
    conf_id = conf_id,
    conf_dir = '/public/home/sunp/sunpeng/hmc_tune_2023_10_15/hmc_tune_sunpeng_v3_fix_mass_1600_1450_b7.00_u0_48_64_2st/beta7.0_mu-0.1600_ms-0.1450_L48x64_cfg_',
    out_path = '/public/home/sush/share_work/chroma/beta7.0_mu-0.1600_ms-0.1450_L48x64/mom_grid_source/new/',
)
if hadron == 'pion':
    hadron_list = 100001515
elif hadron == 'nucleon':
    hadron_list = 200505
    
chroma.begin("beta7.0_mu-0.1600_ms-0.1450_L48x64 pion 2pt test mass%s"%(quark_mass))
chroma.Coulomb_gauge_fix()
chroma.mom_source(grid=[1,1,1,64])
chroma.propagator(blocking1=[6,6,6,4], blocking2=[2,2,2,2], name='prop_1', clovcoeff = '1.103643830721', quark_mass = quark_mass)
chroma.shell_sink_smear(name = 'smeared_prop_1', prop = 'prop_1')
chroma.ERASE_NAMED_OBJECT(name='prop_1')
chroma.HADRON_SPECTRUM_v2(hadron_list, smeared_prop = 'smeared_prop_1')
chroma.ERASE_NAMED_OBJECT(name='smeared_prop_1')
chroma.ERASE_NAMED_OBJECT(name='source')
chroma.end()
