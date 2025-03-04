from creat_chroma import *
import sys
conf_id = str(sys.argv[1])
hadron = 'pion'
quark_mass = '-0.2770'
tsep='10'
chroma = creat_chroma_ini(
    lattice_size = [24, 24, 24, 72],
    hadron = hadron,
    conf_id = conf_id,
    conf_dir = '/public/home/chenchen/configuration/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_',
    out_path = '/public/home/sushihao/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L24x72/pion/',
)
if hadron == 'pion':
    hadron_list = 100001515
elif hadron == 'nucleon':
    hadron_list = 200505
chroma.begin("3pt test")
# chroma.stout_smear(1,0.125)
chroma.point_source()
chroma.propagator(blocking1=[3,3,3,2], blocking2=[2,2,2,2], clovcoeff= '1.160920226', quark_mass = quark_mass)
chroma.point_sink_smear()
chroma.HADRON_SPECTRUM_v2(hadron_list)
chroma.seqsource_fast(multi_tSinks = tsep, SeqSourceType = 'pion_1-pion_1', name = 'seq_source_U', Flavor='U')
chroma.propagator(blocking1=[3,3,3,2], blocking2=[2,2,2,2], clovcoeff= '1.160920226', quark_mass = quark_mass, name='seq_prop_U', source='seq_source_U')
chroma.building_block(links_max = 10, frwd_prop_id='prop', bkwd_prop_id = 'seq_prop_U', Flavor = 'U', conserved = 'true', use_cpu = 'true')
chroma.end()