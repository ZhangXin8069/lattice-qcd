num_quark 2
alttc 0.1057
Nt 72
Nx 24
N_start 10050
Ncnfg 40 
gap 50
link_max 10
t_sep_start 36
t_sep_end 36
t_sep_gap 1
ENV_start 10
ENV_end 100
ENV_gap 10
Pz_start 0
Pz_end 0
Py_start 0
Py_end 0
Px_start 0
Px_end 0
time_fold 0
# if time_fold == 1:then will subtract the forward time and backward time; else do nothing
type ratio
# 2pt or ratio
# 2pt means we will calculate the part of 2pt ratio means we will calculate C3pt, C2pt and C3pt/C2pt
read_type both
# data and iog and both
C2pt_type cosh
# log and cosh
iog_corr_2pt_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog
# corr_2pt_path /public/home/sush/share_work/chroma/beta7.0_mu-0.1600_ms-0.1450_L%dx%d/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.1450.iog
# iog_corr_2pt_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass_-0.2770.iog
iog_quark_3pt_corr_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d.iog
# corr_2pt_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d_24x72/corr_uuu_conf%d_2pt.dat
data_quark_3pt_corr_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_linkZ%d_U.dat
data_corr_2pt_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/ENV_%d/conf%d/corr_ud_2pt.dat
# first_quark_3pt_corr_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_linkZ%d_U.dat
# corr_2pt_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/ENV_%d/conf%d/corr_ud_2pt.dat
save_path .