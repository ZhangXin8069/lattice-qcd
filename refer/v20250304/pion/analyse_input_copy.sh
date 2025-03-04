num_quark 2
alttc 0.04
Nt 64
Nx 48
N_stare 800
Ncnfg 30 
gap 20
link_max 0
t_sep_stare -1
t_sep_end -1
t_sep_gap 1
ENV_stare -1
ENV_end -1
ENV_gap 1
Pz_stare 0
Pz_end 0
Py_stare 0
Py_end 0
Px_stare 0
Px_end 0
time_fold 0
# if time_fold == 1:then will subtract the forward time and backward time; else do nothing
type 2pt
# 2pt or ratio
# 2pt means we will calculate the part of 2pt ratio means we will calculate C3pt, C2pt and C3pt/C2pt
read_type iog
# data and iog and both
C2pt_type cosh
# log and cosh
iog_quark_3pt_corr_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d.iog
# iog_corr_2pt_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog
# iog_quark_3pt_corr_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_3pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep%d_mass-0.2770_linkdir2_linkmax%d.iog
# iog_corr_2pt_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.2770.iog
iog_corr_2pt_path /public/home/sush/share_work/chroma/beta7.0_mu-0.1600_ms-0.1450_L%dx%d/wall_source/pion_2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass-0.1450.iog
# iog_corr_2pt_path /public/home/sush/share_work/chroma/beta6.20_mu-0.2770_ms-0.2400_L%dx%d/2pt_Px%dPy%dPz%d_ENV%d_conf%d_tsep-1_mass_-0.2770.iog
# corr_2pt_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/test/ENV_%d_24x72/corr_uuu_conf%d_2pt.dat
# first_quark_3pt_corr_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/ENV_%d/conf%d/corr_uud_gamma4_3pt_tseq%d_linkZ%d_U.dat
# corr_2pt_path /public/home/sush/3pt_distillation/pion/result/%dx%d/Px%dPy%dPz%d/ENV_%d/conf%d/corr_ud_2pt.dat
save_path ./beta7.0_mu-0.1600_ms-0.1450_L48x64/