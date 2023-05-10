#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:30:46 2021

@author: fearthekraken
"""
import AS
import pwaves
import sleepy
import pandas as pd

#%%
###   FIGURE 1C - example EEGs for NREM, IS, and REM   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'

AS.plot_example(ppath, 'hans_091118n1', ['EEG'], tstart=721.5, tend=728.5, eeg_nbin=4, ylims=[(-0.6, 0.6)]) # NREM EEG
AS.plot_example(ppath, 'hans_091118n1', ['EEG'], tstart=780.0, tend=787.0, eeg_nbin=4, ylims=[(-0.6, 0.6)]) # IS EEG
AS.plot_example(ppath, 'hans_091118n1', ['EEG'], tstart=818.5, tend=825.5, eeg_nbin=4, ylims=[(-0.6, 0.6)]) # REM EEG

#%%
###   FIGURE 1E - example photometry recording   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
AS.plot_example(ppath, 'hans_091118n1', tstart=170, tend=2900, PLOT=['EEG', 'SP', 'EMG_AMP', 'HYPNO', 'DFF'], dff_nbin=1800, 
                eeg_nbin=130, fmax=25, vm=[50,1800], highres=False, pnorm=0, psmooth=[2,5], flatten_is=4, ma_thr=0)

#%%
###   FIGURE 1F - average DF/F signal in each brain state   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]

df = AS.dff_activity(ppath, recordings, istate=[1,2,3,4], ma_thr=20, flatten_is=4, ma_state=3)

#%%
###   FIGURE 1G - example EEG theta burst & DF/F signal   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
AS.plot_example(ppath, 'hans_091118n1', tstart=2415, tend=2444, PLOT=['SP', 'DFF'], dff_nbin=450, fmax=20, 
                vm=[0,5], highres=True, recalc_highres=False, nsr_seg=2.5, perc_overlap=0.8, pnorm=1, psmooth=[4,4])

#%%
###   FIGURE 1H - average spectral field during REM   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]
recordings = [recordings[0]]

pwaves.spectralfield_highres_mice(ppath, recordings, pre=4, post=4, istate=[1], theta=[1,10,100,1000,10000], pnorm=1, 
                                  psmooth=[6,1], fmax=25, nsr_seg=2, perc_overlap=0.8, recalc_highres=True)

#%%
###   FIGURE 2B - recorded P-waveforms  ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions'

# left - example LFP trace with P-waves   
AS.plot_example(ppath, 'Fincher_040221n1', tstart=16112, tend=16119, PLOT=['LFP'], lfp_nbin=7, ylims=[(-0.4, 0.2)])

# right - average P-waveform
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
df = pwaves.avg_waveform(ppath, recordings, istate=[],  win=[0.15,0.15], mode='pwaves', plaser=False, ci='sd')

#%%
###   FIGURE 2C - average P-wave frequency in each brain state   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
istate = [1,2,3,4]; p_iso=0; pcluster=0

_ = pwaves.state_freq(ppath, recordings, istate, plotMode='04', ma_thr=20, flatten_is=4, ma_state=3,
                      p_iso=p_iso, pcluster=pcluster, ylim2=[-0.3, 0.1], mouse_avg='mouse', avg_mode='each')

#%%
###   FIGURE 2D - time-normalized P-wave frequency across brain state transitions   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; vm=[0.2, 2.1]  # NREM --> IS --> REM --> WAKE

_, mx_pwave, _ = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, ma_thr=20, ma_state=3, 
                                       flatten_is=4, fmax=25, pnorm=1, vm=vm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', print_stats=False)

#%%
###   FIGURE 2E - example theta burst & P-waves   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
AS.plot_example(ppath, 'Scrabble_072420n1', tstart=11318.6, tend=11323, PLOT=['SP','EEG','LFP'], eeg_nbin=1, lfp_nbin=6, fmax=20, 
                vm=[0,4.5], highres=True, recalc_highres=False, nsr_seg=1, perc_overlap=0.85, pnorm=1, psmooth=[4,5])

#%%
###   FIGURE 2F - averaged spectral power surrounding P-waves   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
filename = 'sp_win3'

# top - averaged spectrogram
pwaves.avg_SP(ppath, recordings, istate=[1], win=[-3,3], mouse_avg='mouse', plaser=False, pnorm=2, psmooth=[2,2], fmax=25, 
              vm=[0.8,1.5], pload=filename, psave=filename)

# bottom - averaged high theta power
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], bands=[(8,15)], band_colors=['green'], win=[-3,3], mouse_avg='mouse', 
                          plaser=False, pnorm=2, psmooth=0, ylim=[0.6,1.8], pload=filename, psave=filename)

#%%
###   FIGURE 2H - example DF/F signal and P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
AS.plot_example(ppath, 'Fritz_032819n1', tstart=2991, tend=2996.75, PLOT=['DFF','LFP_THRES_ANNOT'], dff_nbin=50, lfp_nbin=10)

#%%
###   FIGURE 2I - DF/F signal surrounding P-waves (20 s window)   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]

# heatmap
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='h', dff_win=[-10,10], 
                          pzscore=2, mouse_avg='mouse', dn=1000, print_stats=False)
# DF/F time course
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='t', dff_win=[-10,10], 
                          pzscore=2, mouse_avg='mouse', dn=250, sf=1000, base_int=2.5)

#%%
###   FIGURE 2J - DF/F signal surrounding P-waves (2 s window)   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]
filename = 'photometry_win1'
# spectrogram
pwaves.avg_SP(ppath, recordings, istate=[], win=[-1,1], mouse_avg='mouse', plaser=False, pnorm=2, psmooth=[5,5], fmax=25, 
              pload=filename, psave=filename)
# DF/F timecourse
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='t', dff_win=[-1,1], pzscore=2, 
                          mouse_avg='mouse', dn=1, sf=800, z_win=[-10,10], base_int=0.2, flatten_is=4, ma_state=3)

#%%
###   FIGURE 2K - DF/F signal surrounding single and clustered P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]

# single P-waves
iso_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='t', dff_win=[-10,10], 
                               pzscore=2, mouse_avg='mouse', dn=250, sf=1000, p_iso=0.8, base_int=2.5)

# clustered P-waves
clus_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='t', dff_win=[-10,10], 
                               pzscore=2, mouse_avg='mouse', dn=250, sf=1000, pcluster=0.5, clus_event='waves', base_int=2.5)

#%%
###   FIGURE 3B - example open loop opto recording   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'
AS.plot_example(ppath, 'Huey_082719n1', tstart=12300, tend=14000, PLOT=['LSR', 'SP', 'HYPNO'], fmax=25, vm=[50,1800], highres=False,
                pnorm=0, psmooth=[2,2], flatten_is=4, ma_thr=10)

#%%
###   FIGURE 3C,D - percent time spent in each brain state surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

BS, mice, t, df, Trials = AS.laser_brainstate(ppath, recordings, pre=400, post=520, flatten_is=4, ma_state=3, 
                                              ma_thr=20, edge=0, sf=0, ci='sem')

#%%
###   FIGURE 3E - averaged SPs and frequency band power surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]
bands=[(0.5,4), (6,10), (11,15), (55,99)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']

EEGSpec, PwrBands, mice, t, df = AS.laser_triggered_eeg_avg(ppath, recordings, pre=400, post=520, fmax=100, 
                                                   laser_dur=120, pnorm=1, psmooth=3, harmcs=10, iplt_level=2,
                                                   vm=[0.6,1.4], sf=7, bands=bands, band_labels=band_labels, 
                                                   band_colors=band_colors, ci=95, ylim=[0.6,1.3])

#%%
###   FIGURE 3G - example closed loop opto recording   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'

AS.plot_example(ppath, 'Cinderella_022420n1', tstart=7100, tend=10100, PLOT=['LSR', 'SP', 'HYPNO'], fmax=25, vm=[0,1500],
                highres=False, pnorm=0, psmooth=[2,3], flatten_is=4, ma_thr=0)

#%%
###   FIGURE 3H - mean laser-on vs laser-off REM duration for ChR2 and eYFP mice   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'

# ChR2 mice
exp_recordings = sleepy.load_recordings(ppath, 'crh_chr2_cl.txt')[1]
df_exp = AS.state_online_analysis(ppath, exp_recordings, istate=1, single_mode=False, print_stats=False)
# eYFP (ChR2 protocol)
ctr_recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_cl.txt')[1]
df_ctr = AS.state_online_analysis(ppath, ctr_recordings, istate=1, single_mode=False, print_stats=False)

# bootstrap experimental and control mice
boot_chr2 = AS.bootstrap_online_analysis(df=df_exp, dv='dur', iv='lsr', virus='chr2', nboots=10000, alpha=0.05, 
                                         shuffle=True, seed=1)
boot_yfp = AS.bootstrap_online_analysis(df=df_ctr, dv='dur', iv='lsr', virus='yfp', nboots=10000, alpha=0.05, 
                                        shuffle=True, seed=1)
BOOT = pd.concat((boot_chr2, boot_yfp), axis=0, ignore_index=True)
AS.plot_bootstrap_online_analysis(BOOT, dv='dur', iv='lsr', mode=1, plotType='violin', ylim=[-80,80])
# opsin vs yfp
opsin_data, yfp_data, p = AS.compare_boot_stats(BOOT, mode=1, dv='dur', iv='lsr', virus=['chr2','yfp'], shuffled=[0,0],
                                                grp_names=['chr2', 'yfp'])
# opsin true vs opsin shuffled
opsin_data2, opsin_shuf_data, p = AS.compare_boot_stats(BOOT, mode=1, dv='dur', iv='lsr', virus=['chr2','chr2'], shuffled=[0,1],
                                                        grp_names=['chr2', 'chr2 shuffled'])

#%%
###   FIGURE 3I - mean laser-on vs laser-off REM duration for iC++ and eYFP mice   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'

# iC++ mice
exp_recordings = sleepy.load_recordings(ppath, 'crh_ic_cl.txt')[1]
df_exp = AS.state_online_analysis(ppath, exp_recordings, istate=1, single_mode=False, print_stats=False)
# eYFP (iC++ protocol)
ctr_recordings = sleepy.load_recordings(ppath, 'crh_yfp_ic_cl.txt')[1]
df_ctr = AS.state_online_analysis(ppath, ctr_recordings, istate=1, single_mode=False, print_stats=False)

# bootstrap experimental and control mice
boot_ic = AS.bootstrap_online_analysis(df=df_exp, dv='dur', iv='lsr', virus='ic++', nboots=10000, alpha=0.05, 
                                       shuffle=True, seed=1)
boot_yfp = AS.bootstrap_online_analysis(df=df_ctr, dv='dur', iv='lsr', virus='yfp', nboots=10000, alpha=0.05, 
                                        shuffle=True, seed=1)
BOOT = pd.concat((boot_ic, boot_yfp), axis=0, ignore_index=True)
AS.plot_bootstrap_online_analysis(BOOT, dv='dur', iv='lsr', mode=1, plotType='violin', ylim=[-80,80])
# opsin vs yfp
opsin_data, yfp_data, p = AS.compare_boot_stats(BOOT, mode=1, dv='dur', iv='lsr', virus=['ic++','yfp'], shuffled=[0,0],
                                                grp_names=['chr2', 'yfp'])
# opsin true vs opsin shuffled
opsin_data2, opsin_shuf_data, p = AS.compare_boot_stats(BOOT, mode=1, dv='dur', iv='lsr', virus=['ic++','ic++'], shuffled=[0,1],
                                                        grp_names=['ic++', 'ic++ shuffled'])

#%%
###   FIGURE 4B - example spontaneous & laser-triggered P-wave   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

AS.plot_example(ppath, 'Huey_101719n1', tstart=5925, tend=5930, PLOT=['LSR', 'EEG', 'LFP'], eeg_nbin=5, lfp_nbin=10)

#%%
###   FIGURE 4C,D,E - waveforms & spectral power surrounding P-waves/laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

# top - averaged waveforms surrounding P-waves & laser
filename = 'wf_win025'; wform_win = [0.25,0.25]; istate=[1]
pwaves.avg_waveform(ppath, recordings, istate, mode='pwaves', win=wform_win, mouse_avg='trials',  # spontaneous & laser-triggered P-waves
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])
pwaves.avg_waveform(ppath, recordings, istate, mode='lsr', win=wform_win, mouse_avg='trials',     # successful & failed laser
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])

# middle - averaged SPs surrounding P-waves & laser
filename = 'sp_win3'; win=[-3,3]; pnorm=2
pwaves.avg_SP(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1,  # spontaneous & laser-triggered P-waves
              mouse_avg='mouse', pnorm=pnorm, psmooth=[(8,8),(8,8)], vm=[(0.82,1.32),(0.8,1.45)], 
              fmax=25, recalc_highres=False, pload=filename, psave=filename)
pwaves.avg_SP(ppath, recordings, istate=[1], mode='lsr', win=win, plaser=True, post_stim=0.1,     # successful & failed laser
              mouse_avg='mouse', pnorm=pnorm, psmooth=[(8,8),(8,8)], vm=[(0.82,1.32),(0.6,1.8)], 
              fmax=25, recalc_highres=False, pload=filename, psave=filename)

# bottom - average high theta power surrounding P-waves & laser
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True,     # spontaneous & laser-triggered P-waves
                          post_stim=0.1, mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], 
                          pnorm=pnorm, psmooth=0, fmax=25, pload=filename, psave=filename, ylim=[0.5,1.5])
# successful and failed laser
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='lsr', win=win, plaser=True,        # successful & failed laser
                          post_stim=0.1, mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], 
                          pnorm=pnorm, psmooth=0, fmax=25, pload=filename, psave=filename, ylim=[0.5,1.5])

#%%
###   FIGURE 4F - spectral profiles: null vs spon vs success lsr vs fail lsr    ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'sp_win3'
spon_win=[-0.5, 0.5]; lsr_win=[0,1]; collect_win=[-3,3]; frange=[0, 20]; pnorm=2; null=True; null_win=0; null_match='lsr'

df, df2 = pwaves.sp_profiles(ppath, recordings, spon_win=spon_win, lsr_win=lsr_win, collect_win=collect_win, frange=frange, 
                             null=null, null_win=null_win, null_match=null_match, plaser=True, post_stim=0.1, pnorm=pnorm, 
                             psmooth=12, mouse_avg='mouse', ci='sem', pload=filename, psave=filename)

#%%
###   FIGURE 4G - probability of laser success per brainstate   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'lsr_stats'
df = pwaves.get_lsr_stats(ppath, recordings, istate=[1,2,3,4], lsr_jitter=5, post_stim=0.1, 
                          flatten_is=4, ma_thr=20, ma_state=3, psave=filename)
_ = pwaves.lsr_state_success(df, istate=[1,2,3,4])  # true laser success
_ = pwaves.lsr_state_success(df, istate=[1], jstate=[1])  # true vs sham laser success

#%%
###   FIGURE 4H - latencies of elicited P-waves to laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
df = pd.read_pickle('lsr_stats.pkl')

pwaves.lsr_pwave_latency(df, istate=1, jitter=True)

#%%
###   FIGURE 4I - phase preferences of spontaneous & laser-triggered P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'lsr_phases'

pwaves.lsr_hilbert(ppath, recordings, istate=1, bp_filt=[6,12], min_state_dur=30, stat='perc', mode='pwaves', 
                   mouse_avg='trials', bins=9, pload=filename, psave=filename)

#%%
###   FIGURE 5B,C - example recordings of hm3dq + saline vs cno   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'

AS.plot_example(ppath, 'Dahl_030321n1', tstart=3960, tend=5210, PLOT=['EEG', 'SP', 'HYPNO', 'EMG_AMP'], eeg_nbin=100,  # saline
                fmax=25, vm=[15,2200], psmooth=(1,2), flatten_is=4, ma_thr=0, ylims=[[-0.6,0.6],'','',[0,300]])
AS.plot_example(ppath, 'Dahl_031021n1', tstart=3620, tend=4870, PLOT=['EEG', 'SP', 'HYPNO', 'EMG_AMP'], eeg_nbin=100,  # CNO
                fmax=25, vm=[15,2200], psmooth=(1,2), flatten_is=4, ma_thr=0, ylims=[[-0.6,0.6],'','',[0,300]])

#%%
###   FIGURE 5D - hm3dq percent time spent in REM   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# hm3dq vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm3dq','hm3dq','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)
    
#%%
###   FIGURE 5E - hm3dq mean REM frequency   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
# hm3dq vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm3dq','hm3dq','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)
    
#%%
###   FIGURE 5F - hm3dq mean REM duration   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
# hm3dq vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm3dq','hm3dq','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)

#%%
###   FIGURE 5G - hm3dq IS->REM probability   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
# hm3dq vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm3dq','hm3dq','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)

#%%
###   FIGURE 5H - hm4di percent time spent in REM   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# hm4di vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm4di','hm4di','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)

#%%
###   FIGURE 5I - hm4di mean REM frequency   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='freq', flatten_is=4, exclude_noise=True, pplot=False)
# hm4di vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm4di','hm4di','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)

#%%
###   FIGURE 5J - hm4di mean REM duration   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='dur', flatten_is=4, exclude_noise=True, pplot=False)
# hm4di vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm4di','hm4di','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)

#%%
###   FIGURE 5K - hm4di IS->REM probability   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='is prob', flatten_is=4, exclude_noise=True, pplot=False)
# hm4di vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm4di','hm4di','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)

#%%
###   FIGURE 6A - example P-waves during NREM-->IS-->REM transitions   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
AS.plot_example(ppath, 'King_071020n1', ['HYPNO', 'EEG', 'LFP'], tstart=16097, tend=16172, ylims=['',(-0.6, 0.6), (-0.3, 0.15)])  # saline
AS.plot_example(ppath, 'King_071520n1', ['HYPNO', 'EEG', 'LFP'], tstart=5600, tend=5675, ylims=['',(-0.6, 0.6), (-0.3, 0.15)])  # CNO

#%%
###   FIGURE 6B - hm3dq and mCherry P-wave frequency during state transitions   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True)
ctr_recordings_cno = ctr_recordings_cno['0.25']
sequence=[3,4,1,2]
state_thres=[0]*len(sequence)
sign=['>']*len(sequence)
nstates=[20]*len(sequence)

# hm3dq mice
exp_mice,exp_cmx,exp_cspe = pwaves.stateseq(ppath, exp_recordings_sal, sequence=sequence, nstates=nstates, sign=sign,  # saline
                                            state_thres=state_thres, fmax=25, pnorm=1, psmooth=[2,2], sf=4, 
                                            mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
exp_mice,exp_emx,exp_espe = pwaves.stateseq(ppath, exp_recordings_cno, sequence=sequence, nstates=nstates, sign=sign,  # CNO
                                            state_thres=state_thres, fmax=25, pnorm=1, psmooth=[2,2], sf=4, 
                                            mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
# mCherry mice
ctr_mice,ctr_emx,ctr_espe = pwaves.stateseq(ppath, ctr_recordings_cno, sequence=sequence, nstates=nstates, sign=sign,  # CNO
                                            state_thres=state_thres, fmax=25, pnorm=1, psmooth=[2,2], sf=4, 
                                            mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
# plot timecourses
pwaves.plot_activity_transitions([exp_cmx, exp_emx, ctr_emx], [exp_mice, exp_mice, ctr_mice], plot_id=['gray', 'blue', 'lightblue'], 
                                 group_labels=['hm3dq saline', 'hm3dq cno', 'mcherry cno'], xlim=nstates, xlabel='Time (normalized)', 
                                 ylabel='P-waves/s', title='NREM-->tNREM-->REM-->Wake')

#%%
###   FIGURE 6C - hm3dq P-wave frequency   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice (hm3dq protocol)
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
# hm3dq vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm3dq','hm3dq','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)
### BOOTSTRAPPED DATA (see Fig. 3h-i for bootstrapping code)
boot_file = os.path.join(ppath, 'hm3dq_boot.pkl')
boot_df = pd.read_pickle(boot_file)
exp_dif = np.subtract(np.array(boot_df.loc[np.where((boot_df.dose=='cno') & (boot_df.virus=='hm3dq'))[0], 'freq']),
                      np.array(boot_df.loc[np.where((boot_df.dose=='saline') & (boot_df.virus=='hm3dq'))[0], 'freq']))
ctr_dif = np.subtract(np.array(boot_df.loc[np.where((boot_df.dose=='cno') & (boot_df.virus=='mCherry'))[0], 'freq']),
                      np.array(boot_df.loc[np.where((boot_df.dose=='saline') & (boot_df.virus=='mCherry'))[0], 'freq']))
boot_ = pd.DataFrame({'virus' : ['hm3dq']*len(exp_dif) + ['mCherry']*len(ctr_dif),
                      'dif. p-wave freq' : np.concatenate([exp_dif, ctr_dif])})

#%%
###   FIGURE 6D - hm4di and mCherry P-wave frequency during state transitions   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=True)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True)
ctr_recordings_cno = ctr_recordings_cno['5']
sequence=[3,4,1,2]
state_thres=[0]*len(sequence)
sign=['>']*len(sequence)
nstates=[20]*len(sequence)

# hm4di mice
exp_mice,exp_cmx,exp_cspe = pwaves.stateseq(ppath, exp_recordings_sal, sequence=sequence, nstates=nstates, sign=sign,  # saline
                                            state_thres=state_thres, fmax=25, pnorm=1, psmooth=[2,2], sf=4, 
                                            mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
exp_mice,exp_emx,exp_espe = pwaves.stateseq(ppath, exp_recordings_cno, sequence=sequence, nstates=nstates, sign=sign,  # CNO
                                            state_thres=state_thres, fmax=25, pnorm=1, psmooth=[2,2], sf=4, 
                                            mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
# mCherry mice
ctr_mice,ctr_emx,ctr_espe = pwaves.stateseq(ppath, ctr_recordings_cno, sequence=sequence, nstates=nstates, sign=sign,  # CNO
                                            state_thres=state_thres, fmax=25, pnorm=1, psmooth=[2,2], sf=4, 
                                            mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
# plot timecourses
pwaves.plot_activity_transitions([exp_cmx, exp_emx, ctr_emx], [exp_mice, exp_mice, ctr_mice], plot_id=['gray', 'red', 'pink'], 
                                 group_labels=['hm4di saline', 'hm4di cno', 'mcherry cno'], xlim=nstates, xlabel='Time (normalized)', 
                                 ylabel='P-waves/s', title='NREM-->tNREM-->REM-->Wake')

#%%
###   FIGURE 6E - hm4di P-wave frequency   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=True)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice (hm4di protocol)
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[1], tbin=18000, n=1, ma_state=3,
                                           stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
# hm4di vs mCherry
df = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                    mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                    dose_list=['saline','cno','saline','cno'],
                                    virus_list=['hm4di','hm4di','mCherry','mCherry'])
# stats
res_anova = ping.mixed_anova(data=df, dv='t0', within='dose', subject='mouse', between='virus')
mc1 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=False)
mc2 = ping.pairwise_tests(data=df, dv='t0', within='dose', subject='mouse', between='virus', 
                          padjust='holm', within_first=True)
### BOOTSTRAPPED DATA (see Fig. 3h-i for bootstrapping code)
boot_file = os.path.join(ppath, 'hm4di_boot.pkl')
boot_df = pd.read_pickle(boot_file)
exp_dif = np.subtract(np.array(boot_df.loc[np.where((boot_df.dose=='cno') & (boot_df.virus=='hm4di'))[0], 'freq']),
                      np.array(boot_df.loc[np.where((boot_df.dose=='saline') & (boot_df.virus=='hm4di'))[0], 'freq']))
ctr_dif = np.subtract(np.array(boot_df.loc[np.where((boot_df.dose=='cno') & (boot_df.virus=='mCherry'))[0], 'freq']),
                      np.array(boot_df.loc[np.where((boot_df.dose=='saline') & (boot_df.virus=='mCherry'))[0], 'freq']))
boot_ = pd.DataFrame({'virus' : ['hm4di']*len(exp_dif) + ['mCherry']*len(ctr_dif),
                      'dif. p-wave freq' : np.concatenate([exp_dif, ctr_dif])})