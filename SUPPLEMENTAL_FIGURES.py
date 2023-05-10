#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:30:46 2021

@author: fearthekraken
"""
import AS
import pwaves
import sleepy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison

#%%
###   Supp. FIGURE 1D - FISH quantification   ###
df = pd.read_csv('/home/fearthekraken/Documents/Data/sleepRec_processed/FISH_counts.csv')
plt.figure()
sns.boxplot(x='MARKER LABEL', y='%CRH + MARKER', order=['VGLUT1','VGLUT2','GAD2'], data=df, whis=np.inf, color='white', fliersize=0)
sns.stripplot(x='MARKER LABEL', y='%CRH + MARKER', hue='Mouse', order=['VGLUT1','VGLUT2','GAD2'], data=df, 
              palette={'Marlin':'lightgreen', 'SERT1':'lightblue', 'Nemo':'lightgray'}, size=10, linewidth=1, edgecolor='black')
plt.show()
print('')
for marker_label in ['VGLUT1', 'VGLUT2', 'GAD2']:
    p = df['%CRH + MARKER'].iloc[np.where(df['MARKER LABEL']==marker_label)[0]]
    print(f'{round(p.mean(),2)}% of CRH+ neurons co-express {marker_label} (+/-{round(p.std(),2)}%)')

#%%
###   Supp. FIGURE 2B - time-normalized DF/F activity across brain state transitions   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]
sequence=[3,4,1,2]; nstates=[20,20,20,20]; vm=[0.2, 1.9]  # NREM --> IS --> REM --> WAKE
state_thres=[0]*len(sequence); sign=['>','>','>','>']
#state_thres=[(0,10000)]*len(sequence); 

_, mx_pwave, _ = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, sign=sign, ma_thr=20, ma_state=3, 
                                       flatten_is=4, fmax=25, pnorm=1, vm=vm, psmooth=[2,2], mode='dff', mouse_avg='mouse', print_stats=False)

#%%
###   Supp. FIGURE 2C,D,E - DF/F activity at brain state transitions   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]
#transitions = [(3,4)]; pre=40; post=15; vm=[0.3, 1.9]; tr_label = 'NtN'  # NREM --> IS (102 transitions/58 pre-REM transitions)
#transitions = [(4,1)]; pre=15; post=40; vm=[0.1, 2.1]; tr_label = 'tNR'  # IS --> REM (43 transitions)
transitions = [(1,2)]; pre=40; post=15; vm=[0.1, 2.1]; tr_label = 'RW'   # REM --> WAKE (52 transitions)

si_threshold = [pre]*6; sj_threshold = [post]*6
mice, tr_act, tr_spe = pwaves.activity_transitions(ppath, recordings, transitions=transitions, pre=pre, post=post, si_threshold=si_threshold, 
                                                   sj_threshold=sj_threshold, ma_thr=20, ma_state=3, flatten_is=False, vm=vm, fmax=25, pnorm=1, 
                                                   psmooth=[3,3], mode='dff', mouse_avg='trials', base_int=5, print_stats=True)

#%%
###   Supp. FIGURE 2F - DF/F activity following single & cluster P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]

# get DF/F timecourse data, store in dataframe
pzscore=0; p_iso=0.8; pcluster=0
mice, iso_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='', dff_win=[0,2], pzscore=pzscore, mouse_avg='mouse',  # single P-waves
                                     p_iso=p_iso, pcluster=pcluster, clus_event='waves', psmooth=(8,15), print_stats=False)
pzscore=0; p_iso=0; pcluster=0.5
mice, clus_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='', dff_win=[0,2], pzscore=pzscore, mouse_avg='mouse',  # clustered P-waves
                                     p_iso=p_iso, pcluster=pcluster, clus_event='waves', psmooth=(8,15), print_stats=False)
df = pd.DataFrame({'Mouse' : np.tile(mice,2),
                   'Event' : np.repeat(['single', 'cluster'], len(mice)),
                   'DFF' : np.concatenate((iso_mx[2].mean(axis=1), clus_mx[2].mean(axis=1))) })

# bar plot
plt.figure(); sns.barplot(x='Event', y='DFF', data=df, ci=68, palette={'single':'salmon', 'cluster':'mediumslateblue'})
sns.pointplot(x='Event', y='DFF', hue='Mouse', data=df, ci=None, markers='', color='black'); plt.gca().get_legend().remove(); plt.show()

# stats
p = stats.ttest_rel(df['DFF'].iloc[np.where(df['Event'] == 'single')[0]], df['DFF'].iloc[np.where(df['Event'] == 'cluster')[0]])
print(f'single vs cluster P-waves -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')

#%%
###   Supp. FIGURE 2G -averaged DF/F surrounding P-waves in each brain state  ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]
for s in [1,2,3,4]:
    pwaves.dff_timecourse(ppath, recordings, istate=s, dff_win=[2,2], plotMode='03', pzscore=0, mouse_avg='mouse', 
                          ma_thr=20, ma_state=3, flatten_tnrem=4, p_iso=0, pcluster=0)

#%%
###   Supp. FIGURE 2H - DFF vs P-waves cross-correlation   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]
win = 1

(CC, mice), (CC_jtr, mice) = pwaves.dff_pwaves_corr(ppath, recordings, win=win, istate=1, dffnorm=True, 
                                                    ptrain=True, ptrial=False, dn=15, sf=30, min_dur=45, 
                                                    ma_thr=20, ma_state=3, flatten_is=4, mouse_avg='mouse', 
                                                    jitter=True, jtr_win=15, seed=15, base_int=0.2, 
                                                    baseline_start=0, baseline_end=-1, pplot=True, print_stats=True)

#%%
###   Supp. FIGURE 2I - DF/F signal surrounding shuffled P-waves (20 s window)   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]

jitter_trig = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[-10,10], 
                                    pzscore=2, mouse_avg='mouse', dn=250, sf=1000, jitter=15, base_int=2.5, 
                                    flatten_is=4, ma_state=6, tstart=120, ylim=[-0.3,0.65])

#%%
###   Supp. FIGURE 2J - isosbestic signal surrounding P-waves (20 s window)   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]

isosbestic_trig = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[-10,10], 
                                        pzscore=2, mouse_avg='mouse', dn=250, sf=1000, use405=True, base_int=2.5, 
                                        flatten_is=4, ma_state=6, tstart=120, ylim=[-0.3,0.65])

#%%
###   Supp. FIGURE 3B,C,D - P-wave frequency at brain state transitions   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
#transitions = [(3,4)]; pre=40; post=15; vm=[0.4,2.0]; tr_label = 'NtN'  # NREM --> IS
#transitions = [(4,1)]; pre=15; post=40; vm=[0.1, 2.0]; tr_label = 'tNR'  # IS --> REM
transitions = [(1,2)]; pre=40; post=15; vm=[0.1, 2.0]; tr_label = 'RW'   # REM --> WAKE

si_threshold = [pre]*6; sj_threshold = [post]*6
mice, tr_act, tr_spe = pwaves.activity_transitions(ppath, recordings, transitions=transitions, pre=pre, post=post, si_threshold=si_threshold, 
                                                   sj_threshold=sj_threshold, ma_thr=20, ma_state=3, flatten_is=False, vm=vm, fmax=25, pnorm=1, 
                                                   psmooth=[3,3], mode='pwaves', mouse_avg='trials', base_int=5, print_stats=True)

#%%
###   Supp. FIGURE 3E - time-normalized frequency of single & clustered P-waves across brain state transitions   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; vm=[0.2,2.0]  # NREM --> IS --> REM --> WAKE

mice,smx,sspe = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25,   # single P-waves
                                 pnorm=1, vm=vm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', p_iso=0.8, pcluster=0, 
                                 clus_event='waves', pplot=False, print_stats=False)
mice,cmx,cspe = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25,   # clustered P-waves
                                 pnorm=1, vm=vm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', p_iso=0, pcluster=0.5, 
                                 clus_event='waves', pplot=False, print_stats=False)
# plot timecourses
pwaves.plot_activity_transitions([smx, cmx], [mice, mice], plot_id=['salmon', 'mediumslateblue'], group_labels=['single', 'cluster'], 
                                 xlim=nstates, xlabel='Time (normalized)', ylabel='P-waves/s', title='NREM-->tNREM-->REM-->Wake')


#%%
###   Supp. FIGURE 3F - average single & cluster P-wave frequency in each brain state   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]

single_df = pwaves.state_freq(ppath, recordings, istate=[1,2,3,4], plotMode='03', ma_thr=20, 
                              flatten_is=4, ma_state=3, return_mode='df', p_iso=0.8, print_stats=False)
single_df['event'] = 'single p-waves'
clus_df = pwaves.state_freq(ppath, recordings, istate=[1,2,3,4], plotMode='03', ma_thr=20, 
                            flatten_is=4, ma_state=3, return_mode='df', pcluster=0.5, clus_event='cluster_start', print_stats=False)
clus_df['event'] = 'clustered p-waves'
df = pd.concat([single_df, clus_df], axis=0, ignore_index=True)
stateMap = {1:'REM', 2:'wake', 3:'NREM', 4:'IS'}
df.replace({'state' : stateMap}, inplace=True)
df.rename(columns={'freq':'p-wave freq'}, inplace=True)
df = df.loc[:, ['mouse','state','event','p-wave freq']]

# stats
res_anova = ping.rm_anova(data=df, dv='p-wave freq', within=['state','event'], subject='mouse')
res_tt = ping.pairwise_tests(data=df, dv='p-wave freq', within=['state','event'], subject='mouse', padjust='bonf')
print_anova(res_anova, res_tt)

# bar plot
fig, axs = plt.subplots(nrows=1, ncols=len(stateMap.values()))
pal = {'single p-waves' : 'lightcoral', 'clustered p-waves' : 'slateblue'}
for i,s in enumerate(stateMap.values()):
    ax = axs[i]
    ddf = df[df.state == s]
    sns.barplot(x='event', y='p-wave freq', data=ddf, errorbar='se', width=1, palette=pal, ax=ax)
    lines = sns.lineplot(x='event', y='p-wave freq', hue='mouse', legend=False, data=ddf, ax=ax)
    _ = [l.set_color('black') for l in lines.get_lines()]
    ax.set_ylim([0,0.35])
    sleepy.box_off(ax)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)

#%%
###   Supp. FIGURE 3G - average spectral power surrounding single & cluster P-waves   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]

# top - averaged spectrograms
filename = 'sp_win3_single'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
pwaves.avg_SP(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm, psmooth=[2,2],  # single P-waves
              fmax=25, vm=[0.6,2.0], p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)
filename = 'sp_win3_cluster'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
pwaves.avg_SP(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm, psmooth=[2,2],  # clustered P-waves
              fmax=25, vm=[0.6,2.0], p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)

# bottom - average high theta power
filename = 'sp_win3_single'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
mice, sdict, t = pwaves.avg_band_power(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm,  # single P-waves
                                       psmooth=0, bands=[(8,15)], band_colors=['green'], p_iso=p_iso, pcluster=pcluster, 
                                       clus_event='waves', ylim=[0.6,1.8], pload=filename, psave=filename)
filename = 'sp_win3_cluster'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
mice, cdict, t = pwaves.avg_band_power(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm,  # clustered P-waves
                                       psmooth=0, bands=[(8,15)], band_colors=['green'], p_iso=p_iso, pcluster=pcluster, 
                                       clus_event='waves', ylim=[0.6,1.8], pload=filename, psave=filename)

# right - mean power in 1 s time window
x = np.intersect1d(np.where(t>=-0.5)[0], np.where(t<=0.5)[0])  # get columns between -0.5 s and +0.5 s
df = pd.DataFrame({'Mouse' : np.tile(mice,2),
                   'Event' : np.repeat(['single', 'cluster'], len(mice)),
                   'Pwr'   : np.concatenate((sdict[(8,15)][:,x].mean(axis=1), cdict[(8,15)][:,x].mean(axis=1))) })
fig = plt.figure(); sns.barplot(x='Event', y='Pwr', data=df, ci=68, palette={'single':'salmon', 'cluster':'mediumslateblue'})
sns.pointplot(x='Event', y='Pwr', hue='Mouse', data=df, ci=None, markers='', color='black'); plt.gca().get_legend().remove()
plt.title('Single vs Clustered P-waves'); plt.show()

# stats
p = stats.ttest_rel(df['Pwr'].iloc[np.where(df['Event'] == 'single')[0]], df['Pwr'].iloc[np.where(df['Event'] == 'cluster')[0]])
print(f'single vs cluster P-waves -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')

#%%
###   Supp. FIGURE 3H - mean raw EEG spectrogram surrounding single vs cluster P-waves   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
#filename = 'sp_win3_pnorm0_single'; win=[-3,3]; pnorm=0; psmooth=[]; vm=[[0,1600]]  # single
filename = 'sp_win3_pnorm0_cluster'; win=[-3,3]; pnorm=0; psmooth=[]; vm=[[0,1500]]  # cluster

pwaves.avg_SP(ppath, [], istate=[1], win=win, plaser=False, mouse_avg='mouse',
              nsr_seg=2, perc_overlap=0.95, fmax=15, recalc_highres=False,
              pnorm=pnorm, psmooth=psmooth, vm=vm, pload=filename, psave=filename)

#%%
###   Supp. FIGURE 3I - power spectrum surrounding single vs cluster vs no P-waves   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]

# single P-waves vs cluster P-waves vs no P-waves (sleep spectrums)
df1 = sleepy.sleep_spectrum_pwaves(ppath, recordings, win_inc=1, win_exc=1, istate=1, 
                                   pnorm=False, nsr_seg=2, perc_overlap=0.95,
                                   recalc_highres=False, fmax=15, exclude_noise=True,
                                   p_iso=0.8, pcluster=0.5, ma_state=3, ma_thr=20)
df1_theta = df1.loc[np.where((df1.freq >= 8) & (df1.freq <= 15))[0], ['mouse', 'pow', 'pwave']]
df1_theta = df1_theta.groupby(['mouse','pwave']).sum().reset_index()
df1_theta = AS.sort_df(df1_theta, 'pwave', ['no','single','cluster'], id_sort='mouse')
# stats
res_anova = ping.rm_anova(data=df1_theta, dv='pow', within='pwave', subject='mouse')
res_tt = ping.pairwise_tests(data=df1_theta, dv='pow', within='pwave', subject='mouse', padjust='holm')

# bar plot
pal = {'single':'dodgerblue', 'cluster':'darkblue', 'no':'gray'}
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, gridspec_kw={'width_ratios':[3,1]})
sns.lineplot(x='freq', y='pow', hue='pwave', data=df1, errorbar=None, palette=pal, ax=ax1)
_ = sns.barplot(x='pwave', y='pow', data=df1_theta, errorbar='se', 
                order=['no','single','cluster'], palette=pal, ax=ax2)
lines = sns.lineplot(x='pwave', y='pow', hue='mouse', data=df1_theta,
                     errorbar=None, markersize=0, legend=False, ax=ax2)
_ = [l.set_color('black') for l in lines.get_lines()]
ax2.set_title('8-15 Hz')

#%%
###   Supp. FIGURE 3J - power spectrum surrounding P-waves vs no P-waves during wake   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]

# P-waves vs no P-waves
df2 = sleepy.sleep_spectrum_pwaves(ppath, recordings, win_inc=1, win_exc=1, istate=2, 
                                   pnorm=False, nsr_seg=2, perc_overlap=0.95, 
                                   recalc_highres=False, fmax=15, exclude_noise=True,
                                   p_iso=0, pcluster=0, ma_state=3, ma_thr=20)
df2_theta = df2.loc[np.where((df2.freq >= 8) & (df2.freq <= 15))[0], ['mouse', 'pow', 'pwave']]
df2_theta = df2_theta.groupby(['mouse','pwave']).sum().reset_index()
df2_theta = AS.sort_df(df2_theta, 'pwave', ['no','yes'], id_sort='mouse')
# stats
mice = [m for i,m in enumerate(df2_theta.mouse) if list(df2_theta.mouse).index(m) == i]
a,b = zip(*[list(df2_theta[df2_theta.mouse==m]['pow']) for m in mice])
p = scipy.stats.ttest_rel(a, b)
print(f'\nT({len(mice)-1})={np.round(p.statistic,2)}, p={pp(p.pvalue)[0]}\n')

# bar plot
pal = {'yes':'blue', 'no':'gray'}
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, gridspec_kw={'width_ratios':[3,1]})
sns.lineplot(x='freq', y='pow', hue='pwave', data=df2, errorbar=None, palette=pal, ax=ax1)
_ = sns.barplot(x='pwave', y='pow', data=df2_theta, errorbar='se', 
                order=['no','yes'], palette=pal, ax=ax2)
lines = sns.lineplot(x='pwave', y='pow', hue='mouse', data=df2_theta,
                     errorbar=None, markersize=0, legend=False, ax=ax2)
_ = [l.set_color('black') for l in lines.get_lines()]
ax2.set_title('8-15 Hz')

#%%
###   Supp. FIGURE 4B,C - FISH c-Fos optogenetics quantification   ###
fos_df = pd.read_csv('/home/fearthekraken/Documents/Data/FISH_cfos_counts.csv')
fos_df = fos_df.iloc[np.where(fos_df['virus'] == 'chr2')[0], :].reset_index(drop=True)

### B
fos_df['# crh+/cfos- cells'] = fos_df['# crh cells'] - fos_df['# co-expressing cells']
fos_df['# crh-/cfos+ cells'] = fos_df['# cfos cells'] - fos_df['# co-expressing cells']
fos_df['% crh cells expressing cfos'] = (fos_df['# co-expressing cells'] / fos_df['# crh cells']) * 100
# bar plot
bar_df = fos_df.groupby(['mouse']).sum().reset_index().loc[:,['mouse',
                                                              '# co-expressing cells',
                                                              '# crh-/cfos+ cells',
                                                              '# crh+/cfos- cells']]
bar_df.plot(kind='bar', stacked=True, color=['black','gray','lightgray'])
# pie chart
n_fos, n_nofos = fos_df.loc[:,['# co-expressing cells','# crh+/cfos- cells']].sum(axis=0)
plt.figure(figsize=(2,2))
ax = plt.gca()
_ = ax.pie([n_fos, n_nofos], explode=[0,1],
            labels=['cfos+','cfos-'], 
            colors=['cornflowerblue','gray'], autopct='%1.1f%%', labeldistance=1.2, pctdistance=0.7,
            startangle=55, radius=2.5, wedgeprops={'linewidth': 2, 'edgecolor':'black'})

### C
fos_df2 = fos_df.groupby(['mouse','hemisphere']).mean().reset_index().loc[:,['mouse','hemisphere','% crh cells expressing cfos']]
plt.figure(); ax2 = plt.gca()
sns.barplot(x='hemisphere', y='% crh cells expressing cfos', data=fos_df2, errorbar='se', 
            palette={'contra':'cornflowerblue','ipsi':'lightblue'}, ax=ax2)
lines = sns.lineplot(x='hemisphere', y='% crh cells expressing cfos', hue='mouse', data=fos_df2, 
                     legend=False, ax=ax2)
_ = [l.set_color('black') for l in lines.get_lines()]

#%%
###  Supp. FIGURE 4D - % time in each brain state before and during the laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

BS, mice, t, df, Trials = AS.laser_brainstate(ppath, recordings, pre=400, post=520, flatten_is=4, ma_state=3, 
                                              ma_thr=20, edge=0, sf=0, ci='sem')

#%%
###   Supp. FIGURE 4E - averaged spectral band power before and during the laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

bands=[(0.5,4), (6,10), (11,15), (55,99)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']
AS.laser_triggered_eeg_avg(ppath, recordings, pre=400, post=520, fmax=100, laser_dur=120, pnorm=1, psmooth=3, harmcs=10, 
                           iplt_level=2, vm=[0.6,1.4], sf=7, bands=bands, band_labels=band_labels, band_colors=band_colors, ci=95)

#%%
###   Supp. FIGURE 4F - laser-triggered change in REM transition probability   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

_ = AS.laser_transition_probability(ppath, recordings, pre=400, post=520, ma_state=3, ma_thr=20, sf=10)

#%%
###   Supp. FIGURE 4G, spectral power during NREM-->REM transitions   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]
pre=40; post=40; si_threshold=[pre]*6; sj_threshold=[post]*6
bands=[(0.5,4), (6,10), (11,15), (55,99)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']

# average spectrograms/band power timecourses
_ = AS.avg_sp_transitions(ppath, recordings, transitions=[(3,1)], pre=pre, post=post, si_threshold=si_threshold, sj_threshold=sj_threshold, 
                          laser=1, bands=bands, band_labels=band_labels, band_colors=band_colors, flatten_tnrem=3, ma_thr=20, ma_state=3, 
                          fmax=100, pnorm=1, psmooth=[3,3], vm=[(0.1,2.5),(0.1,2.5)], mouse_avg='mouse', sf=0)
# mean power per band
mx, freq, df, _ = sleepy.sleep_spectrum_simple(ppath, recordings, istate=1, pmode=1, harmcs=5, fmax=500, ci=95, 
                                               pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True,
                                               exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
df2 = pd.DataFrame()
for band in bands:
    bdf = df.iloc[np.where((df.freq>=band[0]) & (df.freq<=band[1]))[0], :].groupby(['mouse','lsr']).sum().reset_index()
    bdf['pow'] = bdf['pow'] * 0.5
    
    df2 = pd.concat([df2, pd.DataFrame({'mouse' : np.array(bdf['mouse']),
                                        'band':f'{band[0]} - {band[1]} Hz',
                                        'lsr':np.array(bdf['lsr']),
                                        'pwr':np.array(bdf['pow'])})], axis=0, ignore_index=True)
df2.replace({'lsr' : {'no':0, 'yes':1}}, inplace=True)
# bar plot
fig, axs = plt.subplots(nrows=1, ncols=len(stateMap.values()))
pal = {0 : 'gray', 1 : 'blue'}
for i,b in enumerate([b for i,b in enumerate(df2.band) if list(df2.band).index(b) == i]):
    ax = axs[i]
    ddf = df2[df2.band == b]
    sns.barplot(x='lsr', y='pwr', data=ddf, errorbar='se', width=1, palette=pal, ax=ax)
    lines = sns.lineplot(x='lsr', y='pwr', hue='mouse', legend=False, data=ddf, ax=ax)
    _ = [l.set_color('black') for l in lines.get_lines()]
    ax.set_ylim([0,5000])
    sleepy.box_off(ax)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
# stats
res_anova = ping.rm_anova(data=df2, dv='pwr', within=['lsr','band'], subject='mouse')

#%%
###   Supp. FIGURE 4I,J - eYFP percent time spent in each brain state surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_ol.txt')[1]

BS, mice, t, df, Trials = AS.laser_brainstate(ppath, recordings, pre=400, post=520, flatten_is=4, ma_state=3, 
                                              ma_thr=20, edge=0, sf=0, ci='sem')

#%%
###   Supp. FIGURE 4K - eYFP averaged SPs and frequency band power surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_ol.txt')[1]
bands=[(0.5,4), (6,10), (11,15), (55,99)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']

EEGSpec, PwrBands, mice, t, df = AS.laser_triggered_eeg_avg(ppath, recordings, pre=400, post=520, fmax=100, 
                                                   laser_dur=120, pnorm=1, psmooth=3, harmcs=10, iplt_level=2,
                                                   vm=[0.6,1.4], sf=7, bands=bands, band_labels=band_labels, 
                                                   band_colors=band_colors, ci=95, ylim=[0.6,1.3])

#%%
###   Supp. FIGURE 4L - closed loop overall REM stats   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'

# ChR2 experiment
chr2_recordings = sleepy.load_recordings(ppath, 'crh_chr2_cl.txt')[1]
chr2_yfp_recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_cl.txt')[1]
_ = AS.compare_online_analysis(ppath, chr2_yfp_recordings, chr2_recordings, istate=1, stat='perc') # percent time in REM
_ = AS.compare_online_analysis(ppath, chr2_yfp_recordings, chr2_recordings, istate=1, stat='dur')  # overall REM duration

# iC++ experiment
ic_recordings = sleepy.load_recordings(ppath, 'crh_ic_cl.txt')[1]
ic_yfp_recordings = sleepy.load_recordings(ppath, 'crh_yfp_ic_cl.txt')[1]
_ = AS.compare_online_analysis(ppath, ic_yfp_recordings, ic_recordings, istate=1, stat='perc') # percent  time in REM
_ = AS.compare_online_analysis(ppath, ic_yfp_recordings, ic_recordings, istate=1, stat='dur')  # overall REM duration

#%%
###   Supp. FIGURE 4M - ChR2 vs eYFP sleep spectrum   ###
bands=[(0.5,4), (6,10), (11,15), (15.5,20)]

# ChR2 mice
exp_path = '/media/fearthekraken/Mandy_HardDrive1/ChR2_Open'
exp_recordings = sleepy.load_recordings('/home/fearthekraken/Documents/Data/sleepRec_processed/', 'crh_chr2_ol.txt')[1]
exp_df = pd.DataFrame()
print('### ChR2 mice')
for s in [1,2,3,4]:
    print(f'....... state = {s}')
    _, _, ddf, _ = sleepy.sleep_spectrum_simple(exp_path, exp_recordings, istate=s, pmode=1, harmcs=5, fmax=20, mu=[5,100], 
                                                ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf['state'] = s
    exp_df = pd.concat([exp_df, ddf], axis=0, ignore_index=True)
exp_df['virus'] = 'chr2'

# eYFP mice
ctr_path = '/media/fearthekraken/Mandy_HardDrive1/ChR2_YFP_Open'
ctr_recordings = sleepy.load_recordings('/home/fearthekraken/Documents/Data/sleepRec_processed/', 'crh_yfp_chr2_ol.txt')[1]
ctr_df = pd.DataFrame()
print('### eYFP mice')
for s in [1,2,3,4]:
    print(f'....... state = {s}')
    _, _, ddf, _ = sleepy.sleep_spectrum_simple(ctr_path, ctr_recordings, istate=s, pmode=1, harmcs=5, fmax=20, mu=[5,100], 
                                                ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf['state'] = s
    ctr_df = pd.concat([ctr_df, ddf], axis=0, ignore_index=True)
ctr_df['virus'] = 'yfp'
df = pd.concat([exp_df, ctr_df], axis=0, ignore_index=True)
stateMap = {1:'REM', 2:'wake', 3:'NREM', 4:'IS'}
df.rename(columns={'pow':'pwr'}, inplace=True)
df.replace({'lsr' : {'no':0, 'yes':1}, 'state' : stateMap}, inplace=True)
df = df.loc[:, ['mouse','virus','state','lsr','freq','pwr']]

# calculate laser-on - laser-off power 
df2 = pd.DataFrame()
df3 = pd.DataFrame()
for band in bands:
    A = df.iloc[np.where((df.freq>=band[0]) & (df.freq<=band[1]))[0], :].groupby(['virus','mouse','lsr','state']).sum().reset_index()
    A['pwr'] = A['pwr'] * 0.5
    A['band'] = f'{band[0]} - {band[1]} Hz'
    df2 = pd.concat([df2, A], axis=0, ignore_index=True)
    # get laser-on minus laser-off power
    difs = np.array(A[A.lsr==1]['pwr']) - np.array(A[A.lsr==0]['pwr'])
    B = A[A.lsr==1].copy()
    B['dif. pwr'] = difs
    df3 = pd.concat([df3, B], axis=0, ignore_index=True)
df2 = df2.loc[:, ['mouse','band','virus','state','lsr','pwr']]
df3 = df3.loc[:, ['mouse','band','virus','state','dif. pwr']]

# stats
df2_chr2 = df2[df2.virus=='chr2'].reset_index(drop=True)
df2_yfp = df2[df2.virus=='yfp'].reset_index(drop=True)

for s in stateMap.values():
    print('='*len(s) + '\n' + s + '\n' + '='*len(s))
    print('\n###   ChR2 mice   ###\n')
    ddf_chr2 = df2_chr2[df2_chr2.state==s].reset_index(drop=True)
    res_chr2 = ping.rm_anova(data=ddf_chr2, subject='mouse', dv='pwr', within=['band','lsr'])
    print(res_chr2.to_string())
    
    print('\n###   eYFP mice   ###\n')
    ddf_yfp = df2_yfp[df2_yfp.state==s].reset_index(drop=True)
    res_yfp = ping.rm_anova(data=ddf_yfp, subject='mouse', dv='pwr', within=['band','lsr'])
    print(res_yfp.to_string())
    
    print('\n###   ChR2 vs eYFP mice   ###\n')
    ddf_dif = df3[df3.state==s].reset_index(drop=True)
    res_dif = ping.mixed_anova(data=ddf_dif, subject='mouse', dv='dif. pwr', within='band', between='virus')
    print(res_dif.to_string())
    print('\n__________________________________________\n')
    
#%%
###   Supp. FIGURE 5B - average amplitude and half-width of spontaneous & laser-triggered P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'lsr_stats' 
df = pwaves.get_lsr_stats(ppath, recordings, istate=[1,2,3,4], post_stim=0.1, flatten_tnrem=4, ma_thr=20, ma_state=3, psave=filename)

pwaves.lsr_pwave_size(df, stat='amp2', plotMode='03', istate=1, mouse_avg='mouse')
pwaves.lsr_pwave_size(df, stat='halfwidth', plotMode='03', istate=1, mouse_avg='mouse')

#%%
###   Supp. FIGURE 5C - average spectral power surrounding single & cluster P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

# top - averaged spectrograms
filename = 'sp_win3_single_lsr'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
pwaves.avg_SP(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1, mouse_avg='mouse',  # single lsr P-waves
              pnorm=pnorm, psmooth=[(3,3),(5,5)], vm=[(0.6,1.65),(0.8,1.5)], fmax=25, recalc_highres=False, 
              p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)
filename = 'sp_win3_cluster_lsr'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
pwaves.avg_SP(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1, mouse_avg='mouse',  # clustered lsr P-waves
              pnorm=pnorm, psmooth=[(7,7),(5,5)], vm=[(0.6,1.65),(0.8,1.5)], fmax=25, recalc_highres=False, 
              p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)

# bottom - averaged high theta power
filename = 'sp_win3_single_lsr'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
mice,lsr_iso,spon_iso,t = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1,  # single ls P-waves
                                        mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], pnorm=pnorm, psmooth=(4,4), 
                                        fmax=25, p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename, ylim=[0.5,2])
filename = 'sp_win3_cluster_lsr'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
mice,lsr_clus,spon_clus,t = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1,  # clustered lsr P-waves
                                        mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], pnorm=pnorm, psmooth=(4,4), 
                                        fmax=25, p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename, ylim=[0.5,2])

# right - mean power in 1 s time window
x = np.intersect1d(np.where(t>=-0.5)[0], np.where(t<=0.5)[0])  # get columns between -0.5 s and +0.5 s
df = pd.DataFrame({'Mouse' : np.tile(mice,2),
                   'Event' : np.repeat(['single', 'cluster'], len(mice)),
                   'Pwr'   : np.concatenate((lsr_iso[(8,15)][:,x].mean(axis=1), lsr_clus[(8,15)][:,x].mean(axis=1))) })
fig = plt.figure(); sns.barplot(x='Event', y='Pwr', data=df, ci=68, palette={'single':'salmon', 'cluster':'mediumslateblue'})
sns.pointplot(x='Event', y='Pwr', hue='Mouse', data=df, ci=None, markers='', color='black'); plt.gca().get_legend().remove()
plt.title('Laser-triggered Single vs Clustered P-waves'); plt.show()

# stats
p = stats.ttest_rel(df['Pwr'].iloc[np.where(df['Event'] == 'single')[0]], df['Pwr'].iloc[np.where(df['Event'] == 'cluster')[0]])
print(f'single vs cluster laser P-waves -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')

#%%
###   Supp. FIGURE 5D,E,F - spectral power preceding successful & failed laser pulses   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

# D - normalized power spectrum
filename = 'sp_win3pre_pnorm1'; win=[-3,0]; pnorm=1
_ = pwaves.lsr_prev_theta_success(ppath, recordings, win=win, mode='spectrum', theta_band=[0,20], post_stim=0.1, pnorm=pnorm, psmooth=3, 
                                  ci='sem', nbins=14, prange1=(), prange2=(), mouse_avg='trials', pload=filename, psave=filename)
# E - mean theta power
filename = 'sp_win3pre_pnorm1'; win=[-3,0]; pnorm=1
_ = pwaves.lsr_prev_theta_success(ppath, recordings, win=win, mode='power', theta_band=[6,12], post_stim=0.1, pnorm=pnorm, psmooth=0, 
                                  ci='sem', nbins=14, prange1=(), prange2=(0,4), mouse_avg='trials', pload=filename, psave=filename)
# F - mean theta frequency
filename = 'sp_win3pre_pnorm0'; win=[-3,0]; pnorm=0
_ = pwaves.lsr_prev_theta_success(ppath, recordings, win=win, mode='mean freq', theta_band=[6,12], post_stim=0.1, pnorm=pnorm, psmooth=0, 
                                  ci='sem', nbins=14, prange1=(), prange2=(6.5,9.5), mouse_avg='trials', pload=filename, psave=filename)

#%%
###   Supp. FIGURE 5G - probability of eYFP laser success per brainstate   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves_yfp.txt')[1]
filename = 'lsr_stats_yfp'
df = pwaves.get_lsr_stats(ppath, recordings, istate=[1,2,3,4], post_stim=0.1, 
                          flatten_is=4, ma_thr=20, ma_state=3, psave=filename)
_ = pwaves.lsr_state_success(df, istate=[1,2,3,4])

#%%
###   Supp. FIGURE 5H,I - # P-waves/laser pulse summary stats for ChR2 and eYFP mice   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'
chr2_filepath = os.path.join(ppath, 'chr2_lsr_pwave_sumstats')
yfp_filepath = os.path.join(ppath, 'yfp_lsr_pwave_sumstats')

chr2_df = pd.read_csv(chr2_filepath)
chr2_df.insert(1, 'virus', 'chr2')
yfp_df = pd.read_csv(yfp_filepath)
yfp_df.insert(1, 'virus', 'yfp')

#%%
###   Supp. FIGURE 5K - EMG signal surrounding P-waves/laser pulses
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves_emg.txt')[1]

# raw EMG signal
filename = 'emg_win3'; wform_win = [3,3]; istate=[1]
df1 = pwaves.avg_waveform(ppath, recordings, istate, mode='pwaves', win=wform_win, mouse_avg='trial',     # spontaneous &
                          plaser=True, post_stim=0.1, pload=filename, psave=filename, exclude_noise=True, # laser-triggered P-waves
                          ci='sd', ylim=[-100,100], signal_type='EMG', dn=10)
df2 = pwaves.avg_waveform(ppath, recordings, istate, mode='lsr', win=wform_win, mouse_avg='trial',        # successful &
                          plaser=True, post_stim=0.1, pload=filename, psave=filename, exclude_noise=True, # failed laser
                          ci='sd', ylim=[-100,100], signal_type='EMG', dn=10)

# EMG amplitude
data, mice = pwaves.pwave_emg(ppath, recordings, emg_source='raw', win=[-1, 1], istate=[1], rem_cutoff=True, 
                              recalc_amp=True, nsr_seg=0.2, perc_overlap=0.5, recalc_highres=False, 
                              r_mu=[5,500], w0=5/(1000./2), w1=-1, dn=50, smooth=100, pzscore=2, tstart=0, tend=-1, 
                              ma_thr=20, ma_state=3, flatten_is=4, exclude_noise=True, plaser=True, sf=0,
                              post_stim=0.1, lsr_iso=0.5, lsr_mode='pwaves', mouse_avg='mouse', pplot=True, ylim=[-0.5,1.5])
lsr_data, spon_data, success_data, fail_data = data

#%%
###   Supp. FIGURE 6B - FISH c-Fos DREADDs quantification   ###
fos_df = pd.read_csv('/home/fearthekraken/Documents/Data/FISH_cfos_counts.csv')
fos_df = fos_df.iloc[np.where(fos_df['virus'] == 'hm3dq')[0], 
                     np.where(fos_df.columns != 'hemisphere')[0]].reset_index(drop=True)

fos_df['# crh+/cfos- cells'] = fos_df['# crh cells'] - fos_df['# co-expressing cells']
fos_df['# crh-/cfos+ cells'] = fos_df['# cfos cells'] - fos_df['# co-expressing cells']
fos_df['% crh cells expressing cfos'] = (fos_df['# co-expressing cells'] / fos_df['# crh cells']) * 100
# bar plot
bar_df = fos_df.groupby(['mouse']).sum().reset_index().loc[:,['mouse',
                                                              '# co-expressing cells',
                                                              '# crh-/cfos+ cells',
                                                              '# crh+/cfos- cells']]
bar_df.plot(kind='bar', stacked=True, color=['black','gray','lightgray'])

# pie chart
n_fos, n_nofos = fos_df.loc[:,['# co-expressing cells','# crh+/cfos- cells']].sum(axis=0)
plt.figure(figsize=(2,2))
ax = plt.gca()
_ = ax.pie([n_fos, n_nofos], explode=[0,1],
            labels=['cfos+','cfos-'], 
            colors=['limegreen','gray'], autopct='%1.1f%%', labeldistance=1.2, pctdistance=0.7,
            startangle=55, radius=2.5, wedgeprops={'linewidth': 2, 'edgecolor':'black'})

#%%
###   Supp. FIGURE 6C - hm3dq percent time spent in wake   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[2], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[2], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[2], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[2], tbin=18000, n=1, ma_state=3,
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
###   Supp. FIGURE 6D - hm3dq percent time spent in NREM   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[3], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[3], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[3], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[3], tbin=18000, n=1, ma_state=3,
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
###   Supp. FIGURE 6E - hm3dq percent time spent in IS   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']

# hm3dq mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[4], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[4], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[4], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[4], tbin=18000, n=1, ma_state=3,
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
###   Supp. FIGURE 6F,G - hm3dq P-wave frequency in wake, NREM, and IS   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True)
ctr_recordings_cno = ctr_recordings_cno['0.25']

### F
df = pd.DataFrame()
for s in [2,3,4]:
    # hm3dq mice
    exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    # mCherry mice
    ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    # hm3dq vs mCherry
    ddf = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                         mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                         dose_list=['saline','cno','saline','cno'],
                                         virus_list=['hm3dq','hm3dq','mCherry','mCherry'])
    ddf['state'] = s
    df = pd.concat([df, ddf], axis=0, ignore_index=True)
df = df.loc[:, ['mouse','virus','state','dose','t0']]
stateMap = {2:'wake', 3:'NREM', 4:'IS'}
df.replace({'state':stateMap}, inplace=True)
df.rename(columns={'dose':'drug', 't0':'p-wave freq'}, inplace=True)

### G
difs = np.array(df[df.drug=='cno']['p-wave freq']) - np.array(df[df.drug=='saline']['p-wave freq'])
dif_df = df[df.drug=='cno'].copy().reset_index(drop=True)
dif_df['dif. p-wave freq'] = difs
dif_df = dif_df.loc[:, ['mouse','virus','state','dif. p-wave freq']]

# stats
for s in stateMap.values():
    ddf = df[df.state==s].reset_index(drop=True)
    res_anova = ping.mixed_anova(data=ddf, dv='p-wave freq', within='drug', subject='mouse', between='virus')
    mc1 = ping.pairwise_tests(data=ddf, dv='p-wave freq', within='drug', subject='mouse', between='virus', 
                              padjust='holm', within_first=False)
    mc2 = ping.pairwise_tests(data=ddf, dv='p-wave freq', within='drug', subject='mouse', between='virus', 
                              padjust='holm', within_first=True)
    print('='*len(s) + '\n' + s + '\n' + '='*len(s))
    AS.print_anova(res_anova, mc1, mc2, 'SALINE vs CNO', 'hM3D(Gq) vs mCherry')
    
    dif_ddf = dif_df[dif_df.state==s].reset_index(drop=True)
    a,b = [np.array(dif_ddf[dif_ddf.virus==v]['dif. p-wave freq']) for v in ['hm3dq','mCherry']]
    p = scipy.stats.ttest_ind(a, b)
    print('\n\n###   hM3D(Gq) vs eYFP   ###')
    print(f'T({np.sum([len(a)-1, len(b)-1])})={np.round(p.statistic,2)}, p={pp(p.pvalue)[0]}\n')
    print('\n__________________________________________\n')

#%%
###   Supp. FIGURE 6H - hm3dq vs mCherry sleep spectrum   ###
bands=[(0.5,4), (6,10), (11,15), (15.5,20)]
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'

# hm3dq mice
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['0.25']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['0.25']
exp_df = pd.DataFrame()
print('### hm3dq mice')
for s in [1,2,3,4]:
    print(f'....... state = {s}')
    _, _, ddf_sal, _ = sleepy.sleep_spectrum_simple(ppath, exp_recordings_sal, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_sal['drug'] = 'saline'
    
    _, _, ddf_cno, _ = sleepy.sleep_spectrum_simple(ppath, exp_recordings_cno, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_cno['drug'] = 'cno'
    ddf = pd.concat([ddf_sal, ddf_cno], axis=0, ignore_index=True)
    ddf['state'] = s
    exp_df = pd.concat([exp_df, ddf], axis=0, ignore_index=True)
exp_df['virus'] = 'hm3dq'

# mCherry mice
ctr_df = pd.DataFrame()
print('### mCherry mice')
for s in [1,2,3,4]:
    print(f'....... state = {s}')
    _, _, ddf_sal, _ = sleepy.sleep_spectrum_simple(ppath, ctr_recordings_sal, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_sal['drug'] = 'saline'
    
    _, _, ddf_cno, _ = sleepy.sleep_spectrum_simple(ppath, ctr_recordings_cno, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_cno['drug'] = 'cno'
    ddf = pd.concat([ddf_sal, ddf_cno], axis=0, ignore_index=True)
    ddf['state'] = s
    ctr_df = pd.concat([ctr_df, ddf], axis=0, ignore_index=True)
ctr_df['virus'] = 'mCherry'
df = pd.concat([exp_df, ctr_df], axis=0, ignore_index=True)
stateMap = {1:'REM', 2:'wake', 3:'NREM', 4:'IS'}
df.replace({'state' : stateMap}, inplace=True)
df.rename(columns={'pow':'pwr'}, inplace=True)
df = df.loc[:, ['mouse','virus','state','drug','freq','pwr']]

# calculate CNO trials - saline trials power
df2 = pd.DataFrame()
df3 = pd.DataFrame()
for band in bands:
    A = df.iloc[np.where((df.freq>=band[0]) & (df.freq<=band[1]))[0], :].groupby(['virus','mouse','drug','state']).sum().reset_index()
    A['pwr'] = A['pwr'] * 0.5
    A['band'] = f'{band[0]} - {band[1]} Hz'
    df2 = pd.concat([df2, A], axis=0, ignore_index=True)
    # get CNO minus saline power
    difs = np.array(A[A.drug=='cno']['pwr']) - np.array(A[A.drug=='saline']['pwr'])
    B = A[A.drug=='cno'].copy()
    B['dif. pwr'] = difs
    df3 = pd.concat([df3, B], axis=0, ignore_index=True)
df2 = df2.loc[:, ['mouse','band','virus','state','drug','pwr']]
df3 = df3.loc[:, ['mouse','band','virus','state','dif. pwr']]

# stats
df2_exp = df2[df2.virus=='hm3dq'].reset_index(drop=True)
df2_ctr = df2[df2.virus=='mCherry'].reset_index(drop=True)
for s in stateMap.values():
    print('='*len(s) + '\n' + s + '\n' + '='*len(s))
    print('\n###   hM3D(Gq) mice   ###\n')
    ddf_exp = df2_exp[df2_exp.state==s].reset_index(drop=True)
    res_exp = ping.rm_anova(data=ddf_exp, subject='mouse', dv='pwr', within=['band','drug'])
    print(res_exp.to_string())
    
    print('\n###   mCherry mice   ###\n')
    ddf_ctr = df2_ctr[df2_ctr.state==s].reset_index(drop=True)
    res_ctr = ping.rm_anova(data=ddf_ctr, subject='mouse', dv='pwr', within=['band','drug'])
    print(res_ctr.to_string())
    
    print('\n###   hM3D(Gq) vs mCherry mice   ###\n')
    ddf_dif = df3[df3.state==s].reset_index(drop=True)
    res_dif = ping.mixed_anova(data=ddf_dif, subject='mouse', dv='dif. pwr', within='band', between='virus')
    print(res_dif.to_string())
    print('\n__________________________________________\n')

#%%
###   Supp. FIGURE 7A - hm4di percent time spent in wake   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[2], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[2], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[2], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[2], tbin=18000, n=1, ma_state=3,
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
###   Supp. FIGURE 7B - hm4di percent time spent in NREM   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[3], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[3], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[3], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[3], tbin=18000, n=1, ma_state=3,
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
###   Supp. FIGURE 7C - hm4di percent time spent in IS   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[4], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[4], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
# mCherry mice
ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[4], tbin=18000, n=1, ma_state=3,
                                           stats='perc', flatten_is=4, exclude_noise=True, pplot=False)
ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[4], tbin=18000, n=1, ma_state=3,
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
###   Supp. FIGURE 7D,E - hm4di P-wave frequency in wake, NREM, and IS   ###
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=True)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True)
ctr_recordings_cno = ctr_recordings_cno['5']

### D
df = pd.DataFrame()
for s in [2,3,4]:
    # hm4di mice
    exp_mice, exp_cT = pwaves.sleep_timecourse(ppath, exp_recordings_sal, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    exp_mice, exp_eT = pwaves.sleep_timecourse(ppath, exp_recordings_cno, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    # mCherry mice
    ctr_mice, ctr_cT = pwaves.sleep_timecourse(ppath, ctr_recordings_sal, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    ctr_mice, ctr_eT = pwaves.sleep_timecourse(ppath, ctr_recordings_cno, istate=[s], tbin=18000, n=1, ma_state=6,
                                               stats='pwave freq', flatten_is=4, exclude_noise=True, pplot=False)
    # hm4di vs mCherry
    ddf = pwaves.df_from_timecourse_dict(tdict_list=[exp_cT, exp_eT, ctr_cT, ctr_eT],
                                         mice_list=[exp_mice, exp_mice, ctr_mice, ctr_mice],
                                         dose_list=['saline','cno','saline','cno'],
                                         virus_list=['hm4di','hm4di','mCherry','mCherry'])
    ddf['state'] = s
    df = pd.concat([df, ddf], axis=0, ignore_index=True)
df = df.loc[:, ['mouse','virus','state','dose','t0']]
stateMap = {2:'wake', 3:'NREM', 4:'IS'}
df.replace({'state':stateMap}, inplace=True)
df.rename(columns={'dose':'drug', 't0':'p-wave freq'}, inplace=True)

### E
difs = np.array(df[df.drug=='cno']['p-wave freq']) - np.array(df[df.drug=='saline']['p-wave freq'])
dif_df = df[df.drug=='cno'].copy().reset_index(drop=True)
dif_df['dif. p-wave freq'] = difs
dif_df = dif_df.loc[:, ['mouse','virus','state','dif. p-wave freq']]

# stats
for s in stateMap.values():
    ddf = df[df.state==s].reset_index(drop=True)
    res_anova = ping.mixed_anova(data=ddf, dv='p-wave freq', within='drug', subject='mouse', between='virus')
    mc1 = ping.pairwise_tests(data=ddf, dv='p-wave freq', within='drug', subject='mouse', between='virus', 
                              padjust='holm', within_first=False)
    mc2 = ping.pairwise_tests(data=ddf, dv='p-wave freq', within='drug', subject='mouse', between='virus', 
                              padjust='holm', within_first=True)
    print('='*len(s) + '\n' + s + '\n' + '='*len(s))
    AS.print_anova(res_anova, mc1, mc2, 'SALINE vs CNO', 'hM4D(Gi) vs mCherry')
    
    dif_ddf = dif_df[dif_df.state==s].reset_index(drop=True)
    a,b = [np.array(dif_ddf[dif_ddf.virus==v]['dif. p-wave freq']) for v in ['hm4di','mCherry']]
    p = scipy.stats.ttest_ind(a, b)
    print('\n\n###   hM4D(Gi) vs eYFP   ###')
    print(f'T({np.sum([len(a)-1, len(b)-1])})={np.round(p.statistic,2)}, p={pp(p.pvalue)[0]}\n')
    print('\n__________________________________________\n')
    
#%%
###   Supp. FIGURE 7F - hm4di vs mCherry sleep spectrum   ###
bands=[(0.5,4), (6,10), (11,15), (15.5,20)]
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(exp_recordings_sal, exp_recordings_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False)
exp_recordings_cno = exp_recordings_cno['5']
(ctr_recordings_sal, ctr_recordings_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
ctr_recordings_cno = ctr_recordings_cno['5']

# hm4di mice
exp_df = pd.DataFrame()
print('### hm4di mice')
for s in [1,2,3,4]:
    print(f'....... state = {s}')
    _, _, ddf_sal, _ = sleepy.sleep_spectrum_simple(ppath, exp_recordings_sal, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_sal['drug'] = 'saline'
    
    _, _, ddf_cno, _ = sleepy.sleep_spectrum_simple(ppath, exp_recordings_cno, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_cno['drug'] = 'cno'
    ddf = pd.concat([ddf_sal, ddf_cno], axis=0, ignore_index=True)
    ddf['state'] = s
    exp_df = pd.concat([exp_df, ddf], axis=0, ignore_index=True)
exp_df['virus'] = 'hm4di'

# mCherry mice
ctr_df = pd.DataFrame()
print('### mCherry mice')
for s in [1,2,3,4]:
    print(f'....... state = {s}')
    _, _, ddf_sal, _ = sleepy.sleep_spectrum_simple(ppath, ctr_recordings_sal, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_sal['drug'] = 'saline'
    
    _, _, ddf_cno, _ = sleepy.sleep_spectrum_simple(ppath, ctr_recordings_cno, istate=s, pmode=0, harmcs=0, fmax=20, mu=[5,100], 
                                                    ci=95, pnorm=False, harmcs_mode='iplt', iplt_level=1, round_freq=True, 
                                                    exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, pplot=False)
    ddf_cno['drug'] = 'cno'
    ddf = pd.concat([ddf_sal, ddf_cno], axis=0, ignore_index=True)
    ddf['state'] = s
    ctr_df = pd.concat([ctr_df, ddf], axis=0, ignore_index=True)
ctr_df['virus'] = 'mCherry'
df = pd.concat([exp_df, ctr_df], axis=0, ignore_index=True)
stateMap = {1:'REM', 2:'wake', 3:'NREM', 4:'IS'}
df.replace({'state' : stateMap}, inplace=True)
df.rename(columns={'pow':'pwr'}, inplace=True)
df = df.loc[:, ['mouse','virus','state','drug','freq','pwr']]

# calculate CNO trials - saline trials power
df2 = pd.DataFrame()
df3 = pd.DataFrame()
for band in bands:
    A = df.iloc[np.where((df.freq>=band[0]) & (df.freq<=band[1]))[0], :].groupby(['virus','mouse','drug','state']).sum().reset_index()
    A['pwr'] = A['pwr'] * 0.5
    A['band'] = f'{band[0]} - {band[1]} Hz'
    df2 = pd.concat([df2, A], axis=0, ignore_index=True)
    # get CNO minus saline power
    difs = np.array(A[A.drug=='cno']['pwr']) - np.array(A[A.drug=='saline']['pwr'])
    B = A[A.drug=='cno'].copy()
    B['dif. pwr'] = difs
    df3 = pd.concat([df3, B], axis=0, ignore_index=True)
df2 = df2.loc[:, ['mouse','band','virus','state','drug','pwr']]
df3 = df3.loc[:, ['mouse','band','virus','state','dif. pwr']]

# stats
df2_exp = df2[df2.virus=='hm4di'].reset_index(drop=True)
df2_ctr = df2[df2.virus=='mCherry'].reset_index(drop=True)
for s in stateMap.values():
    print('='*len(s) + '\n' + s + '\n' + '='*len(s))
    print('\n###   hM4D(Gi) mice   ###\n')
    ddf_exp = df2_exp[df2_exp.state==s].reset_index(drop=True)
    res_exp = ping.rm_anova(data=ddf_exp, subject='mouse', dv='pwr', within=['band','drug'])
    print(res_exp.to_string())
    
    print('\n###   mCherry mice   ###\n')
    ddf_ctr = df2_ctr[df2_ctr.state==s].reset_index(drop=True)
    res_ctr = ping.rm_anova(data=ddf_ctr, subject='mouse', dv='pwr', within=['band','drug'])
    print(res_ctr.to_string())
    
    print('\n###   hM4D(Gi) vs mCherry mice   ###\n')
    ddf_dif = df3[df3.state==s].reset_index(drop=True)
    res_dif = ping.mixed_anova(data=ddf_dif, subject='mouse', dv='dif. pwr', within='band', between='virus')
    print(res_dif.to_string())
    print('\n__________________________________________\n')