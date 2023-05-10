#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:37:58 2022

@author: fearthekraken
"""
import os.path
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.io as so
import sleepy
import AS
import pwaves

#%%
"""
#######             Example 1: P-wave Detection             #######

Detect P-waves in pontine LFP signals from one recording.

* Code from $pwaves.detect_pwaves()

"""

# path to experimental recording
ppath = os.getcwd() + '/data'
rec = 'mouse1_072420n1'

# set parameters for P-wave detection

channel='S'  # subtract reference from primary LFP
thres=4.5  # threshold to use for detecting P-waves
thres_init=5  # intial threshold to identify the primary LFP channel
dup_win=40  # perform additional validation on P-waves within $dup_win ms of each other
select_thresholds=False  # manually set P-wave threshold (True) or use $thres (False)
set_outlier_th=False  # set outlier thresholds for ampl/halfwidth (True) or use default values (False)
rm_noise=True; n_win=5  # if True, detect LFP artifacts and eliminate waveforms within $nwin s
n_thres=800  # threshold (uV) for detecting LFP noise


print('\nDetecting P-waves for ' + rec + ' ...\n')
# load sampling rate
sr = sleepy.get_snr(ppath, rec)
nbin = int(np.round(sr) * 2.5)
dt = (1.0 / sr) * nbin
        
# load and adjust brain state annotation
M = sleepy.load_stateidx(ppath, rec)[0]
M = AS.adjust_brainstate(M, dt, ma_thr=20, ma_state=6, flatten_tnrem=3)
        
# get indices for each brain state
s_idx = {1:np.array(()), 2:np.array(()), 3:np.array(())}
for s in [1,2,3]:
    sseq = sleepy.get_sequences(np.where(M==s)[0])
    s_idx[s] = np.concatenate([np.arange(seq[0]*nbin, seq[-1]*nbin) for seq in sseq])
    if s == 3:
        # LFP mean/std calculated using 0-10 s of every NREM period >= 1 min
        base_idx = np.concatenate([np.arange(seq[0]*nbin, seq[int(10/dt)]*nbin) \
                                   for seq in sseq if len(seq) >= 60/dt])

LFP_raw = []; LFP_raw2 = []
# if mouse has labeled LFPs in channel allocation, use those to load LFPs
if os.path.exists(os.path.join(ppath, rec, 'LFP_raw.mat')):
    LFP_raw = so.loadmat(os.path.join(ppath, rec, 'LFP_raw'), 
                         squeeze_me=True)['LFP_raw']
# if LFP_raw channels don't exist, LFPs are probably under EMG1 and EMG2
elif os.path.exists(os.path.join(ppath, rec, 'EMG.mat')):
    LFP_raw = so.loadmat(os.path.join(ppath, rec, 'EMG.mat'), 
                         squeeze_me=True)['EMG']
    so.savemat(os.path.join(ppath, rec, 'LFP_raw.mat'), {'LFP_raw': LFP_raw})
if os.path.exists(os.path.join(ppath, rec, 'LFP_raw2.mat')):
    LFP_raw2 = so.loadmat(os.path.join(ppath, rec, 'LFP_raw2'), 
                          squeeze_me=True)['LFP_raw2']
elif os.path.exists(os.path.join(ppath, rec, 'EMG2.mat')):
    LFP_raw2 = so.loadmat(os.path.join(ppath, rec, 'EMG2.mat'), 
                          squeeze_me=True)['EMG2']
    so.savemat(os.path.join(ppath, rec, 'LFP_raw2.mat'), {'LFP_raw2': LFP_raw2})

# handle cases with a single LFP channel
if len(LFP_raw) == 0:
    if len(LFP_raw2) == 0:
        raise Exception("ERROR - No LFP files found")
    else:
        LFP_raw = np.zeros(len(LFP_raw2))
        channel = 2
elif len(LFP_raw2) == 0:
    LFP_raw2 = np.zeros(len(LFP_raw))
    channel = 1

# filter LFP channels
w1 = 1. / (sr/2.0)
w2 = 50. / (sr/2.0)
LFP_raw = sleepy.my_bpfilter(LFP_raw, w1, w2)
LFP_raw2 = sleepy.my_bpfilter(LFP_raw2, w1, w2)

# get approx. number of P-waves on each channel
th1 = np.nanmean(LFP_raw[base_idx]) + thres_init*np.nanstd(LFP_raw[base_idx])
idx1 = pwaves.spike_threshold(LFP_raw, th1) 
th2 = np.nanmean(LFP_raw2[base_idx]) + thres_init*np.nanstd(LFP_raw2[base_idx])
idx2 = pwaves.spike_threshold(LFP_raw2, th2)

# channel = S: subtract reference from primary LFP
if channel == 'S':
    if len(idx1) > len(idx2):
        LFP = LFP_raw - LFP_raw2
    else:
        LFP = LFP_raw2 - LFP_raw
# channel = C: use primary LFP
elif channel == 'C':
    if len(idx1) > len(idx2):
        LFP = LFP_raw
    else:
        LFP = LFP_raw2
# channel = 1: use LFP 1
elif channel == 1 or channel == '1':
    LFP = LFP_raw
# channel = 2: use LFP 2
elif channel == 2 or channel == '2':
    LFP = LFP_raw2
# channel = 1-2: subtract LFP 2 from LFP 1
elif channel == '1-2':
    LFP = LFP_raw - LFP_raw2
# channel = 2-1: subtract LFP 1 from LFP 2
elif channel == '2-1':
    LFP = LFP_raw2 - LFP_raw
    
# manually select optimal threshold value
if select_thresholds:
    thres_range = np.arange(3, 7, 0.5)
    nidx = {1 : np.zeros((len(thres_range))), 
            2 : np.zeros((len(thres_range))), 
            3 : np.zeros((len(thres_range)))}
    # detect P-waves using range of threshold values
    print('')
    for s in [1,2,3]:
        sLFP = LFP[s_idx[s]]
        print('Getting P-waves for state ' + str(s))
        for i, thres in enumerate(thres_range):
            th = np.nanmean(LFP[base_idx]) + thres*np.nanstd(LFP[base_idx])
            si = pwaves.spike_threshold(sLFP, th)
            nidx[s][i] = len(si)
    # plot number of detected P-waves per threshold value
    fig, axs = plt.subplots(figsize=(5,10), nrows=3, ncols=1, 
                            constrained_layout=True)
    for s,st in zip([1,2,3],['REM','wake','NREM']):
        axs[s-1].bar(thres_range, nidx[s])
        axs[s-1].set_ylabel('No. of P-waves')
        axs[s-1].set_title(st)
    axs[-1].set_xlabel('Threshold value')
    fig.suptitle(rec)
    plt.show()
    # enter desired threshold
    p_thres = float(input('Enter the threshold value for P-wave detection ---> '))
    
# use $thres param as common threshold
else:
    p_thres = thres

# get P-wave indices from processed LFP using chosen threshold 
p_th = np.nanmean(LFP[base_idx]) + p_thres*np.nanstd(LFP[base_idx])
pi = pwaves.spike_threshold(LFP, p_th)
# get amplitudes and half-widths of P-waves
p_amps = [pwaves.get_amp(LFP, i, sr) for i in pi]
p_widths = [pwaves.get_halfwidth(LFP, i, sr) for i in pi]
df = pd.DataFrame({'idx':pi, 'amp':p_amps, 'halfwidth':p_widths})
df.dropna(inplace=True)
p_idx = list(df['idx'])

# set default threshold values for outlier amplitudes and half-widths
init_amp_thres = 500.  # uV
init_hw_thres = 80./1000*sr  # ms

if set_outlier_th:
    # plot P-wave amplitudes, mark default outlier threshold
    plt.figure()
    sns.histplot(x='amp', data=df, ax=plt.gca())
    plt.plot([init_amp_thres, init_amp_thres], [0, plt.gca().get_ylim()[1]], color='red')
    plt.title(rec + ' P-wave amplitudes')
    plt.show()
    # enter threshold for outlier amplitudes (uV)
    amp_thres = float(input('Enter threshold for outlier amplitudes (uV) ---> '))
    print('')
    # plot P-wave half-widths, mark default outlier threshold
    plt.figure()
    sns.histplot(x='halfwidth', data=df, ax=plt.gca())
    plt.plot([init_hw_thres, init_hw_thres], [0, plt.gca().get_ylim()[1]], color='red')
    plt.title(rec + ' P-wave half-widths')
    plt.show()
    # enter threshold for outlier half-widths (ms)
    hw_thres = float(input('Enter threshold for outlier half-widths (ms) ---> '))
    hw_thres = hw_thres/1000*sr
    print('')
else:
    amp_thres = init_amp_thres
    hw_thres = init_hw_thres
    
# collect waveforms with outlier amplitudes and/or half-widths
amp_hi = list(df['idx'].iloc[np.where(df['amp'] > amp_thres)[0]])
width_hi = list(df['idx'].iloc[np.where(df['halfwidth'] > hw_thres)[0]])

# check whether closely neighboring P-waves are separate events
pdifs = np.array(([p_idx[i] - p_idx[i-1] for i in range(1, len(p_idx))]))
tst_dups = np.where(pdifs < dup_win/1000*sr)[0]
dup_waves = []
for di in tst_dups:
    p1 = p_idx[di]
    p2 = p_idx[di+1]
    # if first P-wave has already been classified as a duplicate, continue
    if p1 in dup_waves:
        continue
    else:  
        p1_mid = np.abs(LFP[p1] - max(LFP[p1:p2]))
        p2_mid = np.abs(LFP[p2] - max(LFP[p1:p2]))
        # if LFP deflects significantly upward between adjacent P-waves:
        if p1_mid > 100 and p2_mid > 75:
            # count as separate waves
            continue
        # if single wave was "double-detected"
        else:
            # if 2nd threshold crossing is larger, classify 1st as duplicate
            if LFP[p2] < LFP[p1]:
                dup_waves.append(p1)
            # if 1st threshold crossing is larger, classify 2nd as duplicate
            elif LFP[p1] < LFP[p2]:
                dup_waves.append(p2)

# ignore waves within "noisy" LFP regions
if rm_noise:
    noise_idx = pwaves.detect_noise(LFP, sr, n_thres, n_win)
    noise_waves = [i for i in p_idx if i in noise_idx]
        
# eliminate all disqualified waves
p_elim = amp_hi + width_hi + dup_waves
if rm_noise:
    p_elim += noise_waves
p_idx = [i for i in p_idx if i not in p_elim]
# save indices of outlier and duplicate waves for later inspection
q = np.zeros((3,len(LFP)))
q[0, amp_hi] = 1
q[1, width_hi] = 3
q[2, dup_waves] = 5

# save processed LFP
so.savemat(os.path.join(ppath, rec, 'LFP_processed.mat'), {'LFP_processed': LFP})
# save P-wave indices, detection threshold, and outlier waveforms
so.savemat(os.path.join(ppath, rec, 'p_idx.mat'), {'p_idx': p_idx, 
                                                   'thres': p_th, 
                                                   'thres_scale': p_thres,
                                                   'p_elim': q})
# view example P-waves during REM sleep
remseq = sleepy.get_sequences(np.where(M==1)[0])[7]
AS.plot_example(ppath, rec, PLOT=['HYPNO','LFP_ANNOT_THRES'], tstart=remseq[0]*dt-20, tend=remseq[-1]*dt+15)

#%%
"""
#######         Example 2: Spectral Field Estimation (Fig. 1H)        #######

Estimate the "spectral field" optimally mapping the EEG spectrogram onto dmM
CRH neuron calcium activity for one recording.

* Code from $pwaves.spectralfield_highres()

"""

# path to experimental recording
ppath = os.getcwd() + '/data'
rec = 'mouse2_032819n1'

# set parameters for spectral field plot

pre=4; post=4  # no. seconds before & after neural response to estimate spectral field
istate=[1]  # brain state to analyze (1 = REM)
theta=[1,10,100,1000,10000]  # candidate values for regularization term
pnorm=1  # normalize each spectrogram frequency by its mean power
fmax=25  # max frequency in spectral field
nsr_seg=2  # bin size (s) for fast fourier transform (FFT)
perc_overlap=0.8  # overlap (%) of FFT bins
pzscore=False  # use raw DF/F values (False) or z-scored values (True)
nfold=5  # perform 5-fold cross-validation of regression model

# load sampling rate
sr = sleepy.get_snr(ppath, rec)
nbin = int(np.round(sr) * 2.5)

# load and adjust brain state annotation
M = sleepy.load_stateidx(ppath, rec)[0]
M = AS.adjust_brainstate(M, dt=2.5, ma_thr=20, ma_state=3, flatten_tnrem=4)

# load/calculate high-resolution EEG spectrogram
SP, freq, t = AS.highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap,
                                     recalc_highres=True)[0:3]
N = SP.shape[1]
ifreq = np.where(freq <= fmax)[0]
f = freq[ifreq]
nfreq = len(ifreq)
dt = t[1]-t[0]

# get indices of time window to collect EEG spectrogram
ipre  = int(np.round(pre/dt))
ipost = int(np.round(post/dt))
# normalize spectrogram, cutoff rows at $fmax
MX = AS.adjust_spectrogram(SP, pnorm=pnorm, psmooth=0)[ifreq,:]

N = MX.shape[1]  # no. of time bins
nfreq = MX.shape[0]  # no. of frequencies

# create feature mx (1 row per time bin, 1 column per freq per time bin in collection window)
A = np.zeros((N - ipre - ipost, nfreq * (ipre + ipost)))
j = 0
for i in range(ipre, N - ipost-1):
    # collect SP window surrounding each consecutive time bin
    C = MX[:, i - ipre:i + ipost]
    # reshape SP window to 1D vector, store in next row of matrix $A
    A[j,:] = np.reshape(C.T, (nfreq * (ipre + ipost),))
    j += 1
MX = A

# load DF/F calcium signal, downsample to SP time resolution
ndown = int(nsr_seg*sr) - int(nsr_seg*sr*perc_overlap)
ninit = int(np.round(t[0]/dt))
dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dff']*100
dffd = AS.downsample_vec(dff, ndown)
dffd = dffd[ninit:]
if pzscore:
    dffd = (dffd-dffd.mean()) / dffd.std()
dffd = dffd[ipre:N-ipost]

# get indices of all brain states in $istate
ibin = np.array([], dtype='int64')
M,K = sleepy.load_stateidx(ppath, rec)
for s in istate:
    seq = sleepy.get_sequences(np.where(M==s)[0])
    for p in seq:
        # convert to high-resolution indices in SP
        seqm = np.arange(int(p[0]*nbin / ndown), int(((p[-1]+1)*nbin-1)/ ndown))
        ibin = np.concatenate((ibin, seqm))
ibin = ibin[ibin>=ipre]
ibin = ibin[ibin<N-ipost]

# isolate rows where SP window centers on a time bin in $istate
MX = MX[ibin-ipre,:]
dffd = dffd[ibin-ipre]
# normalize DF/F and SP frequencies to mean of zero within time window
rmean = dffd.mean()
dffd = dffd - rmean
mmean = MX.mean(axis=0)
for i in range(MX.shape[1]):
    MX[:,i] -= mmean[i]

# perform $nfold-fold cross-validation for linear regression
ntheta = len(theta)
ninp = dffd.shape[0]
nsub = round(ninp / nfold)
nparam = MX.shape[1]
 # normalize variable values to mean of zero
for i in range(nparam):
    MX[:,i] -= MX[:,i].mean()
# get $nfold subsets of training & test data
test_idx = []
train_idx = []
for i in range(nfold):
    idx = np.arange(i*nsub, min(((i+1)*nsub, ninp)), dtype='int')
    idy = np.setdiff1d(np.arange(0, ninp), idx)
    test_idx.append(idx)
    train_idx.append(idy)

l = 0
# for each regularization param, collect avg model performance (R2 values)
Etest = np.zeros((ntheta,))
Etrain = np.zeros((ntheta,))
for th in theta:
    ptest = np.zeros((nfold,))
    ptrain = np.zeros((nfold,))
    j = 0
    # perform $nfold iterations of training/testing for regression model
    for (p, q) in zip(test_idx, train_idx):
        S = MX[q,:].copy()
        n = S.shape[1]
        AC = np.dot(S.T, S)  # input matrix SP
        AC = AC + np.eye(n)*th  # power constraint
        SR = np.dot(S.T, dffd[q])  # output matrix SP*signal
        # linear solution optimally approximating MX * k = dffd
        k = scipy.linalg.solve(AC, SR)

        pred_test  = np.dot(MX[p,:], k)  # predicted values
        pred_train = np.dot(MX[q,:], k)
        rtest  = dffd[p]  # measured values
        rtrain = dffd[q]
        # model performance
        ptest[j] = 1 - np.var(pred_test - rtest) / np.var(rtest)
        ptrain[j] = 1 - np.var(pred_train - rtrain) / np.var(rtrain)
        j += 1
    Etest[l] = np.mean(ptest)
    Etrain[l] = np.mean(ptrain)
    l += 1
print("CV results on training set:")
print(Etrain)
print("CV results on test set")
print(Etest)

# get optimal theta value, estimate optimal spectral field
imax = np.argmax(Etest)
print("Recording %s; optimal theta: %2.f" % (rec, theta[imax]))
S = MX.copy()
n = S.shape[1]
AC = np.dot(S.T, S)
AC = AC + np.eye(n)*theta[imax]
SR = np.dot(S.T, dffd)
k = scipy.linalg.solve(AC, SR)
# orient spectral field (freq x time)
k = np.reshape(k, ((ipre + ipost), nfreq)).T
t = np.arange(-ipre, ipost) * dt

# plot spectral field as heatmap
plt.ion()
plt.figure()
ax = plt.gca()
im = ax.pcolorfast(t, f, k, cmap='jet')
plt.colorbar(im, ax=ax, pad=0.05)
plt.xlabel('Time (s)')
plt.ylabel('Freq. (Hz)')
plt.title(f'Spectral Field\n{rec}')
plt.show()

#%%
"""
#######          Example 3: Spectral Profiles (Fig. 4C-F)       #######

Calculate average spectrograms surrounding spontaneous P-waves, laser-triggered P-waves,
failed laser pulses, and random control points in one recording, and plot the normalized 
power spectral density of each event.

* Code from $pwaves.avg_SP() and $pwaves.sp_profiles()

"""

# path to experimental recording
ppath = os.getcwd() + '/data'
recordings = ['mouse3_110619n1']

# set parameters for spectral plots

collect_win=[-3,3]  # time window (s) to collect SPs relative to P-waves and laser pulses
spon_win=[-0.5, 0.5]  # time windows (s) to isolate within collected SPs for P-waves and random
lsr_win=[0,1]#           control points (spon_win) and for laser pulses (lsr_win)
mode='pwaves'  # plot SPs for spontaneous vs laser $'pwaves', or successful vs failed $'lsr' pulses
istate=[1]  # brain state(s) to analyze (1=REM, 2=wake, 3=NREM, 4=IS)
pnorm=2  # SP normalization mode (2=normalize each freq by its power within the collected time window)
psmooth=[0,0]; vm=[[],[]]; fmax=25  # smoothing, saturation, and max freq of spectrograms
frange=[0,20]  # [min, max] freq in power spectral density plot
plaser=True  # plot laser pulses & laser-triggered P-waves (True) or spontaneous P-waves only (False)
post_stim=0.1  # max latency (s) of P-wave from laser onset to be classified as "laser-triggered"
null=True  # if True, plot spectral power of random control points
null_win=0  # control points must have no P-waves or laser pulses in surrounding $null_win s
null_match='lsr'  # if 'lsr', no. of control points is matched to the no. of laser pulses
mouse_avg='mouse'  # average SPs within each $'mouse' or $'rec'[ording], or across all $'trials'
pload=False; psave=False  # load and/or save data

# set normalization mode for spectrograms
if pnorm == 0:
    signal_type = 'SP'  # load raw SP, no normalization
    pnorm = False
elif pnorm == 1:
    signal_type = 'SP_NORM'  # load SP normalized by recording 
    pnorm = False
elif pnorm == 2:
    signal_type = 'SP'  # load raw SP, normalize by collected time window
    pnorm = True

states = {'total':'total',1:'REM',2:'Wake',3:'NREM',4:'IS',5:'failed-IS',6:'Microarousals'}
    
# load SP data
if pload:
    data = AS.load_surround_files(ppath, pload=pload, istate=istate, plaser=plaser, 
                                  null=null, signal_type=signal_type)
    if len(data) > 0:
        if plaser:
            lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data
        elif not plaser:
            p_signal, null_pts, data_shape = data
    else:
        pload = False

# collect SPs surrounding events
if not pload:
    if plaser:
        data = pwaves.get_lsr_surround(ppath, recordings, istate=istate, win=collect_win, 
                                       signal_type=signal_type, null=null, null_win=null_win, 
                                       null_match=null_match, post_stim=post_stim, psave=psave)
        lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data

    elif not plaser:
        data = pwaves.get_surround(ppath, recordings, istate=istate, win=collect_win, 
                                   signal_type=signal_type, null=null, null_win=null_win, psave=psave)
        p_signal, null_pts, data_shape = data
    
ifreq = np.arange(0, fmax*2+1)  # frequency idxs up to $fmax
freq = np.arange(0, fmax+0.5, 0.5)  # frequencies
x = np.linspace(-np.abs(collect_win[0]), collect_win[1], data_shape[1])  # x axis

###   PLOT SPECTROGRAMS   ###
if plaser:
    # for each brainstate, create a data matrix of freq x time bins x subject
    for s in istate:
        # mx1 for laser-triggered P-waves, mx2 for spontaneous P-waves
        if mode == 'pwaves':
            mx1, labels = pwaves.mx3d(lsr_pwaves[s], mouse_avg, data_shape)
            mx2, labels = pwaves.mx3d(spon_pwaves[s], mouse_avg, data_shape)
            title1 = 'Laser-triggered P-waves'
            title2 = 'Spontaneous P-waves'
        # mx1 for successful lsr, mx2 for failed lsr
        elif mode == 'lsr':
            mx1, labels = pwaves.mx3d(success_lsr[s], mouse_avg)
            mx2, labels = pwaves.mx3d(fail_lsr[s], mouse_avg)
            title1 = 'Successful laser pulses'
            title2 = 'Failed laser pulses'
        # mx1 for randomized control points, mx2 for failed lsr
        elif mode == 'null':
            mx1, labels = pwaves.mx3d(null_pts[s], mouse_avg)
            mx2, labels = pwaves.mx3d(fail_lsr[s], mouse_avg)
            title1 = 'Null points'
            title2 = 'Failed laser pulses'

        # average, normalize, and smooth SPs
        mx1_plot = AS.adjust_spectrogram(np.nanmean(mx1, axis=2)[ifreq, :], pnorm, psmooth[0])
        mx2_plot = AS.adjust_spectrogram(np.nanmean(mx2, axis=2)[ifreq, :], pnorm, psmooth[1])
        
        # plot SP1 and SP2
        fig, (lax, sax) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        fig.suptitle(f'Average spectrograms ({recordings[0]})')
        lim = lax.pcolorfast(x, freq, mx1_plot, cmap='jet')
        if len(vm[0])==2:
            lim.set_clim(vm[0])
        plt.colorbar(lim, ax=lax, pad=0.0)
        lax.set_xlabel('Time (s)')
        lax.set_ylabel('Freq (Hz)')
        lax.set_title(f'{title1} ({states[s]})')
        sim = sax.pcolorfast(x, freq, mx2_plot, cmap='jet')
        if len(vm[1])==2:
            sim.set_clim(vm[1])
        plt.colorbar(sim, ax=sax, pad=0.0)
        sax.set_xlabel('Time (s)')
        sax.set_ylabel('Freq (Hz)')
        sax.set_title(f'{title2} ({states[s]})')
        plt.show()
elif not plaser:
    # for each brainstate, create a data matrix of freq x time bins x subject
    for s in istate:
        p_mx, labels = pwaves.mx3d(p_signal[s], mouse_avg, data_shape)
        if mode == 'null':
            # plot SP for spontaneous P-waves
            p_mx_plot = AS.adjust_spectrogram(np.nanmean(p_mx, axis=2)[ifreq, :], pnorm, psmooth[0])
            fig, (pax,nax) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
            pim = pax.pcolorfast(x, freq, p_mx_plot, cmap='jet')
            if len(vm[0]) == 2:
                pim.set_clim(vm[0])
            pax.set_xlabel('Time (s)')
            pax.set_ylabel('Freq (Hz)')
            pax.set_title(f'SP surrounding P-waves ({states[s]})')
            plt.colorbar(pim, ax=pax, pad=0.0)
            
            # plot SP for random control points
            n_mx, labels = pwaves.mx3d(null_pts[s], mouse_avg)
            n_mx_plot = AS.adjust_spectrogram(np.nanmean(n_mx, axis=2)[ifreq, :], pnorm, psmooth[1])
            nim = nax.pcolorfast(x, freq, n_mx_plot, cmap='jet')
            if len(vm[1]) == 2:
                nim.set_clim(vm[1])
            nax.set_xlabel('Time (s)')
            nax.set_ylabel('Freq (Hz)')
            nax.set_title(f'SP surrounding null points ({states[s]})')
            plt.colorbar(nim, ax=nax, pad=0.0)
        else:
            # plot averaged SP surrounding P-waves for dataset
            p_mx_plot = AS.adjust_spectrogram(np.nanmean(p_mx, axis=2)[ifreq, :], pnorm, psmooth)
            fig = plt.figure()
            ax = plt.gca()
            im = ax.pcolorfast(x, freq, p_mx_plot, cmap='jet')
            if len(vm) == 2:
                im.set_clim(vm)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Freq (Hz)')
            ax.set_title(f'SP surrounding P-waves ({states[s]})')
            plt.colorbar(im, ax=ax, pad=0.0)
        plt.show()

###   GET SPECTRAL PROFILES   ###
ifreq = np.arange(frange[0]*2, frange[1]*2+1)   # idx of freqs in norm. power spectrum plot
freq = np.linspace(frange[0], frange[1], len(ifreq))  # freqs

# isolate subset time windows from collected & normalized spectrograms
spon_pwaves = pwaves.get_SP_subset(spon_pwaves[1], win=collect_win, sub_win=spon_win, pnorm=pnorm)
null_pts = pwaves.get_SP_subset(null_pts[1], win=collect_win, sub_win=spon_win, pnorm=pnorm)
if plaser:
    lsr_pwaves = pwaves.get_SP_subset(lsr_pwaves[1], win=collect_win, sub_win=spon_win, pnorm=pnorm)
    success_lsr = pwaves.get_SP_subset(success_lsr[1], win=collect_win, sub_win=lsr_win, pnorm=pnorm)
    fail_lsr = pwaves.get_SP_subset(fail_lsr[1], win=collect_win, sub_win=lsr_win, pnorm=pnorm)

# collect spectral power for each event
spon_p_thetapwr = {rec:[] for rec in recordings}
null_thetapwr = {rec:[] for rec in recordings}
if plaser:
    lsr_p_thetapwr = {rec:[] for rec in recordings}
    success_lsr_thetapwr = {rec:[] for rec in recordings}
    fail_lsr_thetapwr = {rec:[] for rec in recordings}
for rec in recordings:
    # isolate mean power of desired frequencies
    spon_p_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in spon_pwaves[rec]]
    null_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in null_pts[rec]]
    if plaser:
        lsr_p_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in lsr_pwaves[rec]]
        success_lsr_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in success_lsr[rec]]
        fail_lsr_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in fail_lsr[rec]]

# create matrices of subject x frequency
spon_mx, mice = pwaves.mx2d(spon_p_thetapwr, mouse_avg)
null_mx, _ = pwaves.mx2d(null_thetapwr, mouse_avg)
if plaser:
    lsr_mx, _ = pwaves.mx2d(lsr_p_thetapwr, mouse_avg)
    success_mx, _ = pwaves.mx2d(success_lsr_thetapwr, mouse_avg)
    fail_mx, _ = pwaves.mx2d(fail_lsr_thetapwr, mouse_avg)
# smooth spectrogram
spon_mx = AS.convolve_data(spon_mx, 12, axis='x')
null_mx = AS.convolve_data(null_mx, 12, axis='x')
if plaser:
    lsr_mx = AS.convolve_data(lsr_mx, 12, axis='x')
    success_mx = AS.convolve_data(success_mx, 12, axis='x')
    fail_mx = AS.convolve_data(fail_mx, 12, axis='x')

# store data in dataframe
spon_data = [(mice[i], freq[j], spon_mx[i,j], 'spon') for j in range(len(freq)) for i in range(spon_mx.shape[0])]
null_data = [(mice[i], freq[j], null_mx[i,j], 'null') for j in range(len(freq)) for i in range(null_mx.shape[0])]
if plaser:
    lsr_data = [(mice[i], freq[j], lsr_mx[i,j], 'lsr') for j in range(len(freq)) for i in range(lsr_mx.shape[0])]
    success_data = [(mice[i], freq[j], success_mx[i,j], 'success') for j in range(len(freq)) for i in range(success_mx.shape[0])]
    fail_data = [(mice[i], freq[j], fail_mx[i,j], 'fail') for j in range(len(freq)) for i in range(fail_mx.shape[0])]
if plaser:
    df = pd.DataFrame(columns=['Mouse', 'Freq', 'Pow', 'Group'], data=spon_data+null_data+lsr_data+fail_data+success_data)
else:
    df = pd.DataFrame(columns=['Mouse', 'Freq', 'Pow', 'Group'], data=spon_data+null_data)
    
# plot normalized power spectrum
plt.figure()
sns.lineplot(data=df, x='Freq', y='Pow', hue='Group', ci=68, hue_order=['spon','null','lsr','fail'], 
             palette={'null':'darkgray', 'fail':'red', 'spon':'green', 'lsr':'blue'})
plt.xlabel('Freq (Hz)')
plt.ylabel('Norm. power')
plt.title(f'Normalized Power Spectrum (REM)\n{recordings[0]}')
plt.show()