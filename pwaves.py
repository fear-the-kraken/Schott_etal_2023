#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Functions for collecting, analyzing, and plotting P-wave data

@author: fearthekraken
"""
import sys
import os
import re
import scipy
import scipy.io as so
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pingouin as ping
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
from itertools import chain
import h5py
import pickle
import pdb
# custom modules
import sleepy
import AS

##############          SUPPORTING FUNCTIONS          ##############

def mx1d(rec_dict, mouse_avg):
    """
    Create a 1-dimensional vector of values for each trial, recording, or mouse
    @Params
    rec_dict - dictionary with recordings as keys and 1D data arrays as values
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    @Returns
    data - 1D vector of data, with 1 value per trial, recording, or mouse
    labels - list of mouse/recording names, or integers to number trials
    """
    recordings = list(rec_dict.keys())
    # average values for each recording
    if mouse_avg == 'recording':
        data = [np.nanmean(rec_dict[rec]) for rec in recordings]
        labels = recordings
    # collect all values
    elif 'trial' in mouse_avg:
        data = []
        for rec in recordings:
            _ = [data.append(i) for i in rec_dict[rec]]
        labels = np.arange(0, len(data))
    # collect all values for each mouse and average within mice
    elif mouse_avg == 'mouse':
        mdict = {rec.split('_')[0]:[] for rec in recordings}
        mice = list(mdict.keys())
        for rec in recordings:
            idf = rec.split('_')[0]
            _ = [mdict[idf].append(i) for i in rec_dict[rec]]
        data = [np.nanmean(mdict[idf]) for idf in mice]
        labels = mice
    return data, labels


def mx2d(rec_dict, mouse_avg='trial', d1_size=''):
    """
    Create a 2-dimensional matrix (subjects x time) from 1-dimensional data (time only)
    @Params
    rec_dict - dictionary with recordings as keys and lists of 1D data arrays as values
    mouse_avg - method for data averaging
               'trial'     - mx rows are individual trials from all recordings
               'recording' - mx rows are trial averages for each recording
               'mouse'     - mx rows are trial averages for each mouse
    d1_size - number of data points/time bins per trial
    @Returns
    mx - 2D data matrix (rows = trials/recordings/mice, columns = time bins)
    labels - list of mouse/recording names, or integers to number trials
    """
    recordings = list(rec_dict.keys())
    # if d1_size not provided, find a sample trial to determine number of data points
    if d1_size == '':
        for rec in recordings:
            if len(rec_dict[rec]) > 0:
                d1_size = len(rec_dict[rec][0])
                break
    if d1_size == '':
        print('No data found for any recordings')
        return np.array(()), []
    
    # create matrix of trials x time bins
    if 'trial' in mouse_avg:
        ntrials = sum([len(rec_dict[rec]) for rec in recordings])
        if ntrials > 0:
            tmx = np.zeros((ntrials, d1_size))
            row = 0
            for rec in recordings:
                for data in rec_dict[rec]:
                    tmx[row, :] = data
                    row += 1
            mx=tmx; labels=np.arange(0,tmx.shape[0])
        else:
            mx=np.array(()); labels=np.array(())
    else:
        # create matrix of recording averages x time bins
        mx_dict = mx2d_dict(rec_dict, mouse_avg, d1_size)
        if mouse_avg == 'recording':
            rmx = np.zeros((len(recordings), d1_size))
            for i,rec in enumerate(recordings):
                rmx[i,:] = np.nanmean(mx_dict[rec], axis=0)
            mx=rmx; labels=recordings
        # create matrix of mouse averages x time bins
        elif mouse_avg == 'mouse':
            mouse_dict = {rec.split('_')[0]:[] for rec in recordings}
            mice = list(mouse_dict.keys())
            mmx = np.zeros((len(mice), d1_size))
            for i,m in enumerate(mice):
                mmx[i,:] = np.nanmean(mx_dict[m], axis=0)
            mx=mmx; labels=mice
    return mx, labels
    

def mx2d_dict(rec_dict, mouse_avg, d1_size=''):
    """
    Create a 2-dimensional matrix (trials x time) from 1-dimensional data (time only) 
    for EACH recording or mouse, and return as dictionary
    @Params
    rec_dict - dictionary with recordings as keys and lists of 1D data arrays as values
    mouse_avg - method for data collecting
               'recording' - 1 dict key per recording, mx rows contain all
                             individual trials for that recording
               'mouse'     - 1 dict key per mouse, mx rows contain all
                             individual trials for that mouse (multiple recordings)
    d1_size - number of data points/time bins per trial
    @Returns
    mx_dict - dictionary (key = recording/mouse, value = mx of trials x time bins)
    """
    recordings = list(rec_dict.keys())
    # if d1_size not provided, find a sample trial to determine number of data points
    if d1_size == '':
        for rec in recordings:
            if len(rec_dict[rec]) > 0:
                d1_size = len(rec_dict[rec][0])
                break
    if d1_size == '':
        print('No data found for any recordings')
        return {}
            
    if mouse_avg == 'recording':
        mx_dict = {rec:0 for rec in recordings}  # 1 key per recording
    elif mouse_avg == 'mouse':
        mx_dict = {rec.split('_')[0]:[] for rec in recordings}  # 1 key per mouse
    
    for rec in recordings:
        idf = rec.split('_')[0]
        trials = rec_dict[rec]
        # if there are no trials for a recording, create empty mx of correct size
        if len(trials) == 0:
            rmx = np.empty((1, d1_size))
            rmx[:] = np.nan
        # if there are 1+ trials, create mx of trials x time bins
        else:
            rmx = np.zeros((len(trials), d1_size))
            for i, data in enumerate(trials):
                rmx[i, :] = data
        if mouse_avg == 'recording':
            mx_dict[rec] = rmx
        # if averaging by mouse, collect mxs for all recordings
        elif mouse_avg == 'mouse':
            mx_dict[idf].append(rmx)

    if mouse_avg == 'recording':
        return mx_dict
    elif mouse_avg == 'mouse':
        # if averaging by mouse, vertically stack trials from all recordings
        for m in mx_dict.keys():
            mmx = np.concatenate((mx_dict[m]), axis=0)
            mx_dict[m] = mmx
        return mx_dict
        
    
def mx3d(rec_dict, mouse_avg, d_shape=''):
    """
    Create a 3-dimensional matrix (freq x time x subject) from 2-dimensional data (freq x time)
    @Params
    rec_dict - dictionary with recordings as keys and lists of 2D data arrays as values
    mouse_avg - method for data averaging
               'trial'     - array layers are individual trials from all recordings
               'recording' - array layers are trial averages for each recording
               'mouse'     - array layers are trial averages for each mouse
    d_shape - 2-element tuple containing the number of rows and columns for each trial mx
    @Returns
    mx - 3D data matrix (rows=frequencies, columns=time bins, layers=trials/recordings/mice)
    labels - list of mouse/recording names, or integers to number trials
    """
    recordings = list(rec_dict.keys())
    # if d_shape not provided, find a sample trial to determine number of data points
    if len(d_shape) == 0:
        for rec in recordings:
            if len(rec_dict[rec]) > 0:
                d_shape = rec_dict[rec][0].shape
                break
    if len(d_shape) == 0:
        raise ValueError('\n\n ###   ERROR: inputted dictionary does not contain any data.   ###')
    
    # create empty 3D matrix of appropriate size
    if 'trial' in mouse_avg:
        ntrials = sum([len(rec_dict[rec]) for rec in recordings])
        mx = np.zeros((d_shape[0], d_shape[1], ntrials))
        labels = np.arange(0, ntrials)
    elif mouse_avg == 'recording':
        mx = np.zeros((d_shape[0], d_shape[1], len(recordings)))
        labels = recordings
    elif mouse_avg == 'mouse':
        mouse_dict = {rec.split('_')[0]:[] for rec in recordings}
        mice = list(mouse_dict.keys())
        mx = np.zeros((d_shape[0], d_shape[1], len(mice)))
        labels = mice
    
    layer = 0
    for rec in recordings:
        trials = rec_dict[rec]
        # if averaging by trial, each mx layer = 1 trial
        if 'trial' in mouse_avg:
            for data in trials:
                mx[:,:,layer] = data
                layer += 1
        else:
            # if there are no trials for a recording, create empty 2D mx of correct size
            if len(trials) == 0:
                rmx = np.empty((d_shape[0], d_shape[1]))
                rmx[:] = np.nan
            else:
                rmx = np.array((trials)).mean(axis=0)
            # if averaging by recording, each mx layer = 1 recording average
            if mouse_avg == 'recording':
                mx[:,:,layer] = rmx
                layer += 1
            elif mouse_avg == 'mouse':
                idf = rec.split('_')[0]
                mouse_dict[idf].append(rmx)
    if 'trial' in mouse_avg:
        return mx, labels
    elif mouse_avg == 'recording':
        return mx, labels
    # if averaging by mouse, each mx layer = 1 mouse average
    elif mouse_avg == 'mouse':
        for m in mice:
            mmx = np.nanmean(mouse_dict[m], axis=0)
            mx[:,:,layer] = mmx
            layer += 1
        return mx, labels


def build_featmx(MX, pre, post):
    """
    For each time bin in input spectrogram, collect SP for a surrounding time window,
    and reshape collected SP into 1D vectors stored in feature matrix rows
    @Params
    MX - EEG spectrogram from one recording (frequencies x time bins)
    pre, post - time window (s) to collect EEG, relative to each bin of $MX
    @Returns
    A - feature matrix with consecutive rows as reshaped SP windows, centered on
        consecutive time bins
    """
    N = MX.shape[1]  # no. of time bins
    nfreq = MX.shape[0]  # no. of frequencies

    j = 0
    # create feature mx (1 row per time bin, 1 column per freq per time bin in collection window)
    A = np.zeros((N - pre - post, nfreq * (pre + post)))

    for i in range(pre, N - post-1):
        # collect SP window surrounding each consecutive time bin
        C = MX[:, i - pre:i + post]
       
        # reshape SP window to 1D vector, store in next row of matrix $A
        A[j,:] = np.reshape(C.T, (nfreq * (pre + post),))
        j += 1
    return A


def cross_validation(S, r, theta, nfold=5):
    """
    Perform $nfold cross-validation for linear regression with power constraint (ridge regression)
    @Params
    S - stimulus matrix of trials x variables
        e.g. for spectrogram stimulus, trials = time bins and variables = frequencies
    r - response vector of measured values for each trial
    theta - list of floats, specifying candidate regularization parameters for linear
            regression model. Optimal parameter value chosen to maximize avg model performance
    nfold - no. of subsets to divide data
    @Returns
    Etest - model performance on test set (R2 values) for each power constraint param 
    Etrain - model performance on training set for each param
    """
    # no. of trials and model variables
    ntheta = len(theta)
    ninp = r.shape[0]
    nsub = round(ninp / nfold)
    nparam = S.shape[1]

    # normalize variable values to mean of zero
    for i in range(nparam):
        S[:,i] -= S[:,i].mean()
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
            k = ridge_regression(S[q,:], r[q], th)
            pred_test  = np.dot(S[p,:], k)  # predicted values
            pred_train = np.dot(S[q,:], k)
            rtest  = r[p]  # measured values
            rtrain = r[q]
            # model performance
            ptest[j] = 1 - np.var(pred_test - rtest) / np.var(rtest)
            ptrain[j] = 1 - np.var(pred_train - rtrain) / np.var(rtrain)
            j += 1
        Etest[l] = np.mean(ptest)
        Etrain[l] = np.mean(ptrain)
        l += 1
    return Etest, Etrain


def ridge_regression(A, r, theta):
    """
    Estimate linear filter to optimally predict neural activity from EEG spectrogram
    @Params
    A - feature matrix; norm. power value for SP frequencies * relative times * trials
    r - response vector; measured neural signal for each trial
    theta - regularization/power constraint parameter 
    @Returns
    k - coefficient matrix optimally relating measured signal to EEG features (freqs * times)
    """
    S = A.copy()
    n = S.shape[1]
    
    AC = np.dot(S.T, S)  # input matrix SP
    AC = AC + np.eye(n)*theta  # power constraint
    SR = np.dot(S.T, r)  # output matrix SP*signal
    
    # linear solution optimally approximating S * k = r
    k = scipy.linalg.solve(AC, SR)
    return k


def df_from_timecourse_dict(tdict_list, mice_list, dose_list, virus_list=[], state_cols=[]):
    """
    Create dataframe from equivalent-length lists of timecourse dictionaries 
    (key = brain state, value = 2D matrix of mice x time bins)
    @Params
    tdict_list - list of dictionaries, each with brain states as keys and timecourse
                  matrices as values; each drug dose has 1 dictionary
    mice_list - list of mouse names, corresponding to the rows in each matrix
    dose_list - list of drug doses, corresponding to each dictionary
    virus_list - optional list of viral vectors, corresponding to each dictionary
    state_cols - list of brain state labels (e.g. ['REM', 'Wake', 'NREM']), corresponding
                 to the keys in each dictionary (e.g. [1,2,3])
    @Returns
    df - dataframe with info columns for mouse, virus, dose, and brain state, and
                        data columns for each time bin
         e.g. 'mouse'  'virus' 'dose'  'state'  't0'  't1'  't2'
               'Bob'    hm3dq   0.25    'REM'    0.2   0.4   0.6
    """
    if len(virus_list) != len(tdict_list):
        virus_list = ['AAV']*len(tdict_list)
    # create labels for each mx column/time bin (e.g. column 0 --> t0)
    state_cols = list(tdict_list[0].keys())
    times = ['t'+str(i) for i in range(tdict_list[0][state_cols[0]].shape[1])]
    
    df = pd.DataFrame(columns=['mouse', 'virus', 'dose', 'state']+times)
    for (tdict, mice, virus, dose) in zip(tdict_list, mice_list, virus_list, dose_list):
        # create mini-dataframe for each brain state
        for state in state_cols:
            state_df = pd.DataFrame(columns=['mouse', 'virus', 'dose', 'state']+times)
            # for each subject, add mouse name, drug dose, and brain state
            state_df['mouse'] = mice
            state_df['virus'] = [virus]*len(mice)
            state_df['dose'] = [dose]*len(mice)
            state_df['state'] = [state]*len(mice)
            # add data values for each subject in each time bin
            for col,t in enumerate(times):
                state_df[t] = tdict[state][:,col]
            df = pd.concat([df, state_df], axis=0, ignore_index=True)
    return df


def df_from_rec_dict(rec_dict, val_label=''):
    """
    Create dataframe from dictionary of recording data
    @Params
    rec_dict - dictionary with recording names as keys and 1D data arrays as values
    val_label - optional label for data in $rec_dict
    @Returns
    df - dataframe with rows containing mouse name, recording name, and data value
    """
    if not val_label:
        val_label = 'value'
    recordings = list(rec_dict.keys())
    data = []
    for rec in recordings:
        idf = rec.split('_')[0]
        for trial_val in rec_dict[rec]:
            data.append((idf, rec, trial_val))
    df = pd.DataFrame(columns=['mouse', 'recording', val_label], data=data)
    return df


def get_iwins(win, sr, inclusive=True):
    """
    Calculate start (iwin1) and stop (iwin2) indices of data collection, relative 
    to an event at idx=0
    @Params
    win - float or 2-element list, specifying the time window (s) to collect data
          e.g. float=0.5 --> collect interval from -0.5 to +0.5 s surrounding event
          e.g. float=[-1.0, 0.2] --> collect interval from -1.0 to +0.2 s surrounding event
    sr - sampling rate (Hz)
    inclusive - if True, include the last sample in the time window
    @Returns
    iwin1, iwin2 - number of data samples before and after the event to collect
    """
    # if $win has 2 elements, convert both to number of samples (abs. value)
    if (isinstance(win, list) or isinstance(win, tuple)):
        if len(win) == 2:
            iwin1 = int(round(np.abs(win[0])*sr))
            iwin2 = int(round(win[1]*sr))
            # include last idx in $win unless it's zero
            if inclusive and win[1] != 0:
                iwin2 += 1
    # if $win is a NEGATIVE number, collect iwin1 samples BEFORE the event
    elif win < 0:
        iwin1 = int(round(np.abs(win)*sr))
        iwin2 = 0
    # if $win is a POSITIVE number, collect iwin2 samples AFTER the event
    elif win > 0:
        iwin1 = 0
        iwin2 = int(round(win*sr)) + 1
    return iwin1, iwin2


def jitter_laser(lsr, jitter, sr):
    """
    Create a sham laser vector with randomly "jittered" stimulations (step pulses only!)
    @Params
    lsr - laser stimulation vector
    jitter - randomly shift indices of true laser pulses by up to +/- $jitter seconds
    sr - sampling rate (Hz)
    @Returns
    jlsr - sham laser vector with randomly jittered pulses
    """
    np.random.seed(0)
    # get the first idx of each true laser pulse, and the pulse duration
    lsr_seq = sleepy.get_sequences(np.where(lsr==1)[0])
    pulse_bins = len(lsr_seq[0])
    laser_start = [seq[0] for seq in lsr_seq]
    
    # get random indices in range -$jitter to +$jitter s, and apply to laser onset indices
    jter = np.random.randint(-jitter*sr, jitter*sr, len(laser_start))
    jidx = laser_start + jter
    
    # create new sham laser vector from jittered laser indices
    jlsr = np.zeros((len(lsr)))
    for ji in jidx:
        # make sure each pulse fits within the laser vector
        if ji < 0:
            ji += jitter*sr
        elif ji > len(lsr) - pulse_bins:
            ji = ji - jitter*sr - pulse_bins
        # fill in jittered laser pulses
        jlsr[ji : ji + pulse_bins] = 1
    
    return jlsr


def spike_threshold(data, th, sign=1):
    """
    Detect potential spikes as waveforms crossing the given threshold $th
    @Params
    data - spiking raw data
    th - threshold to pass to qualify as a spike
    sign - direction of detected spikes
            1 - spikes are "valleys" below signal
           -1 - spikes are "mountains" above signal
    @Returns
    sidx -  indices of spike waveforms
    """
    if sign == 1:
        lidx = np.where(data[0:-2] > data[1:-1])[0]  # < previous point
        ridx = np.where(data[1:-1] <= data[2:])[0]   # <= following point
        thidx = np.where(data[1:-1] < (-1 * th))[0]  # < th
        sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx)) + 1
    else:
        lidx = np.where(data[0:-2] < data[1:-1])[0]
        ridx = np.where(data[1:-1] >= data[2:])[0]
        thidx = np.where(data[1:-1] > th)[0]
        sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx)) + 1
        
    sidx = sidx.astype('int32')
        
    return sidx


def detect_noise(LFP, sr, thres=800, win=5, sep=0):
    """
    Return indices of LFP signal within detected "noise" windows
    @Params
    LFP - LFP signal
    sr - sampling rate (Hz)
    thres - threshold for detecting LFP noise (uV, absolute value)
            * A 2-element list [x1, x2] specifies positive (x1) and negative (x2) threshold
    win - size of noise window (s) surrounding each threshold crossing
    sep - combine noise windows within X s of each other into one sequence
    @Returns
    noise_idx - indices corresponding to LFP noise
    """
    if type(thres) not in [list,tuple]:
        thres = [thres]
    if len(thres)==2:
        # use different thresholds for positive & negative LFP fluctuations
        noise1 = np.where(LFP > thres[0])[0]
        noise2 = np.where(LFP < -thres[1])[0]
        noise = np.sort(np.unique(np.concatenate((noise1,noise2))))
    elif len(thres)==1:
        # use the same threshold for all LFP fluctuations (+/-)
        noise = np.where(np.abs(LFP) > thres[0])[0]
    
    noise_idx = np.array(())
    while len(noise) > 0:
        # collect $win s window surrounding each instance of LFP noise
        pre = 0 if (noise[0] - sr*win/2) < 0 else int(noise[0] - sr*win/2)
        if (noise[0] + sr*win/2) >= len(LFP):
            noise_win = np.arange(pre, len(LFP))
        else:
            noise_win = np.arange(pre, int(noise[0] + sr*win/2))
        # update noise indices, continue to next detected noise outside window
        noise_idx = np.concatenate((noise_idx, noise_win))
        noise = np.setdiff1d(noise, noise_win)
    noise_idx = np.sort(np.unique(noise_idx))
    if sep>0:
        # fill in gaps of $ibreak indices between neighboring noise sequences
        ibreak = int(round(sep*sr))
        noise_seqs = sleepy.get_sequences(noise_idx, ibreak)
        if noise_seqs[0].size>0:
            noise_seqs = [np.arange(nseq[0], nseq[-1]+1) for nseq in noise_seqs]
            noise_idx = np.concatenate(noise_seqs)
    
    return noise_idx

	
##############          DATA COLLECTION FUNCTIONS          ##############

def get_SP_subset(rec_dict, win, sub_win, pnorm, d_shape=''):
    """
    Isolate specified time window from each spectrogram in data dictionary
    @Params
    rec_dict - data dictionary (key=recording name, value=list of SPs;
               freq x time bins)
    win - time window (s) of SPs in $rec_dict
    sub_win - time window (s) indicating subset of SP columns to isolate
    pnorm - if > 0, normalize frequencies in original SP by their mean values
    d_shape - 2-element tuple with no. of rows and columns for each SP
    @Returns
    rec_dict - data dictionary with the isolated SP windows specified by $sub_win
    """
    recordings = list(rec_dict.keys())
    # if d_shape not provided, find a sample SP to determine number of data points
    if len(d_shape) == 0:
        for rec in recordings:
            if len(rec_dict[rec]) > 0:
                d_shape = rec_dict[rec][0].shape
                break
    # get indices of SP columns/time points to isolate
    t = np.linspace(win[0], win[1], d_shape[1])
    sub_idx = np.intersect1d(np.where(t>=sub_win[0])[0], np.where(t<=sub_win[1])[0])
    for rec in recordings:
        # isolate and normalize subset from each SP
        sub_sps = [AS.adjust_spectrogram(sp, pnorm=pnorm, psmooth=0)[:, sub_idx] for sp in rec_dict[rec]]
        rec_dict[rec] = sub_sps
    return rec_dict


def get_lsr_phase(ppath, recordings, istate, bp_filt, min_state_dur=5, ma_thr=20, 
                  ma_state=3, flatten_is=False, post_stim=0.1, psave=False):
    """
    Get instantaneous phase for each P-wave and laser pulse during any brain state
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state to analyze
    bp_filt - 2-element list specifying the lowest and highest frequencies to use
              for bandpass filtering
    min_state_dur - minimum brain state duration (min) to be included in analysis
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if isinstance(istate, list):
        istate = istate[0]
    
    lsr_pwaves = {rec:[] for rec in recordings}  # phase of laser-triggered P-waves
    spon_pwaves = {rec:[] for rec in recordings}  # phase of spontaneous P-waves
    success_lsr = {rec:[] for rec in recordings}  # phase of successful laser pulses
    fail_lsr = {rec:[] for rec in recordings}  # phase of failed laser pulses
    
    # collect phase vector for each state sequence
    all_phases = {rec:[] for rec in recordings}
    success_latencies = {rec:[] for rec in recordings}
    
    # collect raw LFPs associated with instantaneous phases 
    lsr_p_lfp = {rec:[] for rec in recordings}
    spon_p_lfp = {rec:[] for rec in recordings}
    amp_lfp =  {rec:[] for rec in recordings}
    
    for rec in recordings:
        print('Getting P-waves for ' + rec + ' ...')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
               
        # load EEG, LFP, and P-wave indices
        EEG = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        LFP, p_idx = load_pwaves(ppath, rec)[0:2]
        
        # load laser and find laser-triggered P-waves
        lsr = sleepy.load_laser(ppath, rec)
        lsr_p_idx, spon_p_idx, lsr_yes_p, lsr_no_p = get_lsr_pwaves(p_idx, lsr, post_stim)
        
        # create event vectors to match EEG indices
        lvec = np.zeros((len(EEG)))
        lvec[lsr_p_idx] = 1
        svec = np.zeros((len(EEG)))
        svec[spon_p_idx] = 1
        yvec = np.zeros((len(EEG)))
        yvec[lsr_yes_p] = 1
        nvec = np.zeros((len(EEG)))
        nvec[lsr_no_p] = 1
        
        # get brain state sequences
        sseq = sleepy.get_sequences(np.where(M==istate)[0])
        
        for seq in sseq:
            if len(seq) > min_state_dur/2.5:
                # get EEG data for seq
                sidx = int(round(seq[0]*nbin))+20
                eidx = int(round(seq[-1]*nbin))
                seq_eeg = EEG[sidx : eidx]
                seq_lfp = LFP[sidx : eidx]
                
                # bandpass filter EEG
                a1 = float(bp_filt[0]) / (sr/2.0)
                a2 = float(bp_filt[1]) / (sr/2.0)
                seq_eeg = sleepy.my_bpfilter(seq_eeg, a1, a2)
                
                res = scipy.signal.hilbert(seq_eeg)  # hilbert transform
                iphase = np.angle(res)  # get instantaneous phase
                
                # get event indices in seq
                li, si, yi, ni = [np.where(vec[sidx:eidx])[0] for vec in [lvec, svec, yvec, nvec]]
                lsr_i = np.concatenate((yi, ni))
                # get idx of first P-wave triggered by each successful lsr pulse
                fli = [li[np.where(li>=y)[0][0]] for y in yi]
                tri = [yi[np.where(yi<=l)[0][-1]] for l in li]
                yi = [y for y,f in zip(yi,fli) if f-y > 12]
                li = [l for l,t in zip(li,tri) if l-t > 12]
                
                # get event phases
                lphase, sphase, yphase, nphase, lsrphase = [iphase[i] for i in [li, si, yi, ni, lsr_i]]
                _ = [lsr_pwaves[rec].append(lp) for lp in lphase]
                _ = [spon_pwaves[rec].append(sp) for sp in sphase]
                _ = [success_lsr[rec].append(yp) for yp in yphase]
                _ = [fail_lsr[rec].append(np) for np in nphase]
                # get tuples of (lsr pulse phase, LFP amplitude in post_stim seconds)
                _ = [lsr_p_lfp[rec].append((iphase[i], seq_lfp[i-50 : i+50])) for i in li]
                _ = [spon_p_lfp[rec].append((iphase[i], seq_lfp[i-50 : i+50])) for i in si]
                _ = [amp_lfp[rec].append((iphase[l], seq_lfp[l : l + int(post_stim*sr)])) for l in lsr_i]
    
    # save files
    if psave:
        filename = psave if isinstance(psave, str) else f'lsrSurround_LFP'
        so.savemat(os.path.join(ppath, f'{filename}_lsr_phase_{istate}.mat'), lsr_pwaves)
        so.savemat(os.path.join(ppath, f'{filename}_spon_phase_{istate}.mat'), spon_pwaves)
        so.savemat(os.path.join(ppath, f'{filename}_success_phase_{istate}.mat'), success_lsr)
        so.savemat(os.path.join(ppath, f'{filename}_fail_phase_{istate}.mat'), fail_lsr)
        
        so.savemat(os.path.join(ppath, f'{filename}_all_phase_{istate}.mat'), all_phases)
        so.savemat(os.path.join(ppath, f'{filename}_phase_lfp_{istate}.mat'), amp_lfp)
        so.savemat(os.path.join(ppath, f'{filename}_lsr_p_lfp_{istate}.mat'), lsr_p_lfp)
        so.savemat(os.path.join(ppath, f'{filename}_spon_p_lfp_{istate}.mat'), spon_p_lfp)
        
    return lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, all_phases, success_latencies, amp_lfp, lsr_p_lfp, spon_p_lfp


def get_surround(ppath, recordings, istate, win, signal_type, recalc_highres=False,
                 tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=False, 
                 exclude_noise=False, nsr_seg=2, perc_overlap=0.95, null=False, 
                 null_win=[0.5,0.5], p_iso=0, pcluster=0, clus_event='waves', psave=False):
    """
    Collect raw signal surrounding P-wave events
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    win: time window (s) to collect data relative to the event
    signal_type: specifies the type of data to collect 
                'EEG', 'EEG2'                   --> raw hippocampal or prefrontal EEG
                'EMG'                           --> raw nuchal EMG
                'SP', 'SP2'                     --> hippocampal or prefrontal SP
                'SP_NORM', 'SP2_NORM'           --> norm. hippocampal or prefrontal SP
                'SP_CALC', 'SP2_CALC'           --> calculate each SP using surrounding EEG
                'SP_CALC_NORM', 'SP2_CALC_NORM' --> normalize calculated SP by whole SP mean
                'LFP'                           --> processed LFP signal
    recalc_highres - if True, recalculate high-resolution spectrogram from EEG, 
                      using $nsr_seg and $perc_overlap params
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    exclude_noise - if False, ignore manually annotated LFP noise indices
                    if True, exclude time bins containing LFP noise from analysis
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    null - if True, also collect data surrounding randomized control points in $istate
    null_win - if > 0, qualifying "null" points must be free of P-waves and laser pulses in surrounding $null_win interval (s)
               if = 0, "null" points are randomly selected from all state indices
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    @Returns
    p_signal    - dictionary with brain states as keys, and sub-dictionaries as values
                   Sub-dictionaries have mouse recordings as keys, with lists of 2D or 3D signals as values 
                   Signals (SPs, EEGs, or LFPs) represent the time window ($win s) surrounding each P-wave
    null_signal - dictionary structured as above, but containing signals surrounding each
                   randomly selected control point
    data.shape  - tuple with shape of the data from one trial 
    """
    import time
    START = time.perf_counter()
    
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    
    if len(istate) == 0:
        istate = ['total']
        brstate = 'total'
    p_signal = {s:{rec:[] for rec in recordings} for s in istate}  # signal surrounding P-waves
    null_signal = {s:{rec:[] for rec in recordings} for s in istate}  # signal surrounding randomized time points
    
    for rec in recordings:
        print('Getting P-waves for ' + rec + ' ...')
        
        p_signal = {s:[] for s in istate}  # signal surrounding P-waves
        null_signal = {s:[] for s in istate}  # signal surrounding randomized time points
        
        # load sampling rate for LFP (may be Intan or Neuropixel)
        sr = AS.get_snr_pwaves(ppath, rec, default='NP')
        nbin = int(np.round(sr) * 2.5)  # number of NP samples per 2.5 s brain state bin
        dt = (1.0 / sr) * nbin          # 
        
        # number of LFP samples in pre/post-event collection window
        iwin1, iwin2 = get_iwins(win, sr)
        
        # EEG = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        # if os.path.exists(os.path.join(ppath, rec, 'EEG2.mat')):
        #     EEG2 = so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'), squeeze_me=True)['EEG2']
        
        # # load EMG
        # if os.path.exists(os.path.join(ppath, rec, 'EMG.mat')):
        #     EMG = so.loadmat(os.path.join(ppath, rec, 'EMG.mat'), squeeze_me=True)['EMG']
        #     EMG = sleepy.my_hpfilter(EMG, 0.01)  # high-pass filter at 5Hz
        
        # load or calculate entire high-res spectrogram
         # SP calculated using EEG2
        if ('SP2' in signal_type) and (signal_type != 'SP2_CALC'):
            # SP, f, t, sp_nbin, sp_dt = AS.highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
            #                                                      recalc_highres=recalc_highres, mode='EEG', peeg2=True)
            # #sp_nbin = len(EEG) / SP.shape[1]
            # sp_win1 = int(round(iwin1/sp_nbin))
            # sp_win2 = int(round(iwin2/sp_nbin))
            SPEC = AS.highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                          recalc_highres=recalc_highres, mode='EEG', peeg2=True)
            
            SP, f, t = SPEC[0:3]  # high-res SP, freqs, timestamps
            sp_dt = t[1] - t[0]   # seconds per bin in SP
            sp_nbin = sp_dt * sr  # Intan/NP samples per bin in SP
            sp_win1 = int(round(iwin1/sp_nbin))  # number of SP bins in pre/post-event window
            sp_win2 = int(round(iwin2/sp_nbin))
            #cf = nbin/spnbin    # conversion factor from $M to #SP time resolution
            
        # SP calculated using EEG
        elif ('SP' in signal_type) and (signal_type != 'SP_CALC'):
            # SP, f, t, sp_nbin, sp_dt = AS.highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
            #                                                      recalc_highres=recalc_highres, mode='EEG', peeg2=False)
            # sp_win1 = int(round(iwin1/sp_nbin))
            # sp_win2 = int(round(iwin2/sp_nbin))
            SPEC = AS.highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                          recalc_highres=recalc_highres, mode='EEG', peeg2=False)
            
            SP, f, t = SPEC[0:3]  # high-res SP, freqs, timestamps
            sp_dt = t[1] - t[0]   # seconds per bin in SP
            sp_nbin = sp_dt * sr  # Intan/NP samples per bin in SP
            sp_win1 = int(round(iwin1/sp_nbin))  # number of SP bins in pre/post-event window
            sp_win2 = int(round(iwin2/sp_nbin))
            #cf = nbin/spnbin    # conversion factor from $M to #SP time resolution
        
        # calculate SP mean
        if '_NORM' in signal_type:
            SP_mean = SP.mean(axis=1)
            SP_norm = np.divide(SP, np.repeat([SP_mean], SP.shape[1], axis=0).T)  # normalize entire spectrogram
        
        
        # load EEG and EEG2
        EEG = AS.load_eeg_emg(ppath, rec, 'EEG')
        EEG2 = AS.load_eeg_emg(ppath, rec, 'EEG2')
        EMG = AS.load_eeg_emg(ppath, rec, 'EMG')
        
        # load Intan sampling rate for EEG/EMG
        intan_sr = sleepy.get_snr(ppath, rec)
        
        intan_dt = 1/intan_sr       # number of seconds per Intan bin
        intan_nbin = intan_dt * sr  # number of NP samples per Intan bin
        intan_win1 = int(round(iwin1/intan_nbin))  # number of Intan bins in pre/post-event window
        intan_win2 = int(round(iwin2/intan_nbin))  #### may be the same as NP bins, if the sampling rates are the same
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
               
        # load LFP and P-wave indices
        if exclude_noise:
            # load noisy LFP/EMG indices, replace with NaNs
            LFP, p_idx, lfp_noise_idx, emg_noise_idx = load_pwaves(ppath, rec, return_lfp_noise=True, 
                                                                   return_emg_noise=True)[0:4]
            LFP[lfp_noise_idx] = np.nan
            EMG[emg_noise_idx] = np.nan
            #p_idx = np.setdiff1d(p_idx, noise_idx)
        else:
            LFP, p_idx = load_pwaves(ppath, rec)[0:2]
        # isolate single or clustered P-waves
        if p_iso and pcluster:
            print('ERROR: cannot accept both p_iso and pcluster arguments')
            return
        elif p_iso:
            p_idx = get_p_iso(p_idx, sr, win=p_iso)
        elif pcluster:
            p_idx = get_pclusters(p_idx, sr, win=pcluster, return_event=clus_event)
        
        # define start and end points of analysis
        istart = int(np.round(tstart*sr))
        if tend == -1:
            iend = len(LFP) - 1
        else:
            iend = int(np.round(tend*sr))
            
        # find P-wave index in EEG/EMG signal
        if signal_type in ['EEG', 'EEG2', 'EMG']:
            p_idx_trsl = np.round(np.divide(p_idx, intan_nbin)).astype('int32')
            
        # find P-wave index in EEG, calculate surrounding SP
        elif signal_type in ['SP_CALC','SP2_CALC','SP_CALC_NORM','SP2_CALC_NORM']:
            p_idx_trsl = np.round(np.divide(p_idx, intan_nbin)).astype('int32')
        
        # find P-wave index in high-resolution SP
        elif signal_type in ['SP', 'SP2', 'SP_NORM', 'SP2_NORM']:
            # adjust Intan/NP idx to properly translate to SP idx
            #spi_adjust = np.linspace(-sr, sr, len(LFP))
            #spi = int(round((pi + spi_adjust[pi])/sp_nbin))
            #p_idx_trsl = np.round(np.divide((p_idx + spi_adjust[p_idx]), sp_nbin)).astype('int32')
            p_idx_trsl = np.round(np.divide(p_idx-sr, sp_nbin)).astype('int32')
        
        # P-wave index refers directly to LFP signal
        else:
            p_idx_trsl = np.array(p_idx)
            
        # translate P-wave indices to their corresponding locations in target signal
        for pi, pi_trsl in zip(p_idx,p_idx_trsl):
            
            if pi >= iwin1 and pi + iwin2 < len(LFP) and istart <= pi <= iend:
                
                if istate[0] != 'total':
                    try:
                        brstate = int(M[int(pi/nbin)])
                    except:
                        pdb.set_trace()
                
                if signal_type == 'EEG':
                    data = EEG[pi_trsl - intan_win1 : pi_trsl + intan_win2]
                elif signal_type == 'EEG2':
                    data = EEG2[pi_trsl - intan_win1 : pi_trsl + intan_win2]
                elif signal_type == 'EMG':
                    data = EMG[pi_trsl - intan_win1 : pi_trsl + intan_win2]
                    
                # calculate SP from EEG or EEG2
                elif 'CALC' in signal_type:
                    if 'SP2' in signal_type:
                        tmp = EEG2[pi_trsl - intan_win1 : pi_trsl + intan_win2]
                    elif 'SP' in signal_type:
                        tmp = EEG[pi_trsl - intan_win1 : pi_trsl + intan_win2]
                    f, t, data = scipy.signal.spectrogram(tmp, fs=intan_sr, window='hanning', 
                                                          nperseg=int(nsr_seg*intan_sr), noverlap=int(nsr_seg*intan_sr*perc_overlap))
                    # normalize calculated SP based on entire recording
                    if 'NORM' in signal_type:
                        data = np.divide(data, np.repeat([SP_mean], data.shape[1], axis=0).T)
                # if not calculating, get SP or SP_NORM from whole recording calculation
                elif 'SP' in signal_type:
                    #spi = int(round((pi + spi_adjust[pi])/sp_nbin))
                    if pi_trsl >= sp_win1 and pi_trsl + sp_win2 < SP.shape[1]:
                        if 'NORM' in signal_type:
                            data = SP_norm[:, pi_trsl-sp_win1 : pi_trsl+sp_win2]
                        else:
                            data = SP[:, pi_trsl-sp_win1 : pi_trsl+sp_win2]
                    else:
                        data = np.array(())
                elif signal_type == 'LFP':
                    data = LFP[pi-iwin1 : pi+iwin2]
                else:
                    print(signal_type + ' IS AN INVALID SIGNAL TYPE')
                    return
                    
                # collect data in relevant dictionary
                if brstate in istate:
                    if data.size > 0:
                        p_signal[brstate].append(data)
                    
        # collect signals surrounding random control time points
        if null:
            null_iwin1, null_iwin2 = get_iwins(null_win, sr)
            # sample "null" REM epochs with no P-waves/laser pulses
            if null_win != 0:
                # find all points that don't qualify as "null"
                not_null_idx = np.zeros((10000000))
                for i, pi in enumerate(p_idx):
                    p_win = np.arange(pi-null_iwin1, pi+null_iwin2)
                    not_null_idx[i*len(p_win) : i*len(p_win)+len(p_win)] = p_win
                # get rid of trailing zeros (computational efficiency)
                not_null_idx = np.trim_zeros(not_null_idx, 'b')
            
            for s in istate:
                if istate[0] != 'total':
                    # get array of all possible indices in state s
                    sseq = sleepy.get_sequences(np.where(M==s)[0])
                    sseq_idx = [np.arange(seq[0]*nbin, seq[-1]*nbin+nbin) for seq in sseq]
                    #sseq_idx = np.array((list(chain.from_iterable(sseq_idx))))
                    sseq_idx = np.concatenate(sseq_idx)
                    sseq_idx = [sidx for sidx in sseq_idx if sidx > iwin1 and sidx < len(LFP)-iwin2 and istart < sidx < iend]
                else:
                    sseq_idx = np.arange(iwin1, len(LFP)-iwin2)
                # keep only state indices that are not next to a P-wave/laser pulse
                if null_win != 0:
                    sseq_idx = np.setdiff1d(sseq_idx, not_null_idx)
                # randomly select from all state indices
                else:
                    sseq_idx = np.array((sseq_idx))
                np.random.seed(0)
                # select number of random indices matching the number of P-waves
                r_idx = np.random.randint(low=0, high=len(sseq_idx), size=len(p_signal[s]))
                null_idx = sseq_idx[r_idx]
                
                null_idx_trsl = np.round(np.divide(null_idx_trsl, divisor)).astype('int32')
                for ni,ni_trsl in zip(null_idx, null_idx_trsl):
                    # get data of desired signal type
                    if signal_type == 'EEG':
                        data = EEG[ni_trsl-intan_win1 : ni_trsl+intan_win2]
                    elif signal_type == 'EEG2':
                        data = EEG2[ni_trsl-intan_win1 : ni_trsl+intan_win2]
                    elif signal_type == 'EMG':
                        data = EMG[ni_trsl-intan_win1 : ni_trsl+intan_win2]
                    # calculate SP from EEG or EEG2
                    elif 'CALC' in signal_type:
                        if 'SP2' in signal_type:
                            tmp = EEG2[ni_trsl-intan_win1 : ni_trsl+intan_win2]
                        elif 'SP' in signal_type:
                            tmp = EEG[ni_trsl-intan_win1 : ni_trsl+intan_win2]
                        f, t, data = scipy.signal.spectrogram(tmp, fs=intan_sr, window='hanning', 
                                                              nperseg=int(nsr_seg * intan_sr), noverlap=int(nsr_seg * intan_sr * perc_overlap))
                        # normalize calculated SP based on entire recording
                        if 'NORM' in signal_type:
                            data = np.divide(data, np.repeat([SP_mean], data.shape[1], axis=0).T)
                    # if not calculating, get SP or SP_NORM from whole recording calculation
                    elif 'SP' in signal_type:  
                        #spi = int(round((ni + spi_adjust[ni])/sp_nbin))
                        if 'NORM' in signal_type:
                            data = SP_norm[:, ni_trsl-sp_win1 : ni_trsl+sp_win2]
                        else:
                            data = SP[:, ni_trsl-sp_win1 : ni_trsl+sp_win2]
                    elif signal_type == 'LFP':
                        data = LFP[ni-iwin1 : ni+iwin2]
                    else:
                        print(signal_type + ' IS AN INVALID SIGNAL TYPE')
                        return
                    # collect data in null dictionary
                    null_signal[s].append(data)
        
        # save tmp files to free up more room for computation
        for s in istate:
            try:
                so.savemat(f'TMP_{rec}_pwaves_{s}.mat', {'data':p_signal[s]})
                so.savemat(f'TMP_{rec}_null_{s}.mat', {'data':null_signal[s]})
            except:
                pdb.set_trace()
    
    if psave:
        print('\n Assembling data dictionaries and saving .mat files ...\n')
    else:
        print('\n Assembling data dictionaries ...\n')
        
    # collect data from all recordings for each state from tmp files
    p_signal = {s:{rec:0 for rec in recordings} for s in istate}
    null_signal = {s:{rec:0 for rec in recordings} for s in istate}
    
    # assemble final data dictionaries
    for s in istate:
        for rec in recordings:
            p_signal[s][rec] = so.loadmat(f'TMP_{rec}_pwaves_{s}.mat')['data']
            null_signal[s][rec] = so.loadmat(f'TMP_{rec}_null_{s}.mat')['data']
        # remove temporary files
        for rec in recordings:
            os.remove(f'TMP_{rec}_pwaves_{s}.mat')
            os.remove(f'TMP_{rec}_null_{s}.mat')
    # save files
    data_shape = list(p_signal[s].values())[0][0].shape
    if psave:
        for s in istate:
            filename = psave if isinstance(psave, str) else f'Surround_{signal_type}'
            so.savemat(os.path.join(ppath, f'{filename}_pwaves_{s}.mat'), p_signal[s])
            so.savemat(os.path.join(ppath, f'{filename}_null_{s}.mat'), null_signal[s])
            so.savemat(os.path.join(ppath, f'{filename}_data_shape.mat'), {'data_shape':data_shape})
    
    END = time.perf_counter()
    print(f'COMPUTING TIME --> {END-START:0.2f} seconds ({len(recordings)} recordings, {len(istate)} brainstates, signal type = {signal_type})')
    return p_signal, null_signal, data_shape


def get_lsr_surround(ppath, recordings, istate, win, signal_type, recalc_highres=False,
                     tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=False, 
                     exclude_noise=False, nsr_seg=2, perc_overlap=0.95, null=False, 
                     null_win=[0.5,0.5], null_match='spon', post_stim=0.1, lsr_iso=0, 
                     lsr_win=[], p_iso=0, pcluster=0, clus_event='waves', psave=False):
    """
    Collect raw signal surrounding spontaneous and laser-triggered P-waves, and
    successful and failed laser pulses
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    win - time window (s) to collect data relative to P-waves and randomized points
    signal_type: specifies the type of data to collect 
                'EEG', 'EEG2'                   --> raw hippocampal or prefrontal EEG
                'EMG'                           --> raw EMG
                'SP', 'SP2'                     --> hippocampal or prefrontal SP
                'SP_NORM', 'SP2_NORM'           --> norm. hippocampal or prefrontal SP
                'SP_CALC', 'SP2_CALC'           --> calculate each SP using surrounding EEG
                'SP_CALC_NORM', 'SP2_CALC_NORM' --> normalize calculated SP by whole SP mean
                'LFP'                           --> processed LFP signal
    recalc_highres - if True, recalculate high-resolution spectrogram from EEG, 
                      using $nsr_seg and $perc_overlap params
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    exclude_noise - if False, ignore manually annotated LFP/EMG noise indices
                    if True, exclude time bins containing LFP/EMG noise from analysis
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    null - if True, also collect data surrounding randomized control points in $istate
    null_win - if > 0, qualifying "null" points must be free of P-waves and laser pulses in surrounding $null_win interval (s)
               if = 0, "null" points are randomly selected from all state indices
    null_match - the no. of random control points is matched with the no. of some other event type
                 'spon'    - # control points equals the # of spontaneous P-waves
                 'lsr'     - # control points equals the # of laser-triggered P-waves
                 'success' - # control points equals the # of successful laser pulses
                 'fail'    - # control points equals the # of failed laser pulses
                 'all lsr' - # control points equals the # of total laser pulses
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    lsr_iso - if > 0, do not analyze laser pulses that are preceded within $lsr_iso s by 1+ P-waves
    lsr_win - time window (s) to collect data relative to successful and failed laser pulses
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    @Returns
    lsr_pwaves  - dictionary with brain states as keys, and sub-dictionaries as values
                   Sub-dictionaries have mouse recordings as keys, with lists of 2D or 3D signals as values 
                   Signals (SPs, EEGs, or LFPs) represent the time window ($win s) surrounding each laser-triggered P-wave
    spon_pwaves - signals surrounding each spontaneous P-wave
    success_lsr - signals surrounding each successful laser pulse
    fail_lsr    - signals surrounding each failed laser pulse
    null_pts    - signals surrounding each randomly selected control point             
    data.shape  - tuple with shape of the data from one trial 
    """

    import time
    START = time.perf_counter()
    
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    
    if len(istate) == 0:
        istate = ['total']
        brstate = 'total'
    
    # no. of laser pulses not collected due to preceding P-waves
    lsr_elim_counts = {'success':0, 'fail':0}
    
    for rec in recordings:
        print('Getting P-waves for ' + rec + ' ...')
        
        lsr_pwaves = {s:[] for s in istate}  # signal surrounding laser-triggered P-waves
        spon_pwaves = {s:[] for s in istate}  # signal surrounding spontaneous P-waves
        success_lsr = {s:[] for s in istate}  # signal surrounding successful laser pulses
        fail_lsr = {s:[] for s in istate}  # signal surrounding failed laser pulses
        null_pts = {s:[] for s in istate}  # signal surrounding random control points
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        # get time windows for P-waves and laser pulses
        iwin1, iwin2 = get_iwins(win, sr)
        if len(lsr_win) == 2:
            lsr_iwin1, lsr_iwin2 = get_iwins(lsr_win, sr)
        else:
            lsr_iwin1 = iwin1; lsr_iwin2 = iwin2
        
        # load EEG and EEG2
        EEG = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        if os.path.exists(os.path.join(ppath, rec, 'EEG2.mat')):
            EEG2 = so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'), squeeze_me=True)['EEG2']
        
        # load EMG
        if os.path.exists(os.path.join(ppath, rec, 'EMG.mat')):
            EMG = so.loadmat(os.path.join(ppath, rec, 'EMG.mat'), squeeze_me=True)['EMG']
            EMG = sleepy.my_hpfilter(EMG, 0.01)  # high-pass filter at 5Hz
        
        # adjust Intan idx to properly translate to SP idx
        spi_adjust = np.linspace(-sr, sr, len(EEG))
        
        # load or calculate entire high-res spectrogram
        # SP calculated using EEG2
        if ('SP2' in signal_type) and (signal_type != 'SP2_CALC'):
            SP, f, t, sp_nbin, sp_dt = AS.highres_spectrogram(ppath, rec, nsr_seg, perc_overlap, recalc_highres, 'EEG', peeg2=True)
            #sp_nbin = len(EEG) / SP.shape[1]
            sp_win1 = int(round(iwin1/sp_nbin))
            sp_win2 = int(round(iwin2/sp_nbin))
            lsr_sp_win1 = int(round(lsr_iwin1/sp_nbin))
            lsr_sp_win2 = int(round(lsr_iwin2/sp_nbin))
        # SP calculated using EEG
        elif ('SP' in signal_type) and (signal_type != 'SP_CALC'):
            SP, f, t, sp_nbin, sp_dt = AS.highres_spectrogram(ppath, rec, nsr_seg, perc_overlap, recalc_highres,'EEG', peeg2=False)
            sp_win1 = int(round(iwin1/sp_nbin))
            sp_win2 = int(round(iwin2/sp_nbin))
            lsr_sp_win1 = int(round(lsr_iwin1/sp_nbin))
            lsr_sp_win2 = int(round(lsr_iwin2/sp_nbin))
        else:
            sp_win1 = int(round(iwin1/nbin))
            sp_win2 = int(round(iwin2/nbin))
            lsr_sp_win1 = int(round(lsr_iwin1/nbin))
            lsr_sp_win2 = int(round(lsr_iwin2/nbin))
        # calculate SP mean
        if '_NORM' in signal_type:
            SP_mean = SP.mean(axis=1)
            # normalize entire spectrogram
            SP_norm = np.divide(SP, np.repeat([SP_mean], SP.shape[1], axis=0).T)
            
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt=dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
        
        # load LFP and P-wave indices
        if exclude_noise:
            # load noisy LFP indices, make sure no P-waves are in these regions
            LFP, p_idx, lfp_noise_idx, emg_noise_idx = load_pwaves(ppath, rec, return_lfp_noise=True, 
                                                                   return_emg_noise=True)[0:4]
            LFP[lfp_noise_idx] = np.nan
            EMG[emg_noise_idx] = np.nan
            #p_idx = np.setdiff1d(p_idx, noise_idx)
        else:
            LFP, p_idx = load_pwaves(ppath, rec)[0:2]
        # isolate single or clustered P-waves
        if p_iso and pcluster:
            print('ERROR: cannot accept both p_iso and pcluster arguments')
            return
        elif p_iso:
            p_idx = get_p_iso(p_idx, sr, win=p_iso)
        elif pcluster:
            p_idx = get_pclusters(p_idx, sr, win=pcluster, return_event=clus_event)
        
        # load laser and find laser-triggered P=waves
        lsr = sleepy.load_laser(ppath, rec)
        lsr_p_idx, spon_p_idx, lsr_yes_p, lsr_no_p = get_lsr_pwaves(p_idx, lsr, post_stim)
        if exclude_noise:
            lsr_no_p = np.setdiff1d(lsr_no_p, lfp_noise_idx)
            
        # define start and end points of analysis
        istart = int(np.round(tstart*sr))
        if tend == -1:
            iend = len(EEG) - 1
        else:
            iend = int(np.round(tend*sr))
        
        # # get all indices of P-waves and laser pulses
        event_idx = np.concatenate((p_idx, lsr_yes_p, lsr_no_p))

        for ei in event_idx:
            if ei >= iwin1 and ei >= lsr_iwin1 and ei + iwin2 < len(EEG) and ei + lsr_iwin2 < len(EEG) and istart <= ei <= iend:
                if ei in spon_p_idx or ei in lsr_p_idx:
                    w1 = iwin1; w2 = iwin2
                    spw1 = sp_win1; spw2 = sp_win2
                elif ei in lsr_yes_p or ei in lsr_no_p:
                    w1 = lsr_iwin1; w2 = lsr_iwin2
                    spw1 = lsr_sp_win1; spw2 = lsr_sp_win2
                if istate[0] != 'total':
                    brstate = int(M[int(ei/nbin)])
                # get data of desired signal type
                if signal_type == 'EEG':
                    data = EEG[ei-w1 : ei+w2]
                elif signal_type == 'EEG2':
                    data = EEG2[ei-w1 : ei+w2]
                elif signal_type == 'EMG':
                    data = EMG[ei-w1 : ei+w2]
                # calculate SP from EEG or EEG2
                elif 'CALC' in signal_type:
                    if 'SP2' in signal_type:
                        tmp = EEG2[ei-w1 : ei+w2]
                    elif 'SP' in signal_type:
                        tmp = EEG[ei-w1 : ei+w2]
                    f, t, data = scipy.signal.spectrogram(tmp, fs=sr, window='hanning', nperseg=int(nsr_seg * sr), noverlap=int(nsr_seg * sr * perc_overlap))
                    # normalize calculated SP based on entire recording
                    if 'NORM' in signal_type:
                        data = np.divide(data, np.repeat([SP_mean], data.shape[1], axis=0).T)
                # if not calculating, get SP or SP_NORM from whole recording calculation
                elif 'SP' in signal_type:
                    spi = int(round((ei + spi_adjust[ei])/sp_nbin))
                    if spi >= spw1 and spi + spw2 < SP.shape[1]:
                        if 'NORM' in signal_type:
                            data = SP_norm[:, spi-spw1 : spi+spw2]
                        else:
                            data = SP[:, spi-spw1 : spi+spw2]
                    else:
                        continue
                elif signal_type == 'LFP':
                    data = LFP[ei-w1 : ei+w2]
                else:
                    print(signal_type + ' IS AN INVALID SIGNAL TYPE')
                    return
                    
                # collect data in relevant dictionary
                if brstate in istate:
                    if ei in lsr_p_idx:
                        trig_lsr_idx = lsr_yes_p[np.where(lsr_yes_p < ei)[0][-1]]
                        lat = round((ei - trig_lsr_idx)/1000*sr, 4)
                        lsr_pwaves[brstate].append(data)
                    elif ei in spon_p_idx:
                        spon_pwaves[brstate].append(data)
                    else:
                        # check for P-waves within the $lsr_iso window
                        lsr_iso_win = np.arange(ei - int(round(lsr_iso*sr)), ei)
                        preceding_p = np.intersect1d(spon_p_idx, lsr_iso_win)
                        if ei in lsr_yes_p:
                            if len(preceding_p) > 0:
                                lsr_elim_counts['success'] += 1
                            else:
                                trig_p_idx = lsr_p_idx[np.where(lsr_p_idx > ei)[0][0]]
                                lat = round((trig_p_idx - ei)/1000*sr, 4)
                                success_lsr[brstate].append(data)
                        elif ei in lsr_no_p:
                            if len(preceding_p) > 0:
                                lsr_elim_counts['fail'] += 1
                            else:
                                fail_lsr[brstate].append(data)
        # collect signals surrounding random control time points
        if null:
            # sample "null" REM epochs with no P-waves/laser pulses
            if null_win != 0:
                null_iwin1, null_iwin2 = get_iwins(null_win, sr)
                 # find all points that don't qualify as "null"
                not_null_idx = np.zeros((10000000))
                for i, ei in enumerate(event_idx):
                    e_win = np.arange(ei-null_iwin1, ei+null_iwin2)
                    not_null_idx[i*len(e_win) : i*len(e_win)+len(e_win)] = e_win
                # get rid of trailing zeros (computational efficiency)
                not_null_idx = np.trim_zeros(not_null_idx, 'b')
            
            for s in istate:
                # match no. of random control points to the frequency of a given event
                if null_match == 'lsr': num_null = len(lsr_pwaves[s])
                elif null_match == 'success': num_null = len(success_lsr[s])
                elif null_match == 'fail': num_null = len(fail_lsr[s])
                elif null_match == 'all lsr': num_null = len(success_lsr[s]) + len(fail_lsr[s])
                else: num_null = len(spon_pwaves[s])
                if istate[0] != 'total':
                    # get array of all possible indices in state s
                    sseq = sleepy.get_sequences(np.where(M==s)[0])
                    sseq_idx = [np.arange(seq[0]*nbin, seq[-1]*nbin+nbin) for seq in sseq]
                    #sseq_idx = np.array((list(chain.from_iterable(sseq_idx))))
                    sseq_idx = np.concatenate(sseq_idx)
                    sseq_idx = [sidx for sidx in sseq_idx if sidx > iwin1 and sidx < len(EEG)-iwin2 and istart < sidx < iend]
                else:
                    sseq_idx = np.arange(iwin1, len(EEG)-iwin2)
                # keep only state indices that are not next to a P-wave/laser pulse
                if null_win != 0:
                    sseq_idx = np.setdiff1d(sseq_idx, not_null_idx)
                # randomly select from all state indices
                else:
                    sseq_idx = np.array((sseq_idx))
                np.random.seed(0)
                # select no. of control indices matching the number of indicated events
                r_idx = np.random.randint(low=0, high=len(sseq_idx), size=num_null)
                null_idx = sseq_idx[r_idx]
                for ni in null_idx:
                    # get data of desired signal type
                    if signal_type == 'EEG':
                        data = EEG[ni-iwin1 : ni+iwin2]
                    elif signal_type == 'EEG2':
                        data = EEG2[ni-iwin1 : ni+iwin2]
                    elif signal_type == 'EMG':
                        data = EMG[ni-iwin1 : ni+iwin2]
                    # calculate SP from EEG or EEG2
                    elif 'CALC' in signal_type:
                        if 'SP2' in signal_type:
                            tmp = EEG2[ni-iwin1 : ni+iwin2]
                        elif 'SP' in signal_type:
                            tmp = EEG[ni-iwin1 : ni+iwin2]
                        f, t, data = scipy.signal.spectrogram(tmp, fs=sr, window='hanning', nperseg=int(nsr_seg * sr), noverlap=int(nsr_seg * sr * perc_overlap))
                        # normalize calculated SP based on entire recording
                        if 'NORM' in signal_type:  
                            data = np.divide(data, np.repeat([SP_mean], data.shape[1], axis=0).T)
                    # if not calculating, get SP or SP_NORM from whole recording calculation
                    elif 'SP' in signal_type:
                        spi = int(round((ni + spi_adjust[ni])/sp_nbin))
                        if spi >= sp_win1 and spi + sp_win2 < SP.shape[1]:
                            if 'NORM' in signal_type:
                                data = SP_norm[:, spi-sp_win1 : spi+sp_win2]
                            else:
                                data = SP[:, spi-sp_win1 : spi+sp_win2]
                        else:
                            continue
                    elif signal_type == 'LFP':
                        data = LFP[ni-iwin1 : ni+iwin2]
                    else:
                        print(signal_type + ' IS AN INVALID SIGNAL TYPE')
                        return
                    # collect data in null dictionary
                    null_pts[s].append(data)
        
        # save tmp files to free up more room for computation
        for s in istate:
            so.savemat(f'TMP_{rec}_lsr_pwaves_{s}.mat', {'data':lsr_pwaves[s]})
            so.savemat(f'TMP_{rec}_spon_pwaves_{s}.mat', {'data':spon_pwaves[s]})
            so.savemat(f'TMP_{rec}_success_lsr_{s}.mat', {'data':success_lsr[s]})
            so.savemat(f'TMP_{rec}_fail_lsr_{s}.mat', {'data':fail_lsr[s]})
            so.savemat(f'TMP_{rec}_null_pts_{s}.mat', {'data':null_pts[s]})
    
    print(f"{lsr_elim_counts['success']} successful laser pulses and {lsr_elim_counts['fail']} failed laser pulses were eliminated due to closely preceding P-waves.")
    if psave:
        print('\n Assembling data dictionaries and saving .mat files ...\n')
    else:
        print('\n Assembling data dictionaries ...\n')
        
    # collect data from all recordings for each state from tmp files
    lsr_pwaves = {s:{rec:0 for rec in recordings} for s in istate}
    spon_pwaves = {s:{rec:0 for rec in recordings} for s in istate}
    success_lsr = {s:{rec:0 for rec in recordings} for s in istate}
    fail_lsr = {s:{rec:0 for rec in recordings} for s in istate}
    null_pts = {s:{rec:0 for rec in recordings} for s in istate}
    
    # assemble final data dictionaries
    for s in istate:
        for rec in recordings:
            lsr_pwaves[s][rec] = so.loadmat(f'TMP_{rec}_lsr_pwaves_{s}.mat')['data']
            spon_pwaves[s][rec] = so.loadmat(f'TMP_{rec}_spon_pwaves_{s}.mat')['data']
            success_lsr[s][rec] = so.loadmat(f'TMP_{rec}_success_lsr_{s}.mat')['data']
            fail_lsr[s][rec] = so.loadmat(f'TMP_{rec}_fail_lsr_{s}.mat')['data']
            null_pts[s][rec] = so.loadmat(f'TMP_{rec}_null_pts_{s}.mat')['data']
    
        # remove temporary files
        for rec in recordings:
            os.remove(f'TMP_{rec}_lsr_pwaves_{s}.mat')
            os.remove(f'TMP_{rec}_spon_pwaves_{s}.mat')
            os.remove(f'TMP_{rec}_success_lsr_{s}.mat')
            os.remove(f'TMP_{rec}_fail_lsr_{s}.mat')
            os.remove(f'TMP_{rec}_null_pts_{s}.mat')
        # save files
        data_shape = list(spon_pwaves[s].values())[0][0].shape
        if psave:
            for s in istate:
                filename = psave if isinstance(psave, str) else f'lsrSurround_{signal_type}'
                so.savemat(os.path.join(ppath, f'{filename}_lsr_pwaves_{s}.mat'), lsr_pwaves[s])
                so.savemat(os.path.join(ppath, f'{filename}_spon_pwaves_{s}.mat'), spon_pwaves[s])
                so.savemat(os.path.join(ppath, f'{filename}_success_lsr_{s}.mat'), success_lsr[s])
                so.savemat(os.path.join(ppath, f'{filename}_fail_lsr_{s}.mat'), fail_lsr[s])
                so.savemat(os.path.join(ppath, f'{filename}_null_pts_{s}.mat'), null_pts[s])
                so.savemat(os.path.join(ppath, f'{filename}_data_shape.mat'), {'data_shape':data_shape})
    
    END = time.perf_counter()
    print(f'COMPUTING TIME --> {END-START:0.2f} seconds ({len(recordings)} recordings, {len(istate)} brainstates, signal type = {signal_type})')
    return lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape


def get_band_pwr(sp_dict, f, bands, mouse_avg='mouse', pnorm=0, psmooth=0, sf=0, fmax=0):
    """
    Create dictionary with frequency band labels as keys and matrices of averaged EEG power 
    (mice/trials x time bins) as values
    @Params
    sp_dict - dictionary with a key for each mouse and list of SPs for each key
    f - list of frequencies corresponding to  y axis of SPs in $sp_dict
    bands - list of tuples containing lowest and highest frequencies in each power band
    mouse_avg - if 'mouse'  - mx of mice x time bins for each power band
                if 'trial' - mx of trials x time bins for each power band
    pnorm - if > 0, normalize each freq in a band by the time window of SPs in $sp_dict,
             before calculating mean band power
    psmooth, sf - smoothing params for spectrograms/band vectors
    fmax - maximum frequency in analyzed SPs
    @Returns
    band_pwr - dictionary of frequency band power values (keys=freq band labels,
                values=matrices of mice/trials x time bins)
    labels - list of mouse names, or integers to number trials
    """
    band_pwr_dict = {b:0 for b in bands}
    
    # create mx of freq x time bins x subjects, cutoff frequencies > fmax
    sp_mx, labels = mx3d(sp_dict, mouse_avg)
    if fmax:
        sp_mx = sp_mx[np.where(f <= fmax)[0], :, :]
    
    for b in bands:
        # for each band, create mx of mice/trials x time bins
        bmx = np.zeros((sp_mx.shape[2], sp_mx.shape[1]))
        # idx of frequencies in power band
        ifreq = np.intersect1d(np.where(f >= b[0])[0], np.where(f <= b[1])[0])
        for layer in range(sp_mx.shape[2]):
            # for each SP, isolate frequencies in power band $b
            ms_sp = sp_mx[:, :, layer]
            if psmooth:
                ms_sp = AS.convolve_data(ms_sp, psmooth=psmooth)
            ms_band_sp = ms_sp[ifreq, :]
            # normalize frequency band power
            if pnorm:
                ms_band_mean = np.nanmean(ms_band_sp, axis=1)
                ms_band_sp = np.divide(ms_band_sp, np.repeat([ms_band_mean], ms_band_sp.shape[1], axis=0).T)
            # get mean band power for each time bin, smooth across x axis
            ms_band = np.mean(ms_band_sp, axis=0)
            #ms_band = np.sum(ms_band_sp, axis=0)
            if sf > 0:
                ms_band = AS.smooth_data(ms_band, sf)
            bmx[layer, :] = ms_band
        band_pwr_dict[b] = bmx

    return band_pwr_dict, labels


##############          P-WAVE DETECTION FUNCTIONS          ##############

def detect_pwaves(ppath, recordings, channel, thres, thres_init=5, dup_win=40,
                  select_thresholds=False, set_outlier_th=False, rm_noise=False, 
                  n_thres=800, n_win=5):
    """
    Detect P-waves from LFP signal using given threshold value
    @Params
    ppath - base folder
    recordings - list of recordings
    channel - method for processing raw LFPs
              'S'  - subtract the reference LFP from the primary LFP
              'C'  - use the primary LFP
               1   - use LFP channel 1
               2   - use LFP channel 2
               1-2 - subtract LFP channel 2 from LFP channel 1
               2-1 - subtract LFP channel 1 from LFP channel 2
    thres - threshold for detecting P-waves (mean - thres*std)
    thres_init - initial threshold for determining the primary LFP channel
    dup_win - perform additional validation for P-waves within $dup_win ms
              of their nearest neighbor
    select_thresholds - if True, manually set detection threshold for each mouse
                        if False, use $thres as detection threshold for all mice
    set_outlier_th - if True, manually set amplitude/half-width thresholds of outlier waveforms
                     if False, use pre-calculated thresholds
    rm_noise - if True, eliminate P-waves within "noisy" LFP regions
    n_thres - threshold for detecting LFP noise (uV)
    n_win - size of window (s) surrounding each instance of LFP noise
    @Returns
    None
    """
    # params for classifying microarousals/transition states
    ma_thr=20; ma_state=6; flatten_is=3

    for rec in recordings:
        print('Getting P-waves for', rec, '...')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
        
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
        if os.path.exists(os.path.join(ppath, rec, 'LFP_raw2.mat')):
            LFP_raw2 = so.loadmat(os.path.join(ppath, rec, 'LFP_raw2'), 
                                  squeeze_me=True)['LFP_raw2']
        elif os.path.exists(os.path.join(ppath, rec, 'EMG2.mat')):
            LFP_raw2 = so.loadmat(os.path.join(ppath, rec, 'EMG2.mat'), 
                                  squeeze_me=True)['EMG2']
            
        # handle cases with a single LFP channel
        if len(LFP_raw) == 0:
            if len(LFP_raw2) == 0:
                raise ValueError("ERROR: No LFP files found")
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
        idx1 = spike_threshold(LFP_raw, th1) 
        th2 = np.nanmean(LFP_raw2[base_idx]) + thres_init*np.nanstd(LFP_raw2[base_idx])
        idx2 = spike_threshold(LFP_raw2, th2)
        
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
                    si = spike_threshold(sLFP, th)
                    nidx[s][i] = len(si)
            # plot number of detected P-waves per threshold value
            fig, axs = plt.subplots(figsize=(5,10), nrows=3, ncols=1, 
                                    constrained_layout=True)
            for s in [1,2,3]:
                axs[s-1].bar(thres_range, nidx[s])
                axs[s-1].set_title('State ' + str(s))
            fig.suptitle(rec)
            plt.show()
            # enter desired threshold
            p_thres = float(input('Enter the threshold value for P-wave detection ---> '))
            
        # use $thres param as common threshold
        else:
            p_thres = thres
    
    # get P-wave indices from processed LFP using chosen threshold 
    p_th = np.nanmean(LFP[base_idx]) + p_thres*np.nanstd(LFP[base_idx])
    pi = spike_threshold(LFP, p_th)
    # get amplitudes and half-widths of P-waves
    p_amps = [get_amp(LFP, i, sr) for i in pi]
    p_widths = [get_halfwidth(LFP, i, sr) for i in pi]
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
        hw_thres = hw_thres/1000*sr #/1000 
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
        noise_idx = detect_noise(LFP, sr, n_thres, n_win)
        noise_waves = [i for i in p_idx if i in noise_idx]
        
    # eliminate all disqualified waves
    p_elim = amp_hi + width_hi + dup_waves
    if rm_noise:
        p_elim += noise_waves
    p_idx = [i for i in p_idx if i not in p_elim]
    # save indices of outlier and duplicate waves
    q = np.zeros((3,len(LFP)))
    q[0, amp_hi] = 1
    q[1, width_hi] = 3
    q[2, dup_waves] = 5
    
    # save processed LFP
    so.savemat(os.path.join(ppath, rec, 'LFP_processed.mat'), {'LFP_processed': LFP})
    # save P-wave indices and detection threshold
    so.savemat(os.path.join(ppath, rec, 'p_idx.mat'), {'p_idx': p_idx, 
                                                       'thres': p_th, 
                                                       'thres_scale': p_thres,
                                                       'p_elim': q})


def load_pwaves(ppath, name, return_lfp_noise=False, return_emg_noise=False, return_eeg_noise=False):
    """
    Load LFP signal and detected P-wave indices from .mat files
    @Params
    ppath - base folder
    name - recording folder
    return_[lfp/emg/emg]_noise - if True, return indices of LFP, EMG, and/or EEG 
                                 noise identified during P-wave detection
    @Returns
    LFP - loaded LFP signal
    p_idx - loaded P-wave indices
    noise_idx - loaded LFP noise indices (if $return_lfp_noise=True)
    emg_noise_idx - loaded EMG noise indices (if $return_emg_noise=True)
    """
    # load LFP array
    fpath1 = os.path.join(ppath, name, 'LFP_processed.mat')
    if os.path.exists(fpath1):
        try:
            LFP = so.loadmat(fpath1, squeeze_me=True)['LFP_processed']
        except ValueError:
            try:
                with h5py.File(fpath1, 'r') as f:
                    LFP = np.squeeze(f['LFP_processed'])
            except OSError:
                return OSError('Processed LFP file must be saved in .mat or h5py format')
    else:
        return FileNotFoundError('This recording has no processed LFP file. Run "detect_pwaves" first.')
    
    # load P-wave info file
    open_fid = []
    # load P-wave indices
    fpath2 = os.path.join(ppath, name, 'p_idx.mat')
    if os.path.exists(fpath2):
        try:
            pdata = so.loadmat(fpath2, squeeze_me=True)
        except ValueError:
            try:
                pdata = h5py.File(fpath2, 'r')
                open_fid.append(pdata)
            except OSError:
                return OSError('p_idx file must be saved in .mat or h5py format')
    else:
        return FileNotFoundError('This recording has no saved P-wave indices. Run "detect_pwaves" first.')
    
    # get requested P-wave data from file
    arg_list = [[True, 'p'],
                [return_lfp_noise, 'noise'], 
                [return_emg_noise, 'emg_noise'], 
                [return_eeg_noise, 'eeg_noise']]
    arr_list = []
    
    for arg,key in arg_list:
        arr = np.array((), dtype='int32')
        if arg == True:
            # p_idx key in original .mat file
            if key + '_idx' in pdata.keys():
                arr = np.squeeze(pdata[key + '_idx']).astype('int32')
            # p_train key in Neuropixel h5py file
            elif key + '_train' in pdata.keys():
                train = np.squeeze(pdata[key + '_train']).astype('int32')
                if train.shape is not None:
                    arr = np.where(train == 1)[0].astype('int32')
            else:
                if key == 'p':
                    return Exception('Cannot locate P-wave indices in p_idx file') 
                else:
                    x = ['LFP','EEG','EMG'][len(arr_list)-1]
                    print('### WARNING - No ' + x + ' noise indices found ###')
        arr_list.append(arr)
    # close h5py file
    if len(open_fid) > 0:
        _ = [f.close() for f in open_fid]
        
    arr_list.insert(0, LFP)
    return arr_list


def downsample_pwaves(LFP, p_idx, sr, nbin, rec_bins):
    """
    Bin P-waves by $nbin/sr seconds (usually 2.5) to align with spectrogram
    @Params
    LFP - LFP signal
    p_idx - indices of P-wave
    sr - sampling rate (Hz)
    nbin - no. samples per time bin (when sr=1000, usually 2500)
    rec_bins - total time bins in the recording
    @Returns
    p_idx_dn       - ROUNDED P-wave indices in downsampled time 
                     e.g. for sr=1000 and nbin=2500: LFP idx 10,001 --> time bin 4
    p_idx_dn_exact - EXACT P-wave indices in downsampled time
                     e.g. for sr=1000 and nbin=2500: LFP idx 10,001 --> time bin 4.004
    p_count_dn     - number of P-waves per time bin
    p_freq_dn      - avg. P-wave frequency per time bin
    """
    p_idx_dn = []
    p_idx_dn_exact = []
    # downsample P-wave indices
    for px in p_idx:
        PX_DN = px/nbin
        # collect rounded downsampled idx (indexes to a time bin)
        p_idx_dn.append(round(PX_DN))
        # collect exact downsampled idx (convertible back to exact LFP spot if necessary)
        p_idx_dn_exact.append(PX_DN)
    
    # calculate number of P-waves in each time bin
    p_count_dn = np.array([p_idx_dn.count(x) for x in range(0, rec_bins)])
    
    # get P-wave frequency in each time bin
    p = np.zeros((len(LFP),))
    p[p_idx] = sr
    p_freq_dn = AS.downsample_vec(p, nbin)
    # if downsampling shifts values 1 bin behind in time:
    if np.where(p_count_dn>0)[0][0] == np.where(p_freq_dn>0)[0][0] + 1:
        # shift values forward by 1 bin
        p_freq_dn = np.insert(p_freq_dn[0:-1], 0, 0.0)
    
    return p_idx_dn, p_idx_dn_exact, p_count_dn, p_freq_dn


def get_lsr_pwaves(p_idx, lsr, post_stim=0.1, sr=1000., lat_thr=0):
    """
    Get indices of laser-triggered P-waves, spontaneous P-waves, successful
    laser pulses, and failed laser pulses
    @Params
    p_idx - indices of P-waves
    lsr - laser stimulation vector
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    sr - sampling rate (Hz)
    lat_thr - optional threshold for latency of a triggered P-wave from laser pulse onset (ms)
              if > 0, return laser pulses & P-waves with latencies longer than $lat_thr
              if < 0, return laser pulses & P-waves with latencies shorter than abs($lat_thr)
    @Returns
    lsr_p_idx    - indices of laser-triggered P-waves
    spon_p_idx   - indices of P-waves not triggered by the laser
    success_lsr  - indices of laser pulses that successfully triggered a P-wave
    fail_lsr     - indices of laser pulses that failed to trigger a P-wave
    """
    if type(p_idx) == list:
        p_idx = np.array(p_idx)
    # get indices of laser pulse onsets
    ilsr_start = np.array([i[0] for i in sleepy.get_sequences(np.where(lsr!=0.)[0])])
    #lsr_start_idx = [i[0] for i in sleepy.get_sequences(np.where(lsr!=0.)[0])]
    
    # get indices of post-laser windows (defined by $post_stim param)
    trig_seqs = [np.arange(i+1, int(i + post_stim*sr)) for i in ilsr_start]
    trig_idx = np.concatenate(trig_seqs)
    
    # find "laser-triggered" P-waves
    lsr_p_idx = np.intersect1d(p_idx, trig_idx)
    # all other P-waves are "spontaneous"
    spon_p_idx = np.setdiff1d(p_idx, lsr_p_idx)
    
    # find "successful" laser pulses
    ntrigs = np.array([len(np.intersect1d(iseq,p_idx)) for iseq in trig_seqs])
    itrigs = np.where(ntrigs > 0)[0].astype('int')
    success_lsr = ilsr_start[itrigs]
    # all other laser pulses are "failed"
    fail_lsr = np.setdiff1d(ilsr_start, success_lsr)
    
    # apply latency threshold
    lat_elim = []
    #lat_thr = 0
    if lat_thr != 0:
        for li in lsr_p_idx:
            ltrig = success_lsr[np.where(success_lsr < li)[0][-1]]
            lat = round((li - ltrig)/1000*sr, 4)
            if (lat_thr > 0) and lat <= lat_thr:  # remove p-waves with low latencies
                lat_elim.append(li)
            elif (lat_thr < 0) and lat >= lat_thr:  # remove p-waves with high latencies
                lat_elim.append(li)
        for sl in success_lsr:
            ptrigs = lsr_p_idx[np.where((lsr_p_idx > sl) & (lsr_p_idx <= sl + (int(post_stim*sr))))[0]]
            if all(np.in1d(ptrigs, lat_elim)):
                lat_elim.append(sl)               # remove laser pulses with no qualifying triggered p-waves
    lat_elim = np.array(lat_elim)
    lsr_p_idx = np.setdiff1d(lsr_p_idx, lat_elim)
    success_lsr = np.setdiff1d(success_lsr, lat_elim)

    return lsr_p_idx, spon_p_idx, success_lsr, fail_lsr


def get_p_iso(p_idx, sr, win=0.8, order=[1,1]):
    """
    Return indices of isolated/single P-waves
    @Params
    p_idx - indices of P-waves
    sr - sampling rate (Hz)
    win - a P-wave is "isolated" if distance to its nearest neighbor is >= $win s
    order - if [1,0] --> no neighbors in $win s before P-wave
            if [0,1] --> no neighbors in $win s after P-wave
            if [1,1] --> no neighbors in $win s before OR after P-wave
    @Returns
    p_iso_idx - indices of single P-waves
    """
    if type(win) not in [list, tuple]:
        win = [win, win]
    win[0] = np.abs(win[0])

    p_iso_idx = []
    for i, pi in enumerate(p_idx):
        # distance from current to previous P-wave is greater than $win s
        isoPre = (pi - p_idx[i-1] >= win[0]*sr) if i > 0 else True
        # distance from current to next P-wave is greater than $win s
        isoPost = (p_idx[i+1] - pi >= win[1]*sr) if i < len(p_idx)-1 else True
    
        if order == [1,1] and isoPre and isoPost:
            p_iso_idx.append(pi)
        elif order == [1,0] and isoPre:
            p_iso_idx.append(pi)
        elif order == [0,1] and isoPost:
            p_iso_idx.append(pi)
    p_iso_idx = np.array(p_iso_idx)
            
    return p_iso_idx


def get_pclusters(p_idx, sr, win=0.5, iso=0, return_event='waves'):
    """
    Return indices of clustered P-waves
    @Params
    p_idx - indices of P-waves
    sr - sampling rate (Hz)
    win - a P-wave is "clustered" if the distance to its nearest neighbor is <= $win s
    iso - if >0, only return clusters NOT preceded by a P-wave within $iso seconds
    return_event - type of cluster information to return
                   'waves' - indices of all P-waves occurring within clusters
                   'cluster_all' - list of tuples, each containing P-wave indices for 1 cluster
                   'cluster_start' - indices of the first P-wave in each cluster
                   'cluster_mid' - indices of the middle P-wave in each cluster
                   'cluster_end' - indices of the last P-wave in each cluster
                   'cluster_edges' - list of 2-element tuples, each containing indices of 
                                     the first and last P-wave in 1 cluster
    @Returns
    pcluster_idx - indices of P-wave clusters
    """
    pcluster_idx = []
    i=0
    while i < len(p_idx)-1:
        # eliminate closely preceding P-waves (if iso>0)
        if i > 0 and p_idx[i] - p_idx[i-1] >= iso*sr:
            # collect indices of P-waves in 1 cluster
            cl_idx = [p_idx[i]]
            cl_count = 0
            # does each consecutive P-wave follows the previous within $win seconds?
            while p_idx[i+cl_count+1] - p_idx[i+cl_count] <= win*sr:
                cl_idx.append(p_idx[i+cl_count+1])
                cl_count += 1
                if i + cl_count + 1 == len(p_idx):
                    break

            if len(cl_idx) == 1:  # for a "cluster" of 1 wave, continue
                i += 1 
            elif len(cl_idx) > 1:
                # collect all  P-waves in cluster
                if return_event in ['wave', 'waves']:
                    pcluster_idx.append(cl_idx)
                # collect P-wave at cluster beginning
                elif 'start' in return_event:
                    pcluster_idx.append(cl_idx[0])
                # collect P-wave at cluster end
                elif 'end' in return_event:
                    pcluster_idx.append(cl_idx[-1])
                # collect P-wave in middle of cluster
                elif 'mid' in return_event:
                    pcluster_idx.append(cl_idx[int(np.floor(len(cl_idx)/2))])
                # collect first and last P-waves in cluster
                elif 'edge' in return_event:
                    pcluster_idx.append([cl_idx[0], cl_idx[-1]])
                # skip ahead to next P-wave after the cluster
                i += cl_count
        else:
            i += 1
    if return_event in ['wave', 'waves']:
        pcluster_idx = list(chain.from_iterable(pcluster_idx))
    pcluster_idx = np.array(pcluster_idx)
    
    return pcluster_idx


def get_lsr_cluster(ppath, recordings, istate, post_stim=0.1, p_iso=1, pcluster=0.5, 
                    clus_event='waves', ma_thr=20, ma_state=3, flatten_is=False):
    """
    Calculate the percent of single vs clustered laser-triggered P-waves, compared to 
    single vs clustered spontaneous P-waves, in each brain state
    @Params 
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze 
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    @Returns
    None
    """
    
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    
    # count single vs clustered spontaneous and laser-triggered P-waves
    data = {s:{m:{'total lsr':0, 'lsr cluster':0, 'lsr iso':0, 
                  'total spon':0, 'spon cluster':0, 'spon iso':0} for m in mice} for s in istate}
    
    for rec in recordings:
        print('Getting P-waves for ' + rec + ' ...')
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
               
        # load EEG, LFP, and P-wave indices
        EEG = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        LFP, p_idx = load_pwaves(ppath, rec)[0:2]
        # get indices of single and clustered P-waves
        iso_idx = get_p_iso(p_idx, sr, win=p_iso)
        clus_idx = get_pclusters(p_idx, sr, win=pcluster, return_event=clus_event)
        
        # load laser and find laser-triggered P-waves
        lsr = sleepy.load_laser(ppath, rec)
        lsr_p_idx, spon_p_idx, lsr_yes_p, lsr_no_p = get_lsr_pwaves(p_idx, lsr, post_stim)
        
        spon_iso = np.intersect1d(spon_p_idx, iso_idx)  # spontaneous single P-waves
        lsr_iso = np.intersect1d(lsr_p_idx, iso_idx)  # laser-triggered single P-waves
        spon_clus = np.intersect1d(spon_p_idx, clus_idx)  # spontaneous cluster P-waves
        lsr_clus = np.intersect1d(lsr_p_idx, clus_idx)  # laser-triggered cluster P-waves
        
        for s in istate:
            # for each brain state, count number of each event
            if s != 0:
                state_spon_iso = [i for i in spon_iso if M[int(i/nbin)]==s]
                state_lsr_iso = [i for i in lsr_iso if M[int(i/nbin)]==s]
                state_spon_clus = [i for i in spon_clus if M[int(i/nbin)]==s]
                state_lsr_clus = [i for i in lsr_clus if M[int(i/nbin)]==s]
                data[s][idf]['total lsr'] += len([i for i in lsr_p_idx if M[int(i/nbin)]==s])
                data[s][idf]['total spon'] += len([i for i in spon_p_idx if M[int(i/nbin)]==s])
            # count number of events across all brain states
            else:
                state_spon_iso = np.array(spon_iso)
                state_lsr_iso = np.array(lsr_iso)
                state_spon_clus = np.array(spon_clus)
                state_lsr_clus = np.array(lsr_clus)
                data[s][idf]['total lsr'] += len(lsr_p_idx)
                data[s][idf]['total spon']  += len(spon_p_idx)
            # collect data in dictionary
            data[s][idf]['spon iso'] += len(state_spon_iso)
            data[s][idf]['lsr iso'] += len(state_lsr_iso)
            data[s][idf]['spon cluster'] += len(state_spon_clus)
            data[s][idf]['lsr cluster'] += len(state_lsr_clus)
    
    # for each brain state and each mouse, calculate percent of single vs clustered
    # waveforms for spontaneous and laser-triggered P-waves
    for s in istate:
        lsr_perc_clus = np.zeros((len(mice),))
        lsr_perc_iso = np.zeros((len(mice),))
        spon_perc_clus = np.zeros((len(mice),))
        spon_perc_iso = np.zeros((len(mice),))
        for i,m in enumerate(mice):
            lsr_perc_clus[i] = (data[s][m]['lsr cluster'] / data[s][m]['total lsr']) * 100
            lsr_perc_iso[i] = (data[s][m]['lsr iso'] / data[s][m]['total lsr']) * 100
            spon_perc_clus[i] = (data[s][m]['spon cluster'] / data[s][m]['total spon']) * 100
            spon_perc_iso[i] = (data[s][m]['spon iso'] / data[s][m]['total spon']) * 100
        print('')
        print('STATE = ' + str(s))
        print(f'percent iso      --- spon={round(np.nanmean(spon_perc_iso),3)}, lsr={round(np.nanmean(lsr_perc_iso),3)}')
        print(f'percent cluster  --- spon={round(np.nanmean(spon_perc_clus),3)}, lsr={round(np.nanmean(lsr_perc_clus),3)}')
        print('')


def get_lsr_stats(ppath, recordings, istate, post_stim=0.1, p_iso=0, pcluster=0, 
                  clus_event='waves', ma_thr=20, ma_state=3, flatten_is=False, 
                  lsr_jitter=0, lat_thr=0, psave=False):
    """
    Get descriptive statistics for spontaneous and laser-triggered P-waves, and
    successful and failed laser pulses
    @Params 
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze 
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    lsr_jitter - randomly shift indices of true laser pulses by up to +/- $lsr_jitter seconds
                 (as a control)
    psave - optional string specifying a filename to save the dataframe (if False, df is not saved)
    @Returns
    df - dataframe containing descriptive info about P-waves and laser pulses, including
    event type, LFP amplitude and halfwidth, timestamp in recording and brain state episode,
    instantaneous theta phase, etc.
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    
    # collect dataframe rows (1 event = 1 row)
    df_rows = []
        
    for rec in recordings:
        
        print('Getting P-waves for ' + rec + ' ...')
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
        
        # load EEG, LFP, and P-wave indices
        EEG = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        LFP, p_idx = load_pwaves(ppath, rec)[0:2]
        # isolate single or clustered P-waves
        if p_iso and pcluster:
            print('ERROR: cannot accept both p_iso and pcluster arguments')
            return
        elif p_iso:
            p_idx = get_p_iso(p_idx, sr, win=p_iso)
        elif pcluster:
            p_idx = get_pclusters(p_idx, sr, win=pcluster, return_event=clus_event)

        # load laser and find laser-triggered P-waves
        lsr = sleepy.load_laser(ppath, rec)
        lsr_p_idx, spon_p_idx, lsr_yes_p, lsr_no_p = get_lsr_pwaves(p_idx, lsr, post_stim, lat_thr=lat_thr)
        # randomly jitter laser indices and identify sham "successful" and "failed" pulses
        if lsr_jitter > 0:
            jlsr = jitter_laser(lsr, lsr_jitter, sr)
            jlsr_p_idx, _, jlsr_yes_p, jlsr_no_p = get_lsr_pwaves(p_idx, jlsr, post_stim)
            
        # create vector to convert event indices to state sequence indices
        event_vec = np.zeros((len(EEG)))
        event_vec[lsr_p_idx] = 1
        event_vec[spon_p_idx] = 2
        event_vec[lsr_yes_p] = 3
        event_vec[lsr_no_p] = 4
        if lsr_jitter > 0:
            # make separate vector for jitter events so nothing gets overwritten
            jitter_vec = np.zeros((len(EEG)))
            jitter_vec[jlsr_p_idx] = 5
            jitter_vec[jlsr_yes_p] = 6
            jitter_vec[jlsr_no_p] = 7
        
        for s in istate:
            sseq = sleepy.get_sequences(np.where(M==s)[0])
            for seq in sseq:
                if len(seq) > 1:
                    # for each brain state sequence, isolate corresponding LFP indices
                    sidx = int(round(seq[0]*nbin))
                    eidx = int(round(seq[-1]*nbin))
                    seq_lfp = LFP[sidx : eidx+int(post_stim*sr)]
                    # convert event indices to isolated LFP sequence
                    li, si, yi, ni = [np.where(event_vec[sidx:eidx] == n)[0] for n in [1, 2, 3, 4]]  # get event indices in seq
                    if lsr_jitter > 0:
                        ji, jyi, jni = [np.where(jitter_vec[sidx:eidx] == n)[0] for n in [5, 6, 7]]  # get jitter event indices
                        seq_eidx = np.concatenate((li,si,yi,ni,ji,jyi,jni))
                    else:
                        ji=[]; jyi=[]; jni=[]
                        seq_eidx = np.concatenate((li,si,yi,ni))
                    
                    # if sequence is REM sleep, filter theta rhythm and collect theta phase
                    if s == 1:
                        # isolate and bandpass filter EEG for seq
                        seq_eeg = EEG[sidx : eidx]
                        a1 = 6.0 / (sr/2.0)
                        a2 = 12.0 / (sr/2.0)
                        seq_eeg = sleepy.my_bpfilter(seq_eeg, a1, a2)
                        res = scipy.signal.hilbert(seq_eeg)  # hilbert transform
                        iphase = np.angle(res)  # get instantaneous phase
                    else:
                        iphase = np.empty((len(EEG)))
                        iphase[:] = np.nan
                    
                    for sei in seq_eidx:
                        s_into_rec = round((sidx + sei)/sr, 2)
                        # calculate absolute (s) and relative (%) time into brain state
                        s_into_seq = round(sei/sr, 2)
                        perc_into_seq = round((sei / (eidx - sidx))*int(post_stim*sr), 2)
                        phase = iphase[sei]
                        
                        if sei in li:  # laser-triggered P-waves
                            eventID = 'lsr-triggered pwave'  # event label
                            amp = -seq_lfp[sei]  # raw LFP value
                            try:
                                trig_lsr_idx = yi[np.where(yi < sei)[0][-1]]
                            except:
                                print('triggering pulse outside sequence')
                                continue
                            lat = round((sei - trig_lsr_idx)/sr*1000, 4)  # latency from laser onset
                            halfwidth = get_halfwidth(seq_lfp,sei,sr)  # waveform half-width
                            amp2 = get_amp(seq_lfp,sei,sr)  # waveform amplitude
                        elif sei in si:  # spontaneous P-waves
                            eventID = 'spontaneous pwave'
                            amp = -seq_lfp[sei]
                            lat = np.nan
                            halfwidth = get_halfwidth(seq_lfp,sei,sr)
                            amp2 = get_amp(seq_lfp,sei,sr)
                        elif sei in yi:  # successful laser pulse
                            eventID = 'successful lsr'
                            amp = -min(seq_lfp[sei : sei + int(post_stim*sr)])
                            trig_p_idx = li[np.where(li > sei)[0][0]]
                            lat = round((trig_p_idx - sei)/sr*1000, 4)
                            halfwidth = np.nan
                            amp2 = np.nan
                        elif sei in ni:  # failed laser pulse
                            eventID = 'failed lsr'
                            amp = -min(seq_lfp[sei : sei + int(post_stim*sr)])
                            lat = np.nan
                            halfwidth = np.nan
                            amp2 = np.nan
                        if sei in ji:  # jittered "laser-triggered" P-wave
                            eventID = 'sham pwave'
                            amp = -seq_lfp[sei]
                            trig_lsr_idx = jyi[np.where(jyi < sei)[0][-1]]
                            lat = round((sei - trig_lsr_idx)/sr*1000, 4)
                            halfwidth = get_halfwidth(seq_lfp,sei,sr)
                            amp2 = get_amp(seq_lfp,sei,sr)
                        elif sei in jyi:  # jittered "successful" laser pulse
                            eventID = 'jitter success'
                            amp = -min(seq_lfp[sei : sei + int(post_stim*sr)])
                            trig_p_idx = ji[np.where(ji > sei)[0][0]]
                            lat = round((trig_p_idx - sei)/sr*1000, 4)
                            halfwidth = np.nan
                            amp2 = np.nan
                        elif sei in jni:  # jittered "failed" laser pulse
                            eventID = 'jitter fail'
                            amp = -min(seq_lfp[sei : sei + int(post_stim*sr)])
                            lat = np.nan
                            halfwidth = np.nan
                            amp2 = np.nan
                        
                        # collect all information from event into dataframe
                        data_row = {'mouse'     : idf,
                                    'recording' : rec,
                                    'state'     : s,
                                    'event'     : eventID,
                                    's'         : s_into_rec,
                                    's_seq'     : s_into_seq,
                                    'perc_seq'  : perc_into_seq,
                                    'amp'       : amp,
                                    'amp2'      : amp2,
                                    'phase'     : phase,
                                    'latency'   : lat,
                                    'halfwidth' : halfwidth}
                        df_rows.append(pd.Series(data_row))

    # assemble and save dataframe
    df = pd.concat(df_rows, axis=1).transpose()
    if psave:
        filename = psave if isinstance(psave, str) else f'lsr_stats'
        if filename.endswith('.pkl'):
            df.to_pickle(os.path.join(ppath,filename))
        else:
            df.to_pickle(os.path.join(ppath,f'{filename}.pkl'))
    return df


def get_amp(LFP, pi, sr):
    """
    Calculate P-wave amplitude, measured as the distance between the negative peak
    and the beginning of the waveform
    @Params
    LFP - LFP signal (variable lengths)
    pi - index of negative peak of P-wave
    sr - sampling rate (Hz)
    @Returns
    height - amplitude (uV) of waveform
    """
    # cut LFP to ~100 ms preceding the P-wave and downsample by 2 if possible (computational efficiency)
    wform = AS.downsample_vec(LFP[pi-(int(100/1000*sr)) : pi+2], 2)
    if len(wform) > 0:
        i = len(wform)-1  # P-wave trough idx
        amp2 = wform[i]
    else:
        # for short LFPs, keep entire vector up to the P-wave
        wform = LFP[0:pi+2]
        amp2 = LFP[pi]
        i = pi-1
    # iterate backwards from trough to beginning (LFP value increasing)
    while i >= 0:
        i -= 1
        # find idx where LFP value starts decreasing
        if wform[i] <= wform[i+1]:
            # if wave is small but LFP idx is highest in waveform, break
            if np.abs(amp2 - wform[i]) < 100:
                if max(wform) == wform[i+1]:
                    break
                else:
                    continue
            # wave is appropriate size, break
            else:                   
                break
    # eliminate unreasonably small outliers, return amplitude
    amp1 = wform[i+1]
    if np.abs(amp2 - amp1) < 10:
        return np.nan
    else:
        height = np.abs(amp2 - amp1)
        return height
    
    
def get_halfwidth(LFP, pi, sr):
    """
    Calculate width of P-wave at 1/2 maximum amplitude
    @Params
    LFP - LFP signal (variable lengths)
    pi - index of negative peak of P-wave
    sr - sampling rate (Hz)
    @Returns
    width - half-width (ms) of waveform
    """
    # get LFP value (uV) at vertical halfway point of waveform
    half_amp = get_amp(LFP,pi,sr) / 2
    half_up = LFP[pi] + half_amp
    
    # iterate left and right to the indices at the half-amp value
    i1 = pi-1
    while LFP[i1] <= half_up:
        i1 -= 1
    i2 = pi+1
    while LFP[i2] <= half_up:
        i2 += 1
        if i2 == len(LFP):
            return np.nan
    # half-width of P-wave in ms
    width = (i2-i1)/1000*sr
    if width < 5:
        return np.nan
    else:
        return width


def inter_pwave_intervals(ppath, recordings, istate, bins=50, prange=(0,20), 
                          ma_thr=20, ma_state=3, flatten_is=False):
    """
    Plot histogram of time intervals between adjacent P-waves in any brainstate
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state to analyze
    bins, prange - no. histogram bins and histogram range for interval distribution plot
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    @Returns
    intervals_ms - array of all inter-P-wave intervals (ms)
    """
    states = {0:'all', 1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    intervals = []
    for rec in recordings:
        print('Getting P-waves for ' + rec + ' ...')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
               
        # load LFP, load and downsample P-wave indices
        LFP, p_idx = load_pwaves(ppath, rec)[0:2]
        (p_idx_dn, p_idx_dn_exact, p_count_dn, p_freq_dn) = downsample_pwaves(LFP, p_idx, sr=sr, 
                                                                              nbin=nbin, rec_bins=len(M))
        
        # get sequences of $istate
        if istate != 0:
            sseq = sleepy.get_sequences(np.where(M==istate)[0])
        else:
            sseq = [np.arange(0, len(M))]
        
        for seq in sseq:
            # get exactly downsampled P-wave indices occurring during each sequence
            seq_p = [i for i in p_idx_dn_exact if i >= seq[0] and i<= seq[-1]]
            for i, pi in enumerate(seq_p[0:-1]):
                # collect inter-P-wave intervals
                intervals.append(round((seq_p[i+1] - pi)*dt, 2))
    
    # bin and plot intervals
    p_hist, hbins = np.histogram(intervals, bins=bins, range=prange)
    x = np.linspace(min(hbins), max(hbins), len(p_hist))
    width = (x[-1] - x[0])/len(x)
    plt.figure()
    ax = plt.gca()
    ax.bar(x, p_hist, width=width, color='gray', edgecolor='black')
    ax.set_xlabel('Interval time (s)')
    ax.set_ylabel('Number of intervals')
    ax.set_title('Inter-P-Wave Interval Distribution (state = ' + states[istate] + ')')
    plt.show()
    
    intervals_ms = np.array(intervals)*1000
    print('')
    print(f'The 10th percentile of inter-P-wave intervals is {np.percentile(intervals_ms,10)} ms')
    print(f'The 15th percentile of inter-P-wave intervals is {np.percentile(intervals_ms,15)} ms')
    
    return intervals_ms


def pwave_overview(ppath, recordings, tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=False):
    """
    Plot P-wave frequency and brain state annotation for a given recording
    @Params
    ppath - base folder
    recordings - list of recordings
    tstart, tend - time (s) into recording to start and stop plotting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    @Returns
    None
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
        
    # get colormap for brain state annotation
    my_map, vmin, vmax = AS.hypno_colormap()
    
    for rec in recordings:
        
        # load sampling rate
        sr = AS.get_snr_pwaves(ppath, rec, default='NP')
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
    
        # load LFP and P-wave indices, downsample P-waves
        LFP, idx = load_pwaves(ppath, rec)[0:2]
        (p_idx_dn, p_idx_dn_exact, p_count_dn, p_freq_dn) = downsample_pwaves(LFP, idx, sr=sr, 
                                                                              nbin=nbin, rec_bins=len(M))
        p_freq_dn *= dt  # freq in units of P-waves / s 
        
        # define start and end points of plot
        istart = int(round(tstart/dt))
        if tend==-1:
            iend=len(M)-1
            M = M[istart : ]
            p_freq_dn = p_freq_dn[istart : ]
        else:
            iend = int(np.round((1.0*tend) / dt))
            M = M[istart : iend+1]
            p_freq_dn = p_freq_dn[istart : iend+1]
        
        # create plot
        fig, (ax1, ax2) = plt.subplots(figsize=(10,3), nrows=2, ncols=1, sharex = True, gridspec_kw={'height_ratios':[1,3]})
        t = np.linspace(istart*dt, iend*dt, len(p_freq_dn))
        ax1.pcolorfast(t, [0, 1], np.array([M]), vmin=vmin, vmax=vmax, cmap=my_map)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        ax2.plot(t, p_freq_dn, color='black')
        ax2.set_xlabel('Time (s)')
        plt.show()
    
    
def pwave_emg(ppath, recordings, emg_source, win, istate=[], rem_cutoff=True, recalc_amp=False, 
              nsr_seg=2, perc_overlap=0.75, recalc_highres=False, r_mu=[10,500], w0=-1, w1=-1, 
              dn=1, smooth=0, pzscore=0, tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=4, mouse_avg='mouse',
              exclude_noise=False, pemg2=False, p_iso=0, pcluster=0, clus_event='waves', ylim=[], sf=0,
              plaser=False, post_stim=0.1, lsr_iso=0, lsr_mode='pwaves', pplot=True, emg_calculated=None):
    """
    Plot average EMG amplitude surrounding P-waves
    @Params
    ppath - base folder
    recordings - list of recording names
    emg_source - use EMG signal ('raw') or EMG spectrogram ('msp') to calculate amplitude 
    win - time window (s) surrounding P-waves to plot EMG amplitude
    istate - brain state(s) to look for P-waves
    rem_cutoff - if True and 1 is in $istate, do not consider the last bin of each
                  REM period (to eliminate muscle twitch artifacts from waking up)
    recalc_amp - if True, recalculate EMG amplitude; if False, load saved EMG amp
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for mSP calculation
    recalc_highres - if True, recalculate mSP using $nsr_seg and $perc_overlap params
    r_mu - [min,max] frequencies summed in mSP
    w0, w1 - min, max frequencies for raw EMG filtering
             w0=-1 and w1=-1, no filtering; w0=-1, low-pass filter; w1=-1, high-pass filter
    dn, smooth - params for downsampling and smoothing raw EMG
    pzscore - method for z-scoring EMG amplitude (0=no z-scoring
                                                  1=z-score by recording
                                                  2=z-score by collected time window)
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    exclude_noise - if False, ignore manually annotated LFP noise indices
                    if True, exclude time bins containing LFP noise from analysis
    pemg2 - use amplitude of EMG channel 2
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    ylim - set y axis limit for graph
    plaser - if True, plot LFP surrounding laser-triggered P-waves (if $mode='pwaves') or
             laser pulses (if $mode='lsr')
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
    lsr_iso - if > 0, do not analyze laser pulses that are preceded within $lsr_iso s by 1+ P-waves
    lsr_mode - if 'pwaves', plot EMG amplitude surrounding laser-triggered vs. spontaneous P-waves
               if 'lsr', plot EMG amplitude surrounding successful vs. failed laser pulses
    pplot - if True, show EMG amplitude plot
    emg_calculated - list of data [EMG_amp, mnbin, mdt] if EMG amplitude is already calculated
    @Returns
    data_mx - matrix of EMG amplitudes (subject x time)
    mice - mouse/recording names or trial numbers
    """
    # clean data inputs
    if type(recordings) not in [list, tuple]:
        recordings = [recordings]
    if type(istate) not in [list, tuple]:
        istate = [istate]
        
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    if plaser:
        lsr_data = {s:{r:[] for r in recordings} for s in istate}
        spon_data = {s:{r:[] for r in recordings} for s in istate}
        success_data = {s:{r:[] for r in recordings} for s in istate}
        fail_data = {s:{r:[] for r in recordings} for s in istate}
        lsr_elim_counts = {'success':0, 'fail':0}
    else:
        pdata = {s:{r:[] for r in recordings} for s in istate}
    
    mice = []
    
    for rec in recordings:
        print('Getting P-wave EMG data for ' + rec + ' ...')
        idf = rec.split('_')[0]
        if idf not in mice:
            mice.append(idf)

        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load P-waves, isolate single/clustered waves
        if exclude_noise:
            # load LFP indices, exclude P-waves in areas of noisy LFP or EMG signal
            LFP, p_idx, lfp_noise_idx, emg_noise_idx = load_pwaves(ppath, rec, return_lfp_noise=True, 
                                                                   return_emg_noise=True)[0:4]
            noise_idx = np.sort(np.unique(np.concatenate((lfp_noise_idx, emg_noise_idx))))
            p_idx = np.setdiff1d(p_idx, noise_idx)
        else:
            LFP, p_idx = load_pwaves(ppath, rec)[0:2]
            lfp_noise_idx, emg_noise_idx, noise_idx = [np.array(()), np.array(()), np.array(())]
        # isolate single/clustered P-waves
        if p_iso and pcluster:
            print('ERROR: cannot accept both p_iso and pcluster arguments')
            return
        elif p_iso:
            p_idx = np.array(get_p_iso(p_idx, sr, win=p_iso))
        elif pcluster:
            p_idx = np.array(get_pclusters(p_idx, sr, win=pcluster, return_event=clus_event))
        
        # load laser and find laser-triggered P=waves
        if plaser:
            lsr = sleepy.load_laser(ppath, rec)
            ldata = list(get_lsr_pwaves(p_idx, lsr, post_stim, sr))

        # define start and end points of analysis
        istart = int(np.round(tstart*sr))
        iend = len(LFP)-1 if tend==-1 else int(np.round(tend*sr))
        iidx = np.intersect1d(np.where(p_idx >= istart)[0], np.where(p_idx <= iend)[0])
        p_idx = p_idx[iidx]
        if plaser:
            for i,d in enumerate(ldata):
                d = np.array(d)
                iidx = np.intersect1d(np.where(d >= istart)[0], np.where(d <= iend)[0])
                ldata[i] = d[iidx]
            lsr_pi, spon_pi, success_lsr, fail_lsr = ldata
            if exclude_noise:
                success_lsr = np.setdiff1d(success_lsr, noise_idx)
                fail_lsr = np.setdiff1d(fail_lsr, noise_idx)
            all_lsr = np.concatenate((success_lsr, fail_lsr))
        
            if lsr_iso > 0:
                # check for P-waves within the $lsr_iso window
                lwin = int(lsr_iso*sr)
                iso_elim = [l for l in all_lsr if np.intersect1d(p_idx, np.arange(l - lwin, l)).size > 0]
                lsr_elim_counts['success'] += len(np.intersect1d(success_lsr, iso_elim))
                success_lsr = np.setdiff1d(success_lsr, iso_elim)
                lsr_elim_counts['fail'] += len(np.intersect1d(fail_lsr, iso_elim))
                fail_lsr = np.setdiff1d(fail_lsr, iso_elim)
                all_lsr = np.concatenate((success_lsr, fail_lsr))
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = AS.adjust_brainstate(M, dt=2.5, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
        
        # cut off last bin(s) of REM sleep to avoid waking muscle twitch
        if rem_cutoff:
            remseq = sleepy.get_sequences(np.where(M==1)[0])
            elim_idx =  np.concatenate([rs[-2:] for rs in remseq])
            M[elim_idx] = 99
        
        # use inputted EMG amplitude
        if type(emg_calculated) == list and len(emg_calculated) == 3 and len(recordings)==1:
            EMG_amp, mnbin, mdt = emg_calculated
        # load/calculate EMG amplitude
        else:
            EMG_amp, mnbin, mdt = AS.emg_amplitude(ppath, rec, emg_source, recalc_amp=recalc_amp, 
                                                   nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                                   recalc_highres=recalc_highres, r_mu=r_mu, w0=w0, w1=w1, 
                                                   dn=dn, smooth=smooth, exclude_noise=exclude_noise, pemg2=pemg2)
        if pzscore==1:
            EMG_amp = scipy.stats.zscore(EMG_amp)
        # get EMG activity surrounding spontaneous/laser-triggered P-waves and laser pulses
        iwin1 = int(round(np.abs(win[0]) / mdt)); iwin2 = int(round(np.abs(win[1]) / mdt))
        
        # get indices of P-waves in $istate
        for s in istate:
            if plaser:
                # collect EMG amplitudes
                emg_data = [[], [], [], []]
                
                for iev, events in enumerate([lsr_pi, spon_pi, success_lsr, fail_lsr]):
                    sievents = [pi for pi in events if M[int(pi/nbin)]==s]
                    for i in sievents:
                        # eliminate events at start/end of recording
                        if i/mnbin > iwin1 and i/mnbin < len(EMG_amp)-iwin2:
                            d = EMG_amp[int(i/mnbin)-iwin1 : int(i/mnbin)+iwin2+1]
                            if pzscore==2:
                                d = scipy.stats.zscore(d)
                            emg_data[iev].append(d)
                            # eliminate events whose waveforms overlap with EMG noise
                            # ni = range(int(i-np.abs(win[0])*sr), i+int(win[1]*sr)+1)
                            # if len(np.intersect1d(ni, emg_noise_idx)) == 0:
                            #     a = 1
                            #     d = EMG_amp[int(i/mnbin)-iwin1 : int(i/mnbin)+iwin2+1]
                            #     emg_data[iev].append(d)
                                
                # store data in dictionaries
                lsr_data[s][rec] = emg_data[0]
                spon_data[s][rec] = emg_data[1]
                success_data[s][rec] = emg_data[2]
                fail_data[s][rec] = emg_data[3]
                
            elif not plaser:
                emg_data = []
                sievents = [pi for pi in p_idx if M[int(pi/nbin)]==s]
                for i in sievents:
                    # eliminate events at start/end of recording
                    if i/mnbin > iwin1 and i/mnbin < len(EMG_amp)-iwin2:
                        d = EMG_amp[int(i/mnbin)-iwin1 : int(i/mnbin)+iwin2+1]
                        if pzscore==2:
                            d = scipy.stats.zscore(d)
                        emg_data.append(d)
                        # eliminate events whose waveforms overlap with EMG noise
                        # ni = range(int(i-np.abs(win[0])*sr), i+int(win[1]*sr)+1)
                        # if len(np.intersect1d(ni, emg_noise_idx)) == 0:
                        #     d = EMG_amp[int(i/mnbin)-iwin1 : int(i/mnbin)+iwin2+1]
                        #     emg_data.append(d)
                pdata[s][rec] = emg_data
    
    if pplot:
        if plaser:
            if lsr_mode == 'pwaves':
                # EMG surrounding laser-triggered vs. spontaneous P-waves
                ddict, ddict2 = [lsr_data, spon_data]
                title, title2 = ['Laser-triggered P-waves', 'Spontaneous P-waves']
                c, c2 = ['blue','green']
            elif lsr_mode == 'lsr':
                # waveforms surrounding successful vs. failed laser pulses
                ddict, ddict2 = [success_data, fail_data]
                title, title2 = ['Successful laser pulses', 'Failed laser pulses']
                c, c2 = ['blue','red']
            # get mouse or trial-averaged matrix
            mx = [mx2d(ddict[s], mouse_avg=mouse_avg)[0] if any([len(v)>0 for v in ddict[s].values()]) else None for s in istate]
            mx2 = [mx2d(ddict2[s], mouse_avg=mouse_avg)[0] if any([len(v)>0 for v in ddict2[s].values()]) else None for s in istate]
        else:
            mx = [mx2d(pdata[s], mouse_avg=mouse_avg)[0] if any([len(v)>0 for v in pdata[s].values()]) else None for s in istate]
            title = 'All P-waves'
            c='black'
        if sf:
            mx = [AS.convolve_data(m, sf, axis='x') if m is not None else None for m in mx]
            if plaser:
                mx2 = [AS.convolve_data(m2, sf, axis='x') if m2 is not None else None for m2 in mx2]
                
        x = np.linspace(-np.abs(win[0]), win[1], mx[0].shape[1])
        # plot graphs
        fig = plt.figure()
        grid = matplotlib.gridspec.GridSpec(len(mx), int(plaser)+1, figure=fig)
        fig.set_constrained_layout_pads(w_pad=0.3/(int(plaser)+1), h_pad=3.0/(len(mx)**2))
        iplt = 0
        for i in range(len(mx)):
            # plot all P-waves OR laser P-waves/successful laser
            ax = fig.add_subplot(grid[iplt])
            if mx[i] is not None:
                data = np.nanmean(mx[i], axis=0)
                yerr = np.nanstd(mx[i], axis=0) / np.sqrt(mx[i].shape[0])
                ax.plot(x, data, color=c, linewidth=3)
                ax.fill_between(x, data-yerr, data+yerr, color=c, alpha=0.5)
                if i == len(mx)-1:
                    ax.set_xlabel('Time (s)')
            ax.set_ylabel('EMG Amp. (uV)')
            tmp = ' (' + states[istate[i]] + ')'
            ax.set_title(title + tmp)
            iplt += 1
            # plot spontaneous P-waves/failed laser
            if plaser:
                ax2 = fig.add_subplot(grid[iplt])
                if mx2[i] is not None:
                    data2 = np.nanmean(mx2[i], axis=0)
                    yerr2 = np.nanstd(mx2[i], axis=0) / np.sqrt(mx2[i].shape[0])
                    ax2.plot(x, data2, color=c2, linewidth=3)
                    ax2.fill_between(x, data2-yerr2, data2+yerr2, color=c2, alpha=0.5)
                    if i == len(mx2)-1:
                        ax2.set_xlabel('Time (s)')
                ax2.set_title(title2 + tmp)
                # make y axes equivalent
                if len(ylim)==2:
                    y = ylim
                else:
                    y = [min(ax.get_ylim()[0], ax2.get_ylim()[0]), max(ax.get_ylim()[1], ax2.get_ylim()[1])]
                ax.set_ylim(y)
                ax2.set_ylim(y)
                iplt += 1
    
    if plaser:
        return [lsr_data, spon_data, success_data, fail_data], mice
    else:
        return [pdata], mice
    
    
def detect_emg_twitches(ppath, rec, recalc_twitches=False, thres=99, thres_type='perc', 
                        thres_mode=1, thres_first=0, min_twitchdur=0.1, min_twitchsep=0.2, 
                        min_REMdur=10, rem_cutoff=5, recalc_amp=False, emg_source='raw', 
                        nsr_seg=2, perc_overlap=0.75, recalc_highres=False, r_mu=[10,500], 
                        w0=-1, w1=-1, dn=1, smooth=0, exclude_noise=True, pemg2=False):
    """
    Detect phasic muscle twitches during REM by calculating/thresholding EMG amplitude
    @Params
    ppath - base folder
    rec - name of recording
    recalc_twitches - if True, re-detect EMG twitches using loaded/calculated EMG
                       amplitude and given thresholding params
    thres - threshold for detecting phasic EMG twitches
    thres_type - threshold by given value ('raw'), signal mean + X standard deviations ('std'),
                 or by the Xth percentile of signal amplitude ('perc')
    thres_mode - threshold all REM periods (1) or each REM period individually (2)
    thres_first - if > 0, threshold by the first X s of REM sleep
    min_twitchdur - min. duration (s) of qualifying EMG twitch
    min_twitchsep - min. time (s) between neighboring twitches to qualify as separate events
    min_REMdur - min. duration (s) of REM sleep to analyze for EMG twitches
    rem_cutoff - no. seconds to exclude from end of each REM period to eliminate waking muscle twitch
    recalc_amp - if True, recalculate EMG amplitude; if False, load saved EMG amp
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for mSP calculation
    recalc_highres - if True, recalculate mSP using $nsr_seg and $perc_overlap params
    r_mu - [min,max] frequencies summed in mSP
    w0, w1 - min, max frequencies for raw EMG filtering
             w0=-1 and w1=-1, no filtering; w0=-1, low-pass filter; w1=-1, high-pass filter
    dn, smooth - params for downsampling and smoothing raw EMG
    exclude_noise - if True, exclude manually annotated sequences of EMG noise
    pemg2 - use amplitude of EMG channel 2
    @Returns
    twitches - annotation vector for EMG amplitude during qualifying REM sleep
                0 = no event, 1 = EMG twitch, -1 = EMG noise
    remidx - indices corresponding to each bin in $twitches annotation vector
    mnbin - no. Intan samples per bin in EMG amplitude vector
    mdt - no. seconds per bin in EMG amplitude vector
    settingsfile - path to saved dictionary with params used for twitch detection 
    """
    # try loading EMG twitches from file
    fpath = os.path.join(ppath, rec, 'emg_twitches.mat')
    if not recalc_twitches:
        if os.path.isfile(fpath):
            ET = so.loadmat(fpath)
            data = []
            keys = ['twitches','remidx','mnbin','mdt', 'EMG_amp', 'settingsfile']
            for k in keys:
                if k in ET.keys():
                    data.append(ET[k][0])
                else:
                    print('Some twitch info missing - recalculating twitches ...')
                    recalc_twitches = True
                    break
            if len(data) == 6:
                twitches, remidx, mnbin, mdt, EMG_amp, settingsfile = data
                mnbin, mdt = [float(mnbin), float(mdt)]
        else:
            print('No twitch info found - calculating twitches ...')
            recalc_twitches = True

    if recalc_twitches:
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
    
        # load brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        
        # if not recalculating EMG amplitude and twitch file exists
        ET = so.loadmat(fpath) if os.path.isfile(fpath) else {}
        # try loading EMG amplitude vector/mnbin/mdt from twitch file
        if not recalc_amp and all([k in ET.keys() for k in ['EMG_amp' 'mnbin','mdt']]):
            EMG_amp = ET['EMG_amp'][0]
            mnbin = float(ET['mnbin'][0])
            mdt = float(ET['mdt'][0])
        # otherwise, load from general EMG amplitude file or calculate anew 
        else:
            EMGdata = AS.emg_amplitude(ppath, rec, emg_source=emg_source, recalc_amp=recalc_amp, 
                                       nsr_seg=nsr_seg, perc_overlap=perc_overlap, r_mu=r_mu, 
                                       recalc_highres=recalc_highres, w0=w0, w1=w1, dn=dn, 
                                       smooth=smooth, exclude_noise=exclude_noise, pemg2=pemg2)
            EMG_amp, mnbin, mdt = EMGdata

        # get qualifying REM sequences
        remseq = sleepy.get_sequences(np.where(M==1)[0])
        remseq = [rs for rs in remseq if len(rs)*dt >= min_REMdur]
        # cut off last bin(s) to eliminate muscle twitch from waking up
        if rem_cutoff:
            icut = int(round(rem_cutoff / dt))
            remseq = [rs[0:-icut] for rs in remseq if len(rs)>icut]
        
        # get indices of REM sequences in EMG amp vector
        cf = nbin/mnbin 
        EMG_remseq = [np.arange(int(round(rs[0]*cf)), int(round(rs[-1]*cf+cf))) for rs in remseq]
        if EMG_remseq[-1][-1] == len(EMG_amp):
            EMG_remseq[-1] = EMG_remseq[-1][0:-1]
    
        # get indices to use for thresholding signal
        if thres_first > 0:
            ibins = int(round(thres_first/mdt))
            istart = [emgseq[0:ibins] for emgseq in EMG_remseq]
        
        # threshold all REM bins
        if thres_mode==1:
            ridx = np.concatenate(EMG_remseq)
            
            # threshold by first X seconds of each REM bout OR whole REM bouts
            if thres_first > 0:
                th_ridx = np.concatenate(istart)
            else:
                th_ridx = np.array(ridx)
                
            if thres_type == 'raw':         # threshold signal by raw value
                th = float(thres)
            elif thres_type == 'std':       # threshold signal by X standard deviations
                mn = np.nanmean(EMG_amp[th_ridx])
                std = np.nanstd(EMG_amp[th_ridx])
                th = mn + thres*std
            elif thres_type == 'perc':      # threshold signal by Xth percentile
                th = np.nanpercentile(EMG_amp[th_ridx], thres)

            # get indices of twitch sequences in EMG amplitude vector
            idx = np.where(EMG_amp[ridx] > th)[0]
            itwitch = ridx[idx]
            
        # separately threshold each REM period
        elif thres_mode==2:
            itwitch = np.array(())
            
            for i in range(len(EMG_remseq)):
                # indices of REM sequence in EMG signal
                seqi = EMG_remseq[i]
                if thres_first > 0:
                    th_ridx = istart[i]
                else:
                    th_ridx = np.array(seqi)
            
                if thres_type == 'raw':
                    th = float(thres)
                elif thres_type == 'std':
                    mn = np.nanmean(EMG_amp[th_ridx])
                    std = np.nanstd(EMG_amp[th_ridx])
                    th = mn + thres*std
                elif thres_type == 'perc':
                    th = np.nanpercentile(EMG_amp[th_ridx], thres)
                
                # get indices of twitch sequences in REM period
                idx = np.where(EMG_amp[seqi] > th)[0]
                itwitch = np.concatenate([itwitch, seqi[idx]])
            itwitch = itwitch.astype(int)
        
        # combine neighboring twitch indices into sequences using $min_twitchsep param
        ibreak = int(round(min_twitchsep / mdt))
        twitch_seqs = sleepy.get_sequences(itwitch, ibreak)
        # fill in gaps of $ibreak indices, eliminate short twitches under $min_twitchdur s
        twitch_seqs = [np.arange(twseq[0],twseq[-1]+1) for twseq in twitch_seqs]
        twitch_seqs = [twseq for twseq in twitch_seqs if len(twseq)*mdt >= min_twitchdur]
        twitch_idx = np.concatenate(twitch_seqs)
        
        # get noise indices (NaNs) in EMG amplitude vector
        EMGamp_noise = np.nonzero(np.isnan(EMG_amp))[0]
        # create twitch vector with 1's as twitches and -1's as noise
        twitches = np.zeros(len(EMG_amp))
        twitches[twitch_idx] = 1
        twitches[EMGamp_noise] = -1
        
        # save REM indices and corresponding twitch train annotation
        remidx = np.concatenate(EMG_remseq)
        #twitches = twitch_train[remidx]
        settingsfile = os.path.join(ppath, rec, 'tmp_twitch_settings.pkl')
        so.savemat(fpath, {'twitches'     : twitches,   # twitch annotation vector for EMG_amp
                           'remidx'       : remidx,         # indices of REM sleep in twitch annot. vector
                           'mnbin'        : mnbin,          # no. Intan samples per EMG amp bin
                           'mdt'          : mdt,            # no. seconds per EMG amp bin
                           'EMG_amp'      : EMG_amp,        # EMG amplitude vector (noise=NaNs)
                           'settingsfile' : settingsfile})  # saved file with settings for twitches/EMG amp calculation
        # save settings in temporary file
        ddict = {'ampsrc'           : emg_source,
                 'min_dur'          : min_REMdur,
                 'rem_cutoff'       : rem_cutoff,
                 'w0_raw'           : w0,
                 'w1_raw'           : w1,
                 'dn_raw'           : dn,
                 'sm_raw'           : smooth,
                 'nsr_seg_msp'      : nsr_seg,
                 'perc_overlap_msp' : perc_overlap,
                 'r_mu'             : r_mu,
                 'thres'            : thres,
                 'thres_type'       : thres_type,
                 'thres_mode'       : thres_mode,
                 'thres_first'      : thres_first,
                 'min_twitchdur'    : min_twitchdur,
                 'min_twitchsep'    : min_twitchsep,
                 'annot'            : []}
        with open(settingsfile, mode='wb') as f:
            pickle.dump(ddict, f)
        
    return twitches, remidx, mnbin, mdt, EMG_amp, settingsfile


def emg_twitch_freq(ppath, recordings, recalc_twitches=False, thres=99, thres_type='perc', 
                    thres_mode=1, thres_first=0, min_twitchdur=0.1, min_twitchsep=0.2, 
                    min_REMdur=10, rem_cutoff=5, recalc_amp=False, emg_source='raw', 
                    nsr_seg=2, perc_overlap=0.75, recalc_highres=False, r_mu=[10,500], 
                    w0=-1, w1=-1, dn=1, smooth=0, exclude_noise=True, pemg2=False,
                    tstart=0, tend=-1, avg_mode='all', mouse_avg='mouse'):
    """
    Plot frequency of EMG twitches during REM sleep
    @Params
    ppath - base folder
    recordings - list of recordings
    recalc_twitches to pemg2 - see documentation for $detect_emg_twitches
    tstart, tend - time (s) into recording to start and stop collecting data
    avg_mode - for each recording, twitch freq is computed as the average across 
               all REM sleep ('all'), or from the averages of each REM period ('each')
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    @Returns
    twitch_freq_dict - dictionary of EMG twitch frequencies for each recording
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    
    data_dict = {rec:{} for rec in recordings}
    
    for rec in recordings:
        # load or calculate EMG twitches
        twitchdata = detect_emg_twitches(ppath, rec, recalc_twitches=recalc_twitches, 
                                         thres=thres, thres_type=thres_type, 
                                         thres_mode=thres_mode, thres_first=thres_first, 
                                         min_twitchdur=min_twitchdur, min_twitchsep=min_twitchsep, 
                                         min_REMdur=min_REMdur, rem_cutoff=rem_cutoff, 
                                         recalc_amp=recalc_amp, emg_source=emg_source, 
                                         nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                         recalc_highres=recalc_highres, r_mu=r_mu, w0=w0, w1=w1, 
                                         dn=dn, smooth=smooth, exclude_noise=exclude_noise, pemg2=pemg2)
        twitches, remidx, mnbin, mdt, EMG_amp, settingsfile = twitchdata
        
        # find REM sequences, collect no. twitches and min. spent in non-noisy REM
        remseqs = sleepy.get_sequences(remidx)
        REMdurs = []
        num_twitches = []
        for rseq in remseqs:
            noise_idx = np.where(twitches[rseq]==-1)[0]
            # get no. minutes of non-noisy REM sleep
            durREM = ((len(rseq) - len(noise_idx))*mdt)/60
            REMdurs.append(durREM)
            # get EMG twitch sequences
            uA = sleepy.get_sequences(np.where(twitches[rseq]==1)[0])
            if uA[0].size > 0:
                num_twitches.append(len(uA))
            else:
                num_twitches.append(0)
        # eliminate 100% noisy REM periods, calculate twitch frequency in each REM period
        num_twitches, REMdurs = zip(*[[i,j] for i,j in zip(num_twitches, REMdurs) if j!=0])
        num_twitches, REMdurs = [np.array(num_twitches), np.array(REMdurs)]
        data_dict[rec] = {'remDur':REMdurs, 'ntwitch':num_twitches}
    
    twitch_freq_dict = {}
    for rec in recordings:
        # get average twitch frequency across all REM sleep in recording
        if avg_mode == 'all' and mouse_avg != 'trial':
            f = np.array([np.sum(data_dict[rec]['ntwitch']) / np.sum(data_dict[rec]['remDur'])])
        # get twitch frequency for each REM period
        else:
            f = np.divide(data_dict[rec]['ntwitch'], data_dict[rec]['remDur'])
        twitch_freq_dict[rec] = f
    data, labels = mx1d(twitch_freq_dict, mouse_avg=mouse_avg)
    return twitch_freq_dict
        
        
                        
##############         ANALYSIS AND PLOTTING FUNCTIONS         ##############

def cross_correlate(dff, LFP, dn, iwin, ptrain, dffnorm=True):
    """
    Calculate linear cross-correlation between DF/F and LFP signals
    @Params
    dff - N-length DF/F signal
    LFP - N-length LFP signal or P-wave train
    dn - downsample $dff and $LFP vectors by X bins
    iwin - return X bins on either side of center of the cross-correlation
    ptrain - if False, $LFP is a raw LFP signal (continuous)
             if True, $LFP is a vector of 1's / P-waves and 0's / no P-waves (categorical)
    dffnorm - normalize DF/F signal by its mean
    @Returns
    yy - centered cross-correlation of $dff and $LFP inputs
    dffdn, LFPdn - downsampled inputs
    """
    # downsample DF/F signal, normalize to zero mean
    dffdn = AS.downsample_vec(dff, dn)
    if dffnorm:
        dffdn -= dffdn.mean()
    # downsample LFP signal
    LFPdn = AS.downsample_vec(LFP, dn)
    #LFPdn = LFP
    if ptrain:  # restore P-wave train of 1's and 0's
        LFPdn = np.ceil(LFPdn).astype('int')
        # For each timepoint relative to a P-wave, the correlation value is the sum
        # of the DF/F values at each instance in the recording which fulfills that
        # criteria (i.e. has a P-wave X seconds before or after). To normalize, divide
        # each element in the correlation vector by the number of instances that were 
        # summed to produce it; this puts the scale of the data back to original units
        # of DF/F, or percent change from baseline
        norm = np.sum(LFPdn[0:-1])
        #norm = 1.
    else:       # normalize raw LFP signal to zero mean
        LFPdn -= LFPdn.mean()
        norm = np.nanstd(dffdn) * np.nanstd(LFPdn)
    # cross-correlate signals
    xx = scipy.signal.correlate(dffdn[1:], LFPdn[0:-1]) / norm
    ii = np.arange(len(xx)/2 - iwin, len(xx)/2 + iwin + 1).astype('int')
    yy = xx[ii]
    
    return yy, dffdn, LFPdn


def dff_pwaves_corr(ppath, recordings, win=2, istate=1, dffnorm=True, ptrain=True, 
                    ptrial=True, dn=1, sf=0, min_dur=30, ma_thr=20, ma_state=3, flatten_is=4, p_iso=0,
                    mouse_avg='trial', jitter=True, jtr_win=10, use405=False, seed=0, base_int=0.5, 
                    baseline_start=0, baseline_end=-1, pplot=True, print_stats=True):
    """
    Get cross-correlation of DF/F signal and P-wave signal
    @Params
    ppath - base folder
    recordings - list of recordings
    win - time window (s) to show cross-corrrelation
    istate - brain state to analyze
    dffnorm - if True, normalize DF/F signal by its mean
    ptrain - if True, use P-wave 'spike train' (1=wave, 0=no wave) for cross-correlation
             if False, use LFP signal for cross-correlation
    ptrial - if True, consider each P-wave as a 'trial' for single-trial averaging
             if False, consider each state sequence as a 'trial'
    dn, sf - downsampling and smoothing factors for plot
    min_dur - minimum duration (s) of brain state episode
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    p_iso - inter-P-wave interval threshold for analyzing isolated P-waves
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    jitter - if True, also analyze randomly shifted P-wave indices (as a control)
    jtr_win - randomly shift P-wave indices by +/- $jtr_win seconds
    use405 - if True, analyze 'baseline' 405 signal instead of fluorescence signal
    seed - if integer, set seed for shifting laser indices
    base_int - for timecourse stats, size of consecutive time bins (s) to compare
    baseline_start - no. of bins into timecourse to start "baseline" bin
    baseline_end - no. of bins into timecourse to end comparisons
    pplot - if True, show plots
    print_stats - if True, show results of repeating t-tests w/ Bonferroni correction
    @Returns
    [CC_mx, labels] - matrix (subjects x time) of DF/F vs P-wave cross-correlations, mouse/trial labels
    [CC_mx_jtr, jlabels] - matrix of DF/F vs shifted P-wave cross-correlations, mouse/trial labels
    """
    if ptrain==False and ptrial==False:
        jitter = False  # nothing to jitter when analysis doesn't use P-wave indices
    
    cross_corr = {'dff':{}, 'lfp':{}, 'CC':{}}
    cross_corr_jtr = {'dff':{}, 'lfp':{}, 'CC':{}}
    
    for rec in recordings:
        print('Getting data for ' + rec + ' ...')
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        iwin = int(win * sr / dn)
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
        
        # load DF/F signal, LFP signal, and P-wave indices
        dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dff']
        dff *= 100.
        LFP, idx = load_pwaves(ppath, rec)[0:2]
        if p_iso:
            idx = get_p_iso(idx, sr, win=p_iso)
        LFP_train = np.zeros(len(LFP))
        LFP_train[idx] = 1
        
        # cross-correlate signals for each brainstate sequence
        if istate > 0:
            sseq = sleepy.get_sequences(np.where(M==istate)[0])
            sseq = [seq for seq in sseq if len(seq)*2.5 >= min_dur]
            CC = []
            Cdff = []
            Clfp = []
            
            CC_jtr = []
            Cdff_jtr = []
            Clfp_jtr = []
            for seq in sseq:
                # get real and jittered P-wave indices during sequence
                i = seq[0] * nbin
                j = seq[-1] * nbin + nbin + 1
                #ptrain = LFP_train[i:j]
                pseq = LFP_train[i:j]
                pidx = np.where(pseq==1)[0]
                # pidx = np.where(LFP_train[i:j]==1)[0]
                if jitter:
                    np.random.seed(seed)
                    # randomly jitter each P-wave in sequence up to +/- $jtr_win s
                    jtr = np.random.randint(-int(jtr_win*sr), int(jtr_win*sr), size=len(pidx))
                    jidx = pidx + jtr
                    # if P-wave jittered off edge of LFP seq, shift same magnitude in opposite direction
                    jidx = [pi - jt if ji>=(j-i) or ji<0 else ji for pi,ji,jt in zip(pidx,jidx,jtr)]
                    jidx = np.array(jidx).astype('int')
                    # if P-wave jittered off opposite edge, select random available idx
                    iout = np.concatenate((np.where(jidx<0)[0], np.where(jidx>=(j-i))[0]))
                    if iout.size > 0:
                        iavail = np.setdiff1d(range(j-i),jidx)
                        irand = np.random.choice(iavail, size=len(iout), replace=False)
                        jidx[iout] = irand
                    jseq = np.zeros((j-i))
                    jseq[jidx] = 1
                    
                # get DFF and LFP signals during each sequence
                dffcut = dff[i:j]
                if ptrain:             ## correlate P-wave train (1's and 0's)
                    if pidx.size > 0:
                        LFPcut = pseq
                        if jitter:
                            LFPcut_jtr = jseq
                    elif pidx.size == 0:
                        # do not correlate DF/F with zero vector (no P-waves)
                        continue
                else:                  ## correlate raw LFP signal
                    LFPcut = LFP[i:j]
                    if jitter:
                        LFPcut_jtr = LFP[i:j]
                ### treat each P-wave in $istate as single trial
                if ptrial:
                    for pi in pidx:
                        if pi >= iwin and pi < len(LFPcut)-iwin:
                            dffp = dffcut[pi-iwin : pi+iwin+1]
                            LFPp = LFPcut[pi-iwin : pi+iwin+1]
                            x,d,l = cross_correlate(dffp, LFPp, dn, iwin, ptrain, dffnorm=dffnorm)
                            CC.append(x)
                            Cdff.append(d)
                            Clfp.append(l)
                    if jitter:
                        for pi,ji in zip(pidx,jidx):
                            if pi >= iwin and ji >= iwin and pi < len(LFPcut_jtr)-iwin and ji < len(LFPcut_jtr)-iwin:
                                if ptrain:
                                    # DFF surrounding jitter indices vs jitter train (1 in center)
                                    dffj = dffcut[ji-iwin : ji+iwin+1]
                                else:
                                    # DFF surrounding P-waves vs LFP surrounding jitter indices
                                    dffj = dffcut[pi-iwin : pi+iwin+1]
                                LFPj = LFPcut_jtr[ji-iwin : ji+iwin+1]
                                y,d,l = cross_correlate(dffj, LFPj, dn, iwin, ptrain, dffnorm=dffnorm)
                                CC_jtr.append(y)
                                Cdff_jtr.append(d)
                                Clfp_jtr.append(l)
                ### treat each brainstate sequence as single trial
                elif not ptrial:
                    x,d,l = cross_correlate(dffcut, LFPcut, dn, iwin, ptrain, dffnorm=dffnorm)
                    CC.append(x)
                    Cdff.append(d)
                    Clfp.append(l)
                    if jitter:
                        # cross-correlate true DF/F with jittered LFP
                        y,d,l = cross_correlate(dffcut, LFPcut_jtr, dn, iwin, ptrain, dffnorm=dffnorm)
                        CC_jtr.append(y)
                        Cdff_jtr.append(d)
                        Clfp_jtr.append(l)
            # collect correlations in data dictionaries
            cross_corr['CC'][rec] = np.array(CC)
            cross_corr['dff'][rec] = np.array(Cdff, dtype='object')
            cross_corr['lfp'][rec] = np.array(Clfp, dtype='object')
            if jitter:
                cross_corr_jtr['CC'][rec] = np.array(CC_jtr)
                cross_corr_jtr['dff'][rec] = np.array(Cdff_jtr, dtype='object')
                cross_corr_jtr['lfp'][rec] = np.array(Clfp_jtr, dtype='object')
        # cross-correlate entire recording signals
        elif istate <= 0:
            CC = []
            Cdff = []
            Clfp = []
            
            CC_jtr = []
            Cdff_jtr = []
            Clfp_jtr = []

            if jitter:
                # randomly jitter each P-wave up to +/- $jtr_win s
                np.random.seed(seed)
                LFP_jtrain = np.zeros(len(LFP_train))
                jtr = np.random.randint(-int(jtr_win*sr), int(jtr_win*sr), size=len(idx))
                jidx = idx + jtr
                # if P-wave jittered off edge of recording, shift same magnitude in opposite direction
                jidx = [pi - jt if ji>=len(LFP_jtrain) or ji<0 else ji for pi,ji,jt in zip(idx,jidx,jtr)]
                jidx = np.array(jidx).astype('int')
                LFP_jtrain[jidx] = 1
            if ptrain:
                LFPc = np.array(LFP_train)
                if jitter:
                    LFPc_jtr = np.array(LFP_jtrain)
            else:
                LFPc = np.array(LFP)
                if jitter:
                    LFPc_jtr = np.array(LFP)
            # make sure signal vectors are equal lengths
            m = np.min([dff.shape[0], LFPc.shape[0]])
            dffc = dff[0:m]
            LFPc = LFPc[0:m]
            if jitter:
                LFPc_jtr = LFPc_jtr[0:m]
            ### treat each P-wave as single trial
            if ptrial:
                for pi in idx:
                    if pi >= iwin and pi < (m-iwin):
                        dffp = dffc[pi-iwin : pi+iwin+1]
                        LFPp = LFPc[pi-iwin : pi+iwin+1]
                        x,d,l = cross_correlate(dffp, LFPp, dn, iwin, ptrain, dffnorm=dffnorm)
                        CC.append(x)
                        Cdff.append(d)
                        Clfp.append(l)
                if jitter:
                    for ji in jidx:
                        if ji >= iwin and ji < (m-iwin):
                            dffj = dffc[ji-iwin : ji+iwin+1]
                            LFPj = LFPc_jtr[ji-iwin : ji+iwin+1]
                            y,d,l = cross_correlate(dffj, LFPj, dn, iwin, ptrain, dffnorm=dffnorm)
                            CC_jtr.append(y)
                            Cdff_jtr.append(d)
                            Clfp_jtr.append(l)
            ### treat each recording as single trial
            elif not ptrial:
                x,d,l = cross_correlate(dffc, LFPc, dn, iwin, ptrain, dffnorm=dffnorm)
                CC.append(x)
                Cdff.append(d)
                Clfp.append(l)
                if jitter:
                    y,d,l = cross_correlate(dffc, LFPc_jtr, dn, iwin, ptrain, dffnorm=dffnorm)
                    CC_jtr.append(y)
                    Cdff_jtr.append(d)
                    Clfp_jtr.append(l)
            # collect correlations in data dictionaries
            cross_corr['CC'][rec] = np.array(CC)
            cross_corr['dff'][rec] = np.array(Cdff)
            cross_corr['lfp'][rec] = np.array(Clfp)
            if jitter:
                cross_corr_jtr['CC'][rec] = np.array(CC_jtr)
                cross_corr_jtr['dff'][rec] = np.array(Cdff_jtr)
                cross_corr_jtr['lfp'][rec] = np.array(Clfp_jtr)
                
    # get 2D matrix of subjects x time
    CC_mx, labels = mx2d(cross_corr['CC'], mouse_avg=mouse_avg)
    if jitter:
        CC_mx_jtr, jlabels = mx2d(cross_corr_jtr['CC'], mouse_avg=mouse_avg)
                
    if sf:
        CC_mx = AS.convolve_data(CC_mx, sf, axis='x')
        if jitter:
            CC_mx_jtr = AS.convolve_data(CC_mx_jtr, sf, axis='x')
    
    # get timecourse stats for cross-correlation of DF/F vs P-wave train
    if ptrain:
        if print_stats:
            print('P-wave Train vs DF/F\n')
        baseline_start = int(baseline_start*sr)
        baseline_end = int(baseline_end*sr) if baseline_end != -1 else -1
        df = stats_timecourse(CC_mx, pre=-np.abs(win), post=np.abs(win), sr=sr/dn, 
                              base_int=base_int, baseline_start=baseline_start,
                              baseline_end=baseline_end, print_stats=print_stats)
        if jitter:
            if print_stats:
                print('\nJittered P-wave Train vs DF/F\n')
            df_jtr = stats_timecourse(CC_mx_jtr, pre=-np.abs(win), post=np.abs(win), sr=sr/dn,
                                      base_int=base_int, baseline_start=baseline_start,
                                      baseline_end=baseline_end, print_stats=print_stats)

    if pplot:
        t = np.linspace(-win, win, CC_mx.shape[1])
        data = np.nanmean(CC_mx, axis=0)
        sem = np.nanstd(CC_mx, axis=0) / np.sqrt(CC_mx.shape[0])
        fig = plt.figure()
        ax1 = plt.gca()
        # plot cross-correlation of DF/F vs LFP/P-wave train
        ax1.plot(t, data, color='black')
        ax1.fill_between(t, data-sem, data+sem, color='gray', alpha=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Corr. DF/F vs. LFP')
        ax1.set_title(f'Each {"P-wave" if ptrial else "REM sequence"} = single trial')
        if jitter:
            # plot cross-correlation of DF/F vs jittered P-wave train
            data_jtr = np.nanmean(CC_mx_jtr, axis=0)
            sem_jtr = np.nanstd(CC_mx_jtr, axis=0) / np.sqrt(CC_mx_jtr.shape[0])
            ax1.plot(t, data_jtr, color='red')
            ax1.fill_between(t, data_jtr-sem_jtr, data_jtr+sem_jtr, color='red', alpha=0.5)
    if jitter:
        return [CC_mx, labels], [CC_mx_jtr, jlabels]
    else:
        return [CC_mx, labels], [None, None]


def dff_timecourse(ppath, recordings, istate, dff_win=[-10,10], plotMode='0',
                   pzscore=0, z_win=False, p_iso=0, pcluster=0, 
                   clus_event='waves', vm=[], psmooth=0, dn=0, sf=0,
                   mouse_avg='mouse', base_int=10, baseline_start=0, 
                   baseline_end=-1, ma_thr=20, ma_state=3, flatten_is=False, 
                   tstart=0, tend=-1, jitter=0, use405=False, ylim=[], ylim2=[], 
                   print_stats=True, pplot=True, exclude={}):
    """
    Get average DF/F signal surrounding P-waves in any brain state
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state to analyze
    dff_win - time window (s) surrounding P-waves
    plotMode - type of plot(s) to show
              't' - plot timecourse of averaged DF/F signal, +/- SEM
              'h' - plot heatmap of all P-wave trials
              '0' - bar plot of DF/F signal during 2 s preceding vs following P-waves
                  '1' = color-coded dots for mice; '2' = black dots for mice
                  '3' = color-coded lines for mice; '4' = black lines for mice
    pzscore - normalization method for graphs
               0 - use raw DF/F values
               1 - z-score DF/F values by entire recording
               2 - z-score DF/F values within $dff_win or $z_win time window
    z_win - custom window for z-scoring and plotting DF/F signals on different time scales
             * e.g. for dff_win=[-1,1] and z_win=[-10,10], values are z-scored by the 20 s
               interval surrounding P-waves, but plot shows the 2 s surrounding interval
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    vm, psmooth - control saturation and smoothing of trial heatmap
    dn - downsample DF/F timecourse & heatmap along time dimension (for a manageable # of plot points)
    sf - smooth DF/F timecourse along time dimension
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    base_int - for timecourse stats, size of consecutive time bins (s) to compare
    baseline_start - no. of bins into timecourse to start "baseline" bin
    baseline_end - no. of bins into timecourse to end comparisons
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    tstart, tend - time (s) into recording to start and stop collecting data
    jitter - if > 0, randomly shift P-wave indices by +/- $jitter seconds (as a control)
    use405 - if True, analyze 'baseline' 405 signal instead of fluorescence signal
    ylim, ylim2 - set y axis limits of timecourse and bar graph
    print_stats - if True, show results of repeating t-tests w/ Bonferroni correction
    pplot - if True, plot graphs
    @Returns
    ddict - data dictionary containing DF/F signals surrounding each P-wave
            (keys=recordings, values=lists of timecourses)
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if type(pzscore) in [list, tuple]:
        if len(set(pzscore)) > 1:
            print(f'### WARNING - {pzscore[0]} used as z-scoring method for all graphs')
        pzscore = pzscore[0]
    if pzscore != 2 or type(z_win) not in [list, tuple] or len(z_win) != 2:
        z_win = dff_win
    
    pwave_trig = {rec:[] for rec in recordings}
    
    for rec in recordings:
        print('Getting data for ' + rec + ' ...')
        idf = re.split('_', rec) [0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        iwin1, iwin2 = get_iwins(dff_win, sr)
        zwin1, zwin2 = get_iwins(z_win, sr)
        
        # calculate DF/F signal using high cutoff frequency for 465 signal
        # and very low cutoff frequency for 405 signal
        if use405:
            # load artifact control 405 signal
            a405 = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['405']
            dff = sleepy.my_lpfilter(a405, 2/(0.5*sr), N=4)
        else:
            # load DF/F signal
            AS.calculate_dff(ppath, rec, wcut=10, wcut405=2, shift_only=False)
            dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dff']
        dff_z = (dff-dff.mean()) / dff.std()
        dff = dff*100.0
        
        # load and adjust brainstate annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
        
        # load P-waves
        LFP, idx = load_pwaves(ppath, rec)[0:2]
        # isolate single/clustered P-waves
        if p_iso and pcluster:
            print('ERROR: cannot accept both p_iso and pcluster arguments')
            return
        elif p_iso:
            idx = get_p_iso(idx, sr, win=p_iso, order=[1,0])
        elif pcluster:
            idx = get_pclusters(idx, sr, win=pcluster, return_event=clus_event)
        
        # randomly jitter P-wave indices as a control
        if jitter:
            np.random.seed(0)
            jter = np.random.randint(-jitter*sr, jitter*sr, len(idx))
            idx = idx + jter

        # if $istate is an integer, find all P-wave indices in that state
        if not isinstance(istate, str):
            if istate == 0:
                p_idx = idx
            else:
                idx_dn = np.floor(idx/nbin).astype('int')
                if isinstance(istate, int):
                    istate = [istate]
                sidx = [np.where(M==s)[0] for s in istate if rec not in exclude.get(s,[])]
                if len(sidx) == 0:
                    continue
                else:
                    sidx = np.concatenate(sidx)
                idx_dn_state = np.intersect1d(idx_dn, sidx)
                iidx = np.nonzero(np.in1d(idx_dn, idx_dn_state))[0]
                p_idx = idx[iidx]
                
        # define start and end points of analysis
        istart = int(np.round(tstart*sr))
        iend = len(LFP)-1 if tend==-1 else int(np.round(tend*sr))
        z1, z2 = max([iwin1,zwin1]), max([iwin2,zwin2])
        w1, w2 = max(istart, z1), min(iend, len(dff)-z2)
        p_idx = p_idx[np.where((p_idx >= w1) & (p_idx < w2))[0]]
        
        # collect DF/F signal surrounding each P-wave
        dff_rec = []
        for pi in p_idx:
            if pzscore == 2:
                zdata = dff[pi-zwin1 : pi+zwin2]
                data = (dff[pi-iwin1 : pi+iwin2] - zdata.mean()) / zdata.std()
            elif pzscore == 1:
                data = dff_z[pi-iwin1 : pi+iwin2]
            elif pzscore == 0:
                data = dff[pi-iwin1 : pi+iwin2]
            dff_rec.append(data)
        pwave_trig[rec] = dff_rec
    
    # create descriptive title
    title1 = f'STATE = {istate}'
    if p_iso:
        title1 += f'; p_iso={p_iso} s'
    elif pcluster:
        title1 += f'; pcluster={pcluster} s, cluster event={clus_event}'
    title2 = f'\nmouse avg={mouse_avg}'
    if pzscore == 1:
        title2 += '; z-scored by recording'
    elif pzscore == 2:
        if dff_win == z_win:
            title2 += '; z-scored by time window'
        else:
            title2 += f'; z-scored by {-np.abs(z_win[0])} to +{np.abs(z_win[1])} s window'
    title = title1 + title2
    ylabel = '$\Delta$ F/F (%)' if pzscore==0 else '$\Delta$ F/F (z-scored)'
    
    ###   GRAPHS   ###
    if 't' in plotMode:  # timecourse of DF/F signal
        # get matrix of subjects (trials/recordings/mice) x time bins
        mx = mx2d(pwave_trig, mouse_avg, d1_size=(iwin1+iwin2))[0]
    
        if print_stats:
            # stats: when does activity becomes significantly different from baseline?
            print('')
            stat_df=stats_timecourse(mx, pre=-iwin1/sr, post=iwin2/sr, sr=sr, 
                                base_int=base_int, baseline_start=int(baseline_start*sr),
                                baseline_end=int(baseline_end*sr) if baseline_end != -1 else -1,
                                print_stats=print_stats)
            print('')

        # smooth/downsample along time dimension
        if sf:
            mx = AS.convolve_data(mx, sf, axis='x')
        if dn:
            mx = AS.downsample_mx(mx, dn, axis='x')
        
        if pplot:
            # plot timecourse
            data_mean = np.nanmean(mx, axis=0)
            data_yerr = np.nanstd(mx, axis=0) / np.sqrt(mx.shape[0])
            plt.figure()
            ax = plt.gca()
            t = np.linspace(-np.abs(dff_win[0]), np.abs(dff_win[1]), len(data_mean))
            ax.plot(t, data_mean, color='black')
            ax.fill_between(t, data_mean-data_yerr, data_mean+data_yerr, 
                            color='black', alpha=0.3)
            if len(ylim) == 2:
                ax.set_ylim(ylim)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            
    if 'h' in plotMode:  # heatmap of individual P-wave trials
        mx2 = mx2d(pwave_trig, 'trial', d1_size=(iwin1+iwin2))[0]
        # smooth heatmap
        if psmooth:
            mx2 = AS.convolve_data(mx2, psmooth)
        # downsample along time dimension
        if dn:
            mx2 = AS.downsample_mx(mx2, dn, axis='x')
        
        if pplot:
            # plot heatmap
            plt.figure()
            ax = plt.gca()
            t = np.linspace(-np.abs(dff_win[0]), np.abs(dff_win[1]), mx2.shape[1])
            trial_no = np.arange(1, mx2.shape[0]+1)
            im = ax.pcolorfast(t, trial_no, mx2, cmap='bwr')
            if len(vm) == 2:
                im.set_clim(vm)
            plt.colorbar(im, ax=ax, pad=0.0)
            ax.set_ylabel('Trial no.')
            ax.set_xlabel('Time (s)')
            ax.set_title(title)
    
    if '0' in plotMode and pplot:  # bar graph of DF/F activity preceding vs following P-wave
        t = np.linspace(-np.abs(dff_win[0]), np.abs(dff_win[1]), iwin1+iwin2)
        bar_win = 2  # number of seconds before and after P-wave to compare
        ipre = np.where((t>=(-bar_win)) & (t<0))[0]
        ipost = np.where((t>=0) & (t<=bar_win))[0]
        # create dataframe with average pre and post-P-wave DF/F values
        rec_dict = mx2d_dict(pwave_trig, 'recording', d1_size=iwin1+iwin2)
        df = pd.DataFrame(columns=['mouse', 'recording', 'state', 'group', 'dff'])
        for rec in recordings:
            idf = rec.split('_')[0]
            ppre, ppost = [rec_dict[rec][:,ii].mean(axis=1) for ii in [ipre,ipost]]
            df = pd.concat([df, pd.DataFrame({'mouse':idf,
                                              'recording':rec,
                                              'state':istate,
                                              'group':np.repeat(['pre','post'], len(ppre)),
                                              'dff':np.concatenate((ppre,ppost))})],
                           axis=0, ignore_index=True)
        
        # plot bar graph
        plt.figure()
        ax = plt.gca()
        if mouse_avg in ['mouse', 'recording']:
            # plot each mouse/recording individually
            df2 = df.groupby([mouse_avg, 'group']).sum().reset_index()
            df2.sort_values(['mouse','group'], ascending=[True,False], ignore_index=True, inplace=True)
            sns.barplot(data=df2, x='group', y='dff', palette=['lightgray','darkgray'], 
                        errorbar='se', ax=ax)
            lines = sns.lineplot(data=df2, x='group', y='dff', hue=mouse_avg, 
                                 linewidth=2, legend=False, ax=ax)
            _ = [l.set_color('black') for l in lines.get_lines()]
            # paired t-test
            p = scipy.stats.ttest_rel(np.array(df2.dff[np.where(df2.group=='pre')[0]]),
                                      np.array(df2.dff[np.where(df2.group=='post')[0]]))
            ttype = 'paired'
        else:
            # plot all P-wave trials together
            sns.barplot(data=df, x='group', y='dff', palette=['lightgray','darkgray'], 
                        errorbar='se', ax=ax)
            # unpaired t-test
            p = scipy.stats.ttest_ind(np.array(df.dff[np.where(df.group=='pre')[0]]),
                                      np.array(df.dff[np.where(df.group=='post')[0]]))
            ttype = 'unpaired'
        ax.set_ylabel(ylabel)
        ax.set_title(title + f'\nvalues averaged over {bar_win} s pre/post P-wave')
        
        # print stats
        print('')
        print(f'###   Pre-P-wave vs post-P-wave ({bar_win} s, state={istate}, {ttype} t-test)')
        print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}')
        print('')
    return pwave_trig


def spectralfield_highres(ppath, name, pre, post, istate=[1], theta=[1,10,100,1000,10000], pnorm=1,
                          psmooth=0, fmax=30, recalc_highres=False, nsr_seg=2, perc_overlap=0.75, 
                          ma_thr=20, ma_state=3, flatten_is=False, pzscore=False, pplot=True):
    """
    Calculate "spectral field" optimally relating the EEG spectrogram to the DF/F calcium 
    activity for one recording
    @Params
    ppath - base folder
    name - recording folder
    pre, post - time window (s) for spectral field calculation, relative to measured
                neural response
    istate - brain state(s) to include in analysis
    theta - list of floats, specifying candidate regularization parameters for linear
            regression model. Optimal parameter value is chosen for each recording
            to maximize average model performance
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    fmax - maximum frequency in spectral field
    recalc_highres - if True, recalculate high-res spectrogram from EEG using $nsr_seg and $perc_overlap params
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    pzscore - if True, z-score DF/F signal by its mean across the recording
    pplot - if True, plot spectral field
    @Returns
    k - coefficient matrix optimally predicting DF/F calcium activity from matrix of
        EEG spectrogram features (freqs x relative time bins)
    t - array of time points relative to predicted neural activity (t=0 s)
    f - array of SP frequencies
    """
    if not type(istate) == list:
        istate = [istate]
    if not type(theta) == list:
        theta = [theta]
    
    # load sampling rate
    sr = sleepy.get_snr(ppath, name)
    nbin = int(np.round(sr) * 2.5)
    
    # load and adjust brain state annotation
    M = sleepy.load_stateidx(ppath, name)[0]
    M = AS.adjust_brainstate(M, dt=2.5, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
    
    # load/calculate high-resolution EEG spectrogram
    SP, freq, t = AS.highres_spectrogram(ppath, name, nsr_seg=nsr_seg, perc_overlap=perc_overlap,
                                         recalc_highres=recalc_highres)[0:3]
    N = SP.shape[1]
    ifreq = np.where(freq <= fmax)[0]
    f = freq[ifreq]
    nfreq = len(ifreq)
    dt = t[1]-t[0]
    # get indices of time window to collect EEG
    ipre  = int(np.round(pre/dt))
    ipost = int(np.round(post/dt))
    # normalize/smooth spectrogram
    SP = AS.adjust_spectrogram(SP, pnorm=pnorm, psmooth=psmooth)
    
    # collect SP windows surrounding each time bin, reshape SP windows as 1D vectors and 
    # store in consecutive rows of feature matrix
    MX = build_featmx(SP[ifreq,:], ipre, ipost)

    # load DF/F calcium signal, downsample to SP time resolution
    ndown = int(nsr_seg*sr) - int(nsr_seg*sr*perc_overlap)
    ninit = int(np.round(t[0]/dt))
    dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dff']*100
    dffd = AS.downsample_vec(dff, ndown)
    dffd = dffd[ninit:]
    if pzscore:
        dffd = (dffd-dffd.mean()) / dffd.std()
    dffd = dffd[ipre:N-ipost]
    
    ibin = np.array([], dtype='int64')
    M,K = sleepy.load_stateidx(ppath, name)
    for s in istate:
        # get indices of all brain states in $istate
        seq = sleepy.get_sequences(np.where(M==s)[0])
        for p in seq:
            # convert to high-resolution indices
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

    # perform cross validation
    Etest, Etrain = cross_validation(MX, dffd, theta)
    print("CV results on training set:")
    print(Etrain)
    print("CV results on test set")
    print(Etest)

    # get optimal theta value, estimate spectral field
    imax = np.argmax(Etest)
    print("Recording %s; optimal theta: %2.f" % (name, theta[imax]))
    k = ridge_regression(MX, dffd, theta[imax])
    k = np.reshape(k, ((ipre + ipost), nfreq)).T
    t = np.arange(-ipre, ipost) * dt
    
    # plot spectral field as heatmap
    if pplot:
        plt.ion()
        plt.figure()
        # create dataframe for spectral field
        dfk = sleepy.nparray2df(k, f, t, 'coeff', 'freq', 'time')  
        dfk = dfk.pivot("freq", "time", "coeff") 
        ax=sns.heatmap(dfk, cbar=False, cmap="jet") 
        ax.invert_yaxis()        
        plt.ylabel('Freq (Hz)')
        plt.xlabel('Time (s)')
    return k, t, f


def spectralfield_highres_mice(ppath, recordings, pre, post, istate=[1], theta=[1,10,100,1000,10000],
                               pnorm=1, psmooth=0, fmax=30, recalc_highres=False, nsr_seg=2, perc_overlap=0.75, 
                               ma_thr=20, ma_state=3, flatten_is=False, pzscore=False, vm=[]):
    """
    Calculate average "spectral field" optimally relating the EEG spectrogram to the DF/F calcium 
    activity across multiple recordings
    @Params
    ppath - base folder
    name - recording folder
    pre, post - time window (s) for spectral field calculation, relative to measured
                neural response
    istate - brain state(s) to include in analysis
    theta - list of floats, specifying candidate regularization parameters for linear
            regression model. Optimal parameter value is chosen for each recording
            to maximize average model performance
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    fmax - maximum frequency in spectral field
    recalc_highres - if True, recalculate high-res spectrogram from EEG using $nsr_seg and $perc_overlap params
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    pzscore - if True, z-score DF/F signal by its mean across the recording
    vm - controls saturation for averaged spectral field
    @Returns
    SpecFields - matrix of mouse-averaged spectral fields (freq x time bins x mice)
    t - array of time points relative to predicted neural activity (t=0 s)
    f - array of SP frequencies
    """
    mice = []
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice.append(idf)
    
    # collect spectral fields from all recordings
    spec_fields = {k:[] for k in mice}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        k,t,f = spectralfield_highres(ppath, rec, pre=pre, post=post, istate=istate, 
                                      theta=theta, pnorm=pnorm, psmooth=psmooth, 
                                      fmax=fmax, recalc_highres=recalc_highres,
                                      nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                      ma_thr=ma_thr, ma_state=ma_state, 
                                      flatten_is=flatten_is, pzscore=pzscore, pplot=False)
        spec_fields[idf].append(k)
    
    # get averaged spectral field for each mouse
    SpecFields = np.zeros((len(f), len(t), len(mice)))
    i = 0
    for m in mice:
        SpecFields[:,:,i] = np.array(spec_fields[m]).mean(axis=0)
        i += 1
    # plot spectral field as heatmap
    plt.ion()
    plt.figure()
    ax = plt.gca()
    im = ax.pcolorfast(t, f, SpecFields.mean(axis=2), cmap='jet')
    if len(vm) == 2:
        im.set_clim(vm)
    plt.colorbar(im, ax=ax, pad=0.0)
    plt.xlabel('Time (s)')
    plt.ylabel('Freq. (Hz)')
    plt.show()
    return SpecFields, t, f


def state_freq(ppath, recordings, istate, plotMode='0', tstart=0, tend=-1,
               ma_thr=20, ma_state=3, flatten_is=False, wf_win=[0.5,0.5],
               p_iso=0, pcluster=0, clus_event='waves', noise_state=2, 
               exclude_noise=False, mouse_avg='mouse', avg_mode='all', pplot=True, 
               print_stats=True, ylim1=[], ylim2=[], return_mode='mx', handle_sr='NP'):
    """
    Plot average P-wave frequency and waveform in each state
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    plotMode - parameters for P-wave frequency bar plot
               '0' - error bar +/-SEM
               '1' - black dots for mice;  '2' - color-coded dots for mice
               '3' - black lines for mice; '4' - color-coded lines for mice
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    wf_win - time window (s) to collect LFP waveforms
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    noise_state - brain state to assign manually annotated regions of EEG/LFP noise
                  (if 0, do not analyze)
    exclude_noise - if False, ignore manually annotated LFP noise indices
                    if True, exclude time bins containing LFP noise from analysis
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    avg_mode - for each recording, P-wave freq is computed as the average across 
               all state bins ('all'), or from the averages of each state sequence ('each')
    pplot - if True, show plots
    print_stats - if True, show results of repeated measures ANOVA
    ylim1, ylim2 - set y axis limits for P-wave frequency bar plot and avg waveform plot
    @Returns
    mice - list of mouse names, corresponding to rows in freq_mx & waveform_mx
    xlabels - list of brain states, corresponding to columns/layers in freq_mx & waveform_mx
    freq_mx - array of mean P-wave frequencies (mice x brain states)
    waveform_mx - array of LFP waveforms (mice x time bins x brain states)
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # collect number of P-waves in each brainstate
    state_counts = {state:0 for state in istate}
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    
    # collect P-wave frequencies and waveform for each state
    freq_mouse = {s:{rec:[] for rec in recordings} for s in istate}
    waveform_mouse = {s:{rec:[] for rec in recordings} for s in istate}
    
    for rec in recordings:
        print('Getting P-waves for ' + rec + ' ...')
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = AS.get_snr_pwaves(ppath, rec, default='NP')
        #sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt=dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is, noise_state=noise_state)
        # load LFP and P-wave indices
        if exclude_noise:
            # load noisy LFP indices, make sure no P-waves are in these regions
            LFP, idx, noise_idx = load_pwaves(ppath, rec, return_lfp_noise=True)[0:3]
            idx = np.setdiff1d(idx, noise_idx)
        else:
            LFP, idx = load_pwaves(ppath, rec)[0:2]
        # isolate single/clustered P-waves
        if p_iso and pcluster:
            print('ERROR: cannot accept both p_iso and pcluster arguments')
            return
        elif p_iso:
            idx = get_p_iso(idx, sr, win=p_iso)
        elif pcluster:
            idx = get_pclusters(idx, sr, win=pcluster, return_event=clus_event)
        
        # define start and end points of analysis
        istart = int(np.round((1.0*tstart) / dt))
        if tend==-1:
            iend = len(M)
        else:
            iend = int(np.round((1.0*tend) / dt))
        iwin1, iwin2 = get_iwins(wf_win, sr, inclusive=False)
        
        # collect P-wave frequency and waveforms for each brain state
        for s in istate:
            sseq = sleepy.get_sequences(np.where(M==s)[0])
            
            # if no instances of brain state, collect NaNs instead
            if len(sseq[0]) == 0:
                freq_mouse[s][rec].append(np.nan)
                emp = np.empty((iwin1+iwin2,))
                emp[:] = np.nan
                waveform_mouse[s][rec].append([emp])
                continue
            try:
                sseq = [seq for seq in sseq if seq[-1] >= istart and seq[0] <= iend]
            except:
                pdb.set_trace()
            # get Intan indices for each state episode, exclude noise states if indicated
            state_idx = [np.arange(seq[0]*nbin, seq[-1]*nbin+nbin) for seq in sseq]
            if exclude_noise:
                state_idx = [np.setdiff1d(si, noise_idx) for si in state_idx]
                state_idx = [si for si in state_idx if si.size>0]
                
            # get P-waves in each state episode
            state_p_idx = [np.intersect1d(idx, si) for si in state_idx]
            
            # get P-wave freq for each state episode
            sfreq = [len(sp) / (len(si) / sr) * dt for sp,si in zip(state_p_idx, state_idx)]
            if mouse_avg != 'trial':
                if avg_mode == 'all':
                    # get avg P-wave freq over all state bins
                    sfreq = [len(np.concatenate(state_p_idx)) / (len(np.concatenate(state_idx)) / sr) * dt]
                elif avg_mode == 'each':
                    # get avg P-wave freq from averages of each state episode
                    sfreq = [np.nanmean(sfreq)]
                
            freq_mouse[s][rec] = sfreq
            
            # get all P-waveforms in state $s, collect average waveform
            state_waveforms = np.array([LFP[spi-iwin1:spi+iwin2] for spi in np.concatenate(state_p_idx) if spi>=iwin1 and spi+iwin2<len(LFP)])
            if len(state_waveforms) == 0:
                emp = np.empty((iwin1+iwin2,))
                emp[:] = np.nan
                waveform_mouse[s][rec] = [emp]
            else:
                waveform_mouse[s][rec] = [np.mean(state_waveforms, axis=0)]
                state_counts[s] += len(state_waveforms)  # update no. P-waves collected
        
    # print no. P-waves detected during each brain state
    print('')
    _ = [print(states[k], ':', state_counts[k], 'P-waves') for k in state_counts.keys()]
    print('')
    
    df = pd.DataFrame(columns=['mouse', 'recording', 'state', 'freq'])
    for s in istate:
        for rec in recordings:
            idf = rec.split('_')[0]
            ddict = {'mouse':idf, 'recording':rec, 'state':s, 'freq':freq_mouse[s][rec]}
            df = pd.concat([df, pd.DataFrame(ddict)], axis=0, ignore_index=True)
            
    if mouse_avg == 'mouse':
        df = df.groupby(['mouse','state']).mean().reset_index()
    
    if mouse_avg != 'trial':
        units = mice if mouse_avg=='mouse' else recordings
        # create matrices for P-wave frequencies (mice/recordings x state) 
        # and waveforms (mice/recordings x time bins x state)
        freq_mx = np.zeros((len(units), len(istate)))
        waveform_mx = np.zeros((len(units), iwin1+iwin2, len(istate)))
        for sidx, state in enumerate(istate):
            for uidx, unit in enumerate(units):
                if mouse_avg=='mouse':
                    f = [freq_mouse[state][rec][0] for rec in recordings if rec.split('_')[0] == unit]
                    f = np.nanmean(f)
                    w = [waveform_mouse[state][rec][0] for rec in recordings if rec.split('_')[0] == unit]
                    w = np.nanmean(w, axis=0)
                else:
                    f = freq_mouse[state][unit][0]
                    w = waveform_mouse[state][unit][0]
                freq_mx[uidx, sidx] = f
                waveform_mx[uidx, :, sidx] = w
        
        xlabels = [states[s] for s in istate]
    
        # plot P-wave frequency and avg waveform in each brain state
        if pplot:
            plot_state_freq(xlabels, units, freq_mx, waveform_mx, plotMode=plotMode, 
                            ylim2=ylim2, group_colors='states', group_labels=[], legend='all')
        
        # stats
        if print_stats and len(istate) > 2 and len(mice) > 1:
            res_anova = ping.rm_anova(data=df, subject=mouse_avg, dv='freq', within=['state'])
            ping.print_table(res_anova)
            if res_anova.loc[0,'p-unc'] < 0.05:
                res_tt = ping.pairwise_tests(data=df, subject=mouse_avg, dv='freq', 
                                             within=['state'], padjust='Holm')
                ping.print_table(res_tt)
            
        if return_mode == 'mx':
            return units, xlabels, freq_mx, waveform_mx
        elif return_mode == 'df':
            return df
    
    elif mouse_avg == 'trial':
        return df


def plot_state_freq(brainstates, mice, freq_mx_list, waveform_mx_list, plotMode='0', 
                    plotBrainstates=[], group_colors=[], group_labels=[], title='', 
                    legend='', ylim1=[], ylim2=[]):
    """
    Plot average P-wave frequency and waveforms across brain states, for 1-2 groups of mice
    @Params
    brainstates - list of brain states, corresponding to columns/layers in freq_mx & waveform_mx
    mice - list of mouse names, corresponding to rows in freq_mx & waveform_mx
    freq_mx_list - list with 1-2 arrays of mean P-wave frequencies (mice x brain states)
    waveform_mx_list - list with 1-2 arrays of LFP waveforms (mice x time bins x brain states)
    plotMode - parameters for P-wave frequency bar plot
               '0' - error bar +/-SEM
               '1' - black dots for mice;  '2' - color-coded dots for mice
               '3' - black lines for mice; '4' - color-coded lines for mice
               '5' - black lines between groups; '6' - color-coded lines between groups
    plotBrainstates - list of indices or brain state names, corresponding to elements
                       in $brainstates to include in plots
    group_colors - optional list of colors for each group
    group_labels - optional list of legend labels for each group
    title - optional plot title
    legend - info included in legend (''=no legend, 'all'=mouse names & group labels,
                                      'mice'=mouse names, 'groups'=group labels)
    ylim1, ylim2 - set y axis limits for P-wave frequency bar plot and avg waveform plot
    @Returns
    None
    """
    # clean data inputs
    if not isinstance(brainstates, list):
        brainstates = [brainstates]
    if not isinstance(mice[0], list):  # mice should be a list of lists
        mice = [mice]
    if not isinstance(freq_mx_list, list):
        freq_mx_list = [freq_mx_list]
    if not isinstance(waveform_mx_list, list):
        waveform_mx_list = [waveform_mx_list]
    if not isinstance(group_colors, list):
        group_colors = [group_colors]
    if not isinstance(group_labels, list):
        group_labels = [group_labels]
    # raise error if input lists are different lengths
    if not len(mice) == len(freq_mx_list) == len(waveform_mx_list):
        print('ERROR: Lists of mouse names, frequency matrices, and waveform matrices must all be the same length.')
        sys.exit()
    # raise error if not all frequency matrices contain no. of columns in $brainstates
    if not set([fm.shape[1] for fm in freq_mx_list]) == {len(brainstates)}:
        print('ERROR: Number of columns in all frequency matrices must be equal to the number of inputted brainstates.')
        sys.exit()
    
    # plot selected brainstates
        if all([type(pB) == int for pB in plotBrainstates]):
            bidx = [pB for pB in plotBrainstates if 0 <= pB <len(brainstates)]
        elif all([type(pB) == str for pB in plotBrainstates]):
            bidx = [i for i,B in enumerate(brainstates) if B in plotBrainstates]
            
        if len(bidx) > 0:
            brainstates = [brainstates[b] for b in bidx]
            for i in range(len(freq_mx_list)):
                freq_mx_list[i] = freq_mx_list[i][:, bidx]
                waveform_mx_list[i] = waveform_mx_list[i][:, :, bidx]
    
    STATE_COLORS=False
    # if there is only one mouse group and group_colors is set to 'states'
    if ('state' in group_colors or 'states' in group_colors) and len(mice) == 1:
        STATE_COLORS=True
        brainstate_names = [b.split(' ')[1] if b.startswith('pre-') or b.startswith('post') else b for b in brainstates]
        # plot each brainstate bar with the color specified in the mouse_colors txt file
        group_colors = [list(AS.colorcode_mice(brainstate_names).values())]
        if legend == 'all':
            legend='mice'
        else:
            legend = [l for l in legend if l!='groups']
        
    if len(group_colors) < len(mice):
        group_colors = AS.colorcode_mice('', return_colorlist=True)
    if len(group_labels) < len(mice):
        group_labels = [str(i+1) for i in range(len(mice))]
    
    if title=='':
        title='P-wave Frequency Across Brain States'
    
    # set colors for plotting
    mcs = {}
    for m in mice:
        mcs.update(AS.colorcode_mice(m))
    linecolor = ''
    markercolor = ''
    
    # plot avg P-wave frequencies by brain state
    fig = plt.figure()
    ax = plt.gca()
    ms_handles = []
    # print warning if mouse has no examples of event or brain state
    mouse_warnings = [] 
    
    t = np.arange(0, len(brainstates))        
    width=0.75
    width /= len(mice)
    # for each set of bars from a mouse group, position to left or right of the x ticks based on total number of groups
    tick_bar = np.linspace(-((width/2)*(len(mice)-1)), (width/2)*(len(mice)-1), len(mice))
    
    # no. of mice contributing data (aka, exclude NaN mice)
    nmice = [list(np.invert(np.isnan(fm)).sum(axis=0)) for fm in freq_mx_list]
    
    # store dataframe of x and y values to connect with lines
    line_coors = pd.DataFrame(index=list(mcs.keys()))
    for b in brainstates:
        line_coors[b] = [ [] for row in range(len(line_coors))]
    
    for (idx, mouse_names, freq_mx) in zip(np.arange(0, len(mice)), mice, freq_mx_list):
        # for each tick, the 1st bar is shifted left by 1/2 the width of the bar for every group of mice following
        x = t+tick_bar[idx]
        data = np.nanmean(freq_mx, axis=0)
        std = np.nanstd(freq_mx, axis=0)
        sem = [std[i] / np.sqrt(n) for i,n in enumerate(nmice[idx])]
        
        # plot errorbars
        if '0' in plotMode:
            ax.bar(x, data, width=width, yerr=sem, color=group_colors[idx], edgecolor='black', 
                   label=group_labels[idx], error_kw=dict(lw=3, capsize=0))
        else:
            ax.bar(x, data, width=width, color=group_colors[idx], edgecolor='black', label=group_labels[idx])
        for mrow, mname in enumerate(mouse_names):
            for bcol, brstate in enumerate(brainstates):
                X_ = x[bcol]
                Y_ = freq_mx[mrow, bcol]
                # plot individual mouse markers/lines
                if '1' in plotMode: markercolor = 'black'
                elif '2' in plotMode: markercolor = mcs[mname]
                if '3' in plotMode: linecolor = 'black'
                elif '4' in plotMode: linecolor = mcs[mname]
                if '1' in plotMode or '2' in plotMode:
                    mh = ax.plot(X_, Y_, color=markercolor, marker='o', ms=7, markeredgewidth=1, 
                                 markeredgecolor='black', label=mname, clip_on=False)[0]
                if '3' in plotMode or '4' in plotMode:
                    if bcol > 0:
                        xy = freq_mx[mrow, bcol]
                        ax.plot([x[bcol-1], x[bcol]], [freq_mx[mrow, bcol-1], xy], 
                                color=linecolor, linewidth=2, label=mname)
                    # add current x and y values to dataframe
                    line_coors.loc[mname, brstate].append((X_, Y_))
    
    if '5' in plotMode or '6' in plotMode:
        # check if the same mice are used in each frequency mx
        same_mice = True
        if len(set([len(m) for m in mice])) == 1:
            for midx in range(len(mice[0])):
                if len(set([m[midx] for m in mice])) != 1:
                    same_mice = False
                    break
        else:
            same_mice = False
        # plot lines between same mice in different groups
        if same_mice and len(mice)>1:
            for bcol in t:
                x_vals = [bcol+tb for tb in tick_bar]
                for mrow, mname in enumerate(mice[0]):
                    y_vals = [fmx[mrow,bcol] for fmx in freq_mx_list]
                    if '5' in plotMode: linecolor = 'black'
                    elif '6' in plotMode: linecolor = mcs[mname]
                    ax.plot(x_vals, y_vals, color=linecolor, linewidth=2, label=mname)
    ax.set_xticks(t)
    ax.set_xticklabels(brainstates, rotation=45, ha='right')
    ax.set_ylabel('P-waves/s')
    if len(ylim1) == 2:
        ax.set_ylim(ylim1)
    if 'all' in legend:
        AS.get_unique_labels(ax)
    elif 'groups' in legend:
        AS.legend_bars(ax)
    elif 'mice' in legend:
        if '2' in plotMode or '4' in plotMode or '6' in plotMode:
            AS.legend_mice(ax, list(chain.from_iterable(mice)))
    ax.set_title(title)

    # plot avg waveform for each brain state
    iwin = int(waveform_mx_list[0].shape[1]/2)
    t = np.arange(-iwin, iwin)

    subplot_map = {1:(1,1), 2:(1,2), 3:(1,3), 4:(2,2), 5:(2,3), 6:(2,3), 
                   7:(2,4), 8:(2,4), 9:(3,3), 10:(2,5), 11:(3,4), 12:(3,4)}
    
    # set up axes in a grid
    if len(mice) == 1:
        nrows = subplot_map[len(brainstates)][0]
        ncols = subplot_map[len(brainstates)][1]
    else:
        nrows = len(brainstates)
        ncols = len(mice)
    
    figsize = (5, 4+nrows)
    fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, constrained_layout=True)
    if len(brainstates) == 1 and len(mice) == 1:
        axs = np.array(axs)
    axs = axs.reshape(-1)
    ax_idx = 0
    
    # color-code waveforms by group or brain state
    if STATE_COLORS:
        group_colors = [[gc] for gc in group_colors[0]]
    else:
        group_colors = [group_colors]*len(brainstates)
    
    for br_idx, br_name in enumerate(brainstates):
        y = []
        # get mean waveform for each group and brain state
        for (group_idx, mouse_names, waveform_mx) in zip(np.arange(0, len(mice)), mice, waveform_mx_list):
            ax = axs[ax_idx]
            # get mean waveform and sem, divide by 1000 to convert Intan data from uV to m
            data = np.nanmean(waveform_mx[:,:,br_idx], axis=0) / 1000
            yerr = np.nanstd(waveform_mx[:,:,br_idx])  # +/- std
            yerr /= 1000
            
            # plot waveforms
            if all(np.isnan(data)):
                ax.plot(t, np.zeros((len(data))), color=group_colors[br_idx][group_idx], label=group_labels[group_idx])
            else:
                ax.plot(t, data, color=group_colors[br_idx][group_idx])
                ax.fill_between(t, data-yerr, data+yerr, color=group_colors[br_idx][group_idx], 
                                alpha=0.3, edgecolor=None, label=group_labels[group_idx])
            ax.set_title(br_name)
            if len(ylim2) ==2:
                ax.set_ylim((ylim2))
            
            # set axis labels on left and bottom of grid
            if ax_idx % ncols == 0: ax.set_ylabel('Amp. (mV)')
            if ax_idx >= len(axs)-ncols: ax.set_xlabel('Time (ms)')
            # set group labels on top of grid
            if ax_idx <= 1:
                if 'groups' in legend or 'all' in legend:
                    AS.legend_lines(ax, skip=list(chain.from_iterable(mice)))
            y.append(ax.get_ylim())
            ax_idx += 1
        
        # set ylims to be identical for each graph in a row
        max_ylim = (max([yy[0] for yy in y]), max([yy[1] for yy in y]))
        [axs[i].set_ylim(max_ylim) for i in np.arange(ax_idx - len(y), ax_idx)]

    plt.show()
    # print warnings for mice missing waveforms
    print('')
    for mw in mouse_warnings: print(mw)
    
    
def activity_transitions(ppath, recordings, transitions, pre, post, si_threshold, 
                         sj_threshold, mode='pwaves', tstart=0, tend=-1, ma_thr=20, 
                         ma_state=3, flatten_is=False, fmax=30, pnorm=1, psmooth=0, 
                         sf=0, vm=[], mouse_avg='mouse', base_int=10, base_start=0, 
                         base_end=-1, xlim=[], ylim=[], pzscore=False, pplot=True, 
                         print_stats=True):
    """
    Plots timecourse of signals during X-->Y brain state transitions (absolute time)
    @Params
    ppath - base folder
    recordings - list of recordings
    transitions - list of tuples specifying brain state transitions to analyze
                  e.g. [(4,1), (1,2)] --> (IS to REM) and (REM to wake) transitions
    pre, post - time before and after state transition (s)
    si_threshold, sj_threshold - lists containing minimum duration of each of the following brain states: 
                                 ['REM', 'Wake', 'NREM', 'transition', 'failed transition', 'microarousal']
                                  * si_threshold indicates min. duration for pre-transition states, and
                                    sj_threshold indicates min. duration for post-transition states
    mode - type of signal to plot
           'pwaves' - plot P-wave frequency
           'dff' - plot DF/F signal
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    fmax - maximum frequency in spectrogram
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    psmooth, sf - smoothing params for spectrograms/timecourse vector
    vm - controls spectrogram saturation
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    base_int - for timecourse stats, size of consecutive time bins (s) to compare
    base_start - no. of bins into timecourse to start "baseline" bin
    base_end - no. of bins into timecourse to end comparisons 
    xlim, ylim - set x and y axis limits for timecourse graph
    pzscore - if True, z-score signal by its mean across the recording
    pplot - if True, plot timecourse graph
    print_stats - if True, show results of repeating t-tests w/ Bonferroni correction
    @Returns
    mice - list of mouse names
    trans_pwave - timecourse dictionary (key=transitions, value=matrix of subject x time)
    trans_spe - SP dictionary (key=transitions, value=matrix of freq x time x subject)
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
            
    states = {1:'R', 2:'W', 3:'N', 4:'tN', 5:'ftN', 6:'MA'}
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    
    # create dictionaries to collect signal and SPs for each mouse
    trans_pwave = dict()
    trans_spe = dict()
    trans_pwave_trials = dict()
    
    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        # dict: transition type -> mouse -> signal
        trans_pwave[sid] = []
        trans_spe[sid] = []
        trans_pwave_trials[sid] = []
        
    for (si,sj) in transitions:
        print('')
        print(f'NOW COLLECTING INFORMATION FOR {states[si]}{states[sj]} TRANSITIONS ...' )
        print('')
        # collect timecourses for each set of transitions
        sid = states[si] + states[sj]
        pwave_mouse = {m:[] for m in mice}
        spe_mouse = {m:[] for m in mice}
        
        for rec in recordings:
            idf = re.split('_', rec)[0]
            print("Getting data for", rec, "...")
            
            # load sampling rate
            sr = AS.get_snr_pwaves(ppath, rec, default='NP')
            nbin = int(np.round(sr)*2.5)
            dt = (1.0/sr)*nbin
            
            # load and adjust brain state annotation
            M, _ = sleepy.load_stateidx(ppath, rec)
            M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                     flatten_is=flatten_is)
            
            if mode == 'pwaves':
                # load LFP and P-wave indices, and downsample P-waves
                LFP, idx = load_pwaves(ppath, rec)[0:2]
                d = downsample_pwaves(LFP, idx, sr=sr, nbin=nbin, rec_bins=len(M))
                p_idx_dn, p_idx_dn_exact, p_count_dn, p_freq_dn = d
                if pzscore:
                    p_freq_dn = (p_freq_dn-p_freq_dn.mean()) / p_freq_dn.std()
                else:
                    p_freq_dn *= dt  # freq in units of P-waves / s
            elif mode == 'dff':
                # load DF/F calcium signal
                dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd']
                if pzscore:
                    p_freq_dn = (dff-dff.mean()) / dff.std()
                else:
                    p_freq_dn = dff*100.0
            
            # load and normalize spectrogram
            P = so.loadmat(os.path.join(ppath, rec,   'sp_' + rec + '.mat'), squeeze_me=True)
            SP = P['SP']
            freq = P['freq']
            ifreq = np.where(freq <= fmax)[0]
            if pnorm:
                sp_mean = SP.mean(axis=1)
                SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            SP = SP[ifreq,:]
            
            # define start and end points of analysis
            istart = int(np.round((1.0*tstart) / dt))
            if tend == -1: iend = len(M)
            else: iend = int(np.round((1.0*tend) / dt))
            # define start and end points for collecting transition data
            ipre  = int(np.round(pre/dt))
            ipost = int(np.round(post/dt)) + 1
            
            # get sequences of pre-transition brain state
            seq = sleepy.get_sequences(np.where(M==si)[0])
            # if no instances of pre-transition state, continue to next recording
            if len(seq) <= 1:
                continue
        
            for s in seq:
                # last idx in pre-transition state si
                ti = s[-1]
                # check if next state is post-transition state sj; only then continue
                if ti < len(M)-1 and M[ti+1] == sj:
                    # go into future
                    p = ti+1
                    while p<len(M)-1 and M[p] == sj:
                        p += 1
                    p -= 1
                    sj_idx = list(range(ti+1, p+1))
                    # indices of state si = seq
                    # indices of state sj = sj_idx
                    
                    # if si and sj meet duration criteria, collect signal and SP
                    if ipre <= ti < len(M)-ipost and len(s)*dt >= si_threshold[si-1]:
                        if len(sj_idx)*dt >= sj_threshold[sj-1] and istart <= ti < iend:
                            # signal timecourse for transition
                            pwave_si = p_freq_dn[ti-ipre+1:ti+1]
                            pwave_sj = p_freq_dn[ti+1:ti+ipost+1]
                            pwave = np.concatenate((pwave_si, pwave_sj))
                            # SP for transition
                            spe_si = SP[ifreq,ti-ipre+1:ti+1]
                            spe_sj = SP[ifreq,ti+1:ti+ipost+1]
                            spe = np.concatenate((spe_si, spe_sj), axis=1)
                            pwave_mouse[idf].append(pwave)
                            spe_mouse[idf].append(spe)
                        
        # for mice with no examples of transition, fill in with NaN arrays
        for mouse in pwave_mouse:
            if pwave_mouse[mouse] == []:
                emp = np.empty((ipre+ipost, ))
                emp[:] = np.nan
                pwave_mouse[mouse].append(emp)
                emp2 = np.empty((len(ifreq), ipre+ipost))
                emp2[:] = np.nan
                spe_mouse[mouse].append(emp2)
        trans_pwave[sid] = pwave_mouse
        trans_spe[sid] = spe_mouse
    
    # create signal matrix for each transition type and mouse (trials x time)
    for tr in trans_pwave_trials:
        for mouse in trans_pwave[tr]:
            trans_pwave_trials[tr] += trans_pwave[tr][mouse] 
    # delete transitions not found in any recording
    for i, tr in enumerate(list(trans_pwave_trials.keys())):
        if len(trans_pwave_trials[tr]) == 0:
            del trans_pwave_trials[tr]
            del trans_pwave[tr]
            del trans_spe[tr]
            del transitions[i]
            print('')
            print('')
            print('')
            print(f'### NO INSTANCES OF {tr} TRANSITIONS WERE FOUND IN THIS DATASET ###')
            print('')
            print('')
            print('')
    
    # create trial matrix (trials x time)
    for tr in trans_pwave_trials:
        trans_pwave_trials[tr] = np.vstack( trans_pwave_trials[tr] )
    # create mouse-averaged matrix (mouse x time)
    for tr in trans_pwave: 
        for mouse in trans_pwave[tr]:
            trans_pwave[tr][mouse] = np.nanmean(np.array(trans_pwave[tr][mouse]), axis=0)
            trans_spe[tr][mouse] = np.nanmean(np.array(trans_spe[tr][mouse]), axis=0)
    for tr in trans_pwave:
        trans_pwave[tr] = np.array(list(trans_pwave[tr].values()))
        trans_spe[tr] = np.array(list(trans_spe[tr].values()))
            
    # graph timecourses for each transition
    tinit = -ipre*dt
    if pplot:
        # set plotting variables
        ntrans = len(trans_pwave)
        nmice = len(mice)
        nx = 1.0/ntrans
        dx = 0.2 * nx
        f = freq[ifreq]
        t = np.arange(-ipre*dt, ipost*dt-dt + dt/2, dt)
        
        i = 0
        plt.ion()
        if len(transitions) > 2:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure()
        for (si,sj) in transitions:
            tr = states[si] + states[sj]
            # plot signal
            ax = plt.axes([nx*i+dx, 0.15, nx-dx-dx/3.0, 0.3])
            if nmice == 1:
                plt.plot(t, trans_pwave[tr].mean(axis=0), color='black')
            else:
                # mean is linear
                tmp = trans_pwave[tr].mean(axis=0)
                # std is not linear
                sem = np.std(trans_pwave[tr],axis=0) / np.sqrt(trans_pwave[tr].shape[0])
                if sf > 0:
                    tmp = AS.smooth_data2(tmp, sf)
                    sem = AS.smooth_data2(sem, sf)
                plt.plot(np.linspace(-pre, post, len(tmp)), tmp, color='black')
                ax.fill_between(np.linspace(-pre, post, len(tmp)), tmp - sem, tmp + sem, 
                                color=(0, 0, 0), alpha=0.4, edgecolor=None)
            sleepy.box_off(ax)
            # set axis labels and limits
            plt.xlabel('Time (s)')
            if len(xlim) == 2:
                plt.xlim(xlim)
            else: plt.xlim([t[0], t[-1]])
            if i==0:
                if pzscore:
                    pz = ' (z-scored)'
                else:
                    pz = ''
                if mode == 'pwaves':
                    plt.ylabel('P-waves/s' + pz)
                elif mode == 'dff':
                    plt.ylabel('DF/F' + pz)
            if len(ylim) == 2:
                plt.ylim(ylim)
        
            # plot spectrogram
            if i==0:
                axes_cbar = plt.axes([nx * i + dx+dx*2, 0.55+0.25+0.03, nx - dx-dx/3.0, 0.1])
            ax = plt.axes([nx * i + dx, 0.55, nx - dx-dx/3.0, 0.25])
            plt.title(states[si] + ' $\\rightarrow$ ' + states[sj])
            # average SPs
            tr_sp_avg = trans_spe[tr].mean(axis=0)
            tr_sp_avg = AS.adjust_spectrogram(tr_sp_avg, pnorm=0, psmooth=psmooth)

            im = ax.pcolorfast(t, f, tr_sp_avg, cmap='jet')
            if len(vm) > 0:
                im.set_clim(vm)
            ax.set_xticks([0])
            ax.set_xticklabels([])
            # set axis labels
            if i==0:
                plt.ylabel('Freq. (Hz)')
            if i>0:
                ax.set_yticklabels([])
            sleepy.box_off(ax)
            if i==0:
                # colorbar for spectrogram
                cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, orientation='horizontal')
                cb.set_label('Rel. Power')
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position('top')
                axes_cbar.set_alpha(0.0)
                axes_cbar.spines["top"].set_visible(False)
                axes_cbar.spines["right"].set_visible(False)
                axes_cbar.spines["bottom"].set_visible(False)
                axes_cbar.spines["left"].set_visible(False)
                axes_cbar.axes.get_xaxis().set_visible(False)
                axes_cbar.axes.get_yaxis().set_visible(False)
            i += 1
        plt.show()

    # stats -  when does activity becomes significantly different from baseline?
    if print_stats:
        for tr in trans_pwave:
            if mouse_avg == 'mouse':
                trans = trans_pwave[tr]
            elif 'trial' in mouse_avg:
                trans = trans_pwave_trials[tr]
            print('STATISTICS FOR TRANSITION ' + tr)
            df = stats_timecourse(trans, -pre, post, sr=(1.0/dt), base_int=base_int, 
                                  baseline_start=int(base_start*sr), 
                                  baseline_end=int(base_end*sr) if base_end != -1 else -1, 
                                  print_stats=print_stats)
    if mouse_avg == 'mouse':
        return mice, trans_pwave, trans_spe
    elif 'trial' in mouse_avg:
        return mice, trans_pwave_trials, trans_spe


def stateseq(ppath, recordings, sequence, nstates, state_thres, sign=['>','>','>'], 
             mode='pwaves', tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=False,
             p_iso=0, pcluster=0, clus_event='waves', fmax=30, pnorm=1, psmooth=0, 
             vm=[], sf=0, mouse_avg='mouse', base_int=10, base_start=0, base_end=-1, 
             xlim=[], ylim=[], pzscore=False, pplot=True, print_stats=True):
    """
    Plot timecourse of signals during X-->Y-->Z brain state transitions (normalized time)
    @Params
    ppath - base folder
    recordings - list of recordings
    sequence - list of brain states in order of transition
                  e.g. [3,4,1,2] --> NREM-->IS-->REM-->Wake transitions
    nstates - no. of bins for each brain state
    state_thres - list of floats specifying min or max duration of each state
    sign - list specifying for each state whether the corresponding value in $state_thres 
            indicates the minimum ('>') or maximum ('<') duration
    mode - type of signal to plot
           'pwaves' - plot P-wave frequency
           'dff' - plot DF/F signal
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    fmax - maximum frequency in spectrogram
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    vm - controls spectrogram saturation
    sf - smooth activity timecourse along time dimension
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    base_int - for timecourse stats, no. of consecutive time-normalized bins to compare
    base_start - no. of bins into timecourse to start "baseline" bin
    base_end - no. of bins into timecourse to end comparisons 
    xlim, ylim - set x and y axis limits for timecourse graph
    pzscore - if True, z-score signal by its mean across the recording
    pplot - if True, plot timecourse graph
    print_stats - if True, show results of repeating t-tests w/ Bonferroni correction
    @Returns
    mice - list of mouse names
    mx_pwave - timecourse dictionary (key=transitions, value=matrix of subject x time)
    mx_spe - SP dictionary (key=transitions, value=matrix of freq x time x subject)
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    # get title of transition sequence
    transition_title = states[sequence[0]]
    for s in sequence[1:]:
        transition_title = transition_title + ' to ' + states[s]
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    
    # create dictionaries to collect signal and spectrograms for each mouse
    pwave_mouse = {m: [] for m in mice}
    spe_mouse = {m: [] for m in mice}
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        print("Getting data for", rec, "...")
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
               
        if mode == 'pwaves':
            # load LFP and P-wave indices
            LFP, idx = load_pwaves(ppath, rec)[0:2]
            # isolate single/clustered P-waves
            if p_iso and pcluster:
                print('ERROR: cannot accept both p_iso and pcluster arguments')
                return
            elif p_iso:
                idx = get_p_iso(idx, sr, win=p_iso)
            elif pcluster:
                idx = get_pclusters(idx, sr, win=pcluster, return_event=clus_event)
            # downsample P-waves
            d = downsample_pwaves(LFP, idx, sr=sr, nbin=nbin, rec_bins=len(M))
            p_idx_dn, p_idx_dn_exact, p_count_dn, p_freq_dn = d
            if pzscore:
                p_freq_dn = (p_freq_dn-p_freq_dn.mean()) / p_freq_dn.std()
            else:
                p_freq_dn *= dt  # freq in units of P-waves / s
        elif mode == 'dff':
            # load DF/F calcium signal
            dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd']
            if pzscore:
                p_freq_dn = (dff-dff.mean()) / dff.std()
            else:
                p_freq_dn = dff*100.0
        
        # load and normalize spectrogram
        P = so.loadmat(os.path.join(ppath, rec,   'sp_' + rec + '.mat'), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        ifreq = np.where(freq <= fmax)[0]
        if pnorm:
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        SP = SP[ifreq,:]
        
        # define start and end points of analysis
        istart = int(np.round((1.0*tstart) / dt))
        if tend == -1: iend = -1
        else: iend = int(np.round((1.0*tend) / dt))
        M = M[istart : iend]
        p_freq_dn = p_freq_dn[istart : iend]
        SP = SP[:, istart:iend]
        
        # get sequences of FIRST brain state specified in $sequences
        seqi = sleepy.get_sequences(np.where(M==sequence[0])[0])
        if len(seqi) <= 1:
            continue
        for init_seq in seqi:
            qual = []
            # check if initial state meets duration criteria
            if sign[0] == '>':
                Q = len(init_seq)*dt >= state_thres[0]
            elif sign[0] == '<':
                Q = len(init_seq)*dt <= state_thres[0]
            elif sign[0] == 'x':
                Q = True
            if Q:
                qual.append(init_seq)
                si = init_seq[-1]
                # for each consecutive state in $sequences
                for i,state in enumerate(sequence[1:]):
                    sj = si+1
                    while sj < len(M) and M[sj] == state:
                        sj += 1
                    next_seq = np.arange(si+1, sj)
                    # check if each consecutive state meets duration criteria
                    if sign[i+1] == '>':
                        Q2 = len(next_seq)*dt >= state_thres[i+1] and len(next_seq) > 0
                    elif sign[i+1] == '<':
                        Q2 = len(next_seq)*dt <= state_thres[i+1] and len(next_seq) > 0
                    elif sign[i+1] == 'x':
                        Q2 = len(next_seq) > 0
                    if Q2:
                        si = next_seq[-1]
                        qual.append(next_seq)
                    else:
                        break
                # if each state in sequence meets duration criteria, collect transition
                if len(qual) == len(sequence):
                    p_morph = []
                    sp_morph = []
                    # temporally normalize each state in no. of bins specified by $nstates
                    for i,q in enumerate(qual):
                        if q[-1] == len(p_freq_dn):
                            p_morph.append(AS.time_morph(p_freq_dn[q[0:-1]], nstates[i]))
                            sp_morph.append(AS.time_morph(SP[:, q[0:-1]], nstates[i]))
                        else:
                            p_morph.append(AS.time_morph(p_freq_dn[q], nstates[i]))
                            sp_morph.append(AS.time_morph(SP[:, q], nstates[i]))
                    pwave_mouse[idf].append(np.concatenate(p_morph))
                    spe_mouse[idf].append(np.concatenate(sp_morph, axis=1))
    
    # for mice with no examples of transition, fill in with NaN arrays
    if all([len(pwave_mouse[m]) for m in mice]) == 0:
        print('')
        print('')
        print('')
        print(f'###   NO INSTANCES OF {transition_title} FOUND IN THIS DATASET   ###')
        print('')
        print('')
        print('')
        return np.nan, np.nan, np.nan
        
    seq_pwave = {'trial':[]}
    seq_spe = {'trial':[]}
    for mouse in mice:
        # for mouse average, average for each transition tr and mouse over trials
        seq_pwave[mouse] = np.array(pwave_mouse[mouse]).mean(axis=0)
        seq_spe[mouse] = np.array(spe_mouse[mouse]).mean(axis=0)
        # for single trial mode, collect each transition
        [seq_pwave['trial'].append(trial) for trial in pwave_mouse[mouse]]
        [seq_spe['trial'].append(trial) for trial in spe_mouse[mouse]]
            
    # create matrix for signal (subject x time bins) and SPs (freq x time bins x subject)
    i = 0
    if mouse_avg == 'mouse':
        mx_pwave = np.zeros((len(mice), sum(nstates)))
        mx_spe = np.zeros((len(mice), len(ifreq), sum(nstates)))
        for mouse in mice:
            mx_pwave[i,:] = seq_pwave[mouse]
            mx_spe[i,:,:] = seq_spe[mouse]
            i += 1
    elif 'trial' in mouse_avg:
        mx_pwave = np.zeros((len(seq_pwave['trial']), sum(nstates)))
        mx_spe = np.zeros((len(seq_spe['trial']), len(ifreq), sum(nstates)))
        for pt, st in zip(seq_pwave['trial'], seq_spe['trial']):
            mx_pwave[i,:] = pt
            mx_spe[i,:,:] = st
            i += 1
    if sf:
        mx_pwave = AS.convolve_data(mx_pwave, sf, axis='x')
            
    # graph time-normalized avg signal and SP for each transition
    if pplot:
        x = np.arange(0, sum(nstates))
        cumx = np.concatenate(([0],np.cumsum(nstates)))
        
        plt.ion()
        plt.figure()
        # average and plot spectrogram
        ax_spe = plt.axes([0.1, 0.6, 0.75, 0.35])
        ax_col = plt.axes([0.9, 0.6, 0.03, 0.35])
        sp_avg = mx_spe.mean(axis=0)
        sp_avg = AS.adjust_spectrogram(sp_avg, pnorm=0, psmooth=psmooth)
        im = ax_spe.pcolorfast(x, freq[ifreq], sp_avg, cmap='jet')
        ax_spe.set_xticks(cumx)
        ax_spe.set_xticklabels([])
        sleepy.box_off(ax_spe)
        ax_spe.set_ylabel('Freq. (Hz)')
        ax_spe.set_title(transition_title.replace('to', ' $\\rightarrow$ '))
        if len(vm) > 0:
            im.set_clim(vm)
        # SP colorbar
        plt.colorbar(mappable=im, cax=ax_col, pad=0.0, fraction=1)
        if len(xlim) > 0:
            plt.xlim(xlim)
            
        # plot signal
        nmice = len(mice)
        ax_pwave = plt.axes([0.1, 0.1, 0.75, 0.35])
        ax_pwave.plot(x, mx_pwave.mean(axis=0), color='black')
        if nmice > 1:
            tmp = mx_pwave.mean(axis=0)
            sem = np.std(mx_pwave, axis=0) / np.sqrt(mx_pwave.shape[0])
            ax_pwave.fill_between(x, tmp - sem, tmp + sem, color=(0, 0, 0), 
                                  alpha=0.3, edgecolor=None)
        # set axis labels
        ax_pwave.set_xticks(cumx)
        ax_pwave.set_xticklabels([])
        plt.xlim([1, sum(nstates)])
        sleepy.box_off(ax_pwave)
        if pzscore:
            pz = ' (z-scored)'
        else:
            pz = ''
        if mode == 'pwaves':
            plt.ylabel('P-waves/s' + pz)
        elif mode == 'dff':
            plt.ylabel('DF/F' + pz)
        if len(ylim) > 0:
            plt.ylim(ylim)
        if len(xlim) > 0:
            plt.xlim(xlim)
        plt.show()
        
    # stats
    stats_timecourse(mx_pwave, pre=0, post=sum(nstates), sr=1, base_int=base_int, 
                     baseline_start=base_start, baseline_end=base_end, print_stats=print_stats)
    
    return mice, mx_pwave, mx_spe


def plot_activity_transitions(mx_list, mice=[], sem=True, plot_id=[], group_labels=[], 
                              xlim=[], xlabel='', ylabel='', title=''):
    """
    Plot timecourses of signals (absolute or normalized time) during brain state transitions
    @Params
    mx_list - list with arrays of signal values (mouse/trial x time bins) for each group
    mice - list of mouse names or integers, corresponding to rows of arrays in $mx_list
    sem - if True, plot timecourse mean +/- SEM for each mouse group
          if False, plot timecourse for each mouse in each group
    plot_id - optional list specifying a color (for mouse group) or linestyle 
              (for individual mice) to plot timecourses
    group_labels - optional list of legend labels for each group 
    xlim - if plotting absolute time transitions between 2 states:
              --> 2-element list if [-pre, post] seconds before/after transition
           if plotting time-normalized transitions across N states:
              --> N-element list of the no. of bins in each state
    xlabel - optional label for x axis
    ylabel - optional label for y axis
    title - optional plot title
    @Returns
    None
    """
    # clean data inputs
    if type(mx_list) != list:
        mx_list = [mx_list]
    if type(mice[0]) != list:
        mice = [mice]
    if type(group_labels) != list:
        group_labels = [group_labels]
    if type(plot_id) != list:
        plot_id = [plot_id]
    if len(group_labels) < len(mx_list):
        group_labels = ['']*len(mx_list)
    if len(xlim) == 0:
        t = np.arange(0, len(mx_list[0][0]))
        t = [round(x*2.5, 1) for x in t]
        xticks = t
    elif len(xlim) == 2:
        t = np.linspace(xlim[0], xlim[1], len(mx_list[0][0]))
        xticks = t
    elif len(xlim) > 2:
        t = np.arange(0, len(mx_list[0][0]))
        xticks = [sum(xlim[0:i+1]) for i in range(len(xlim)-1)]
    
    # default plotting params
    if len(plot_id) < len(mx_list):
        if sem:
            plot_id = ['gray', 'blue', 'darkblue', 'purple']
        else:
            plot_id = ['-', '--', ':', '-.']
    if len(mice) > 0:
        mcs = {}
        for m in mice: mcs.update(AS.colorcode_mice(m))
    else:
        mcs = AS.colorcode_mice('', return_colorlist=True)
        
    # create plot
    plt.ion()
    plt.figure()
    ax = plt.gca()
    
    for i, mx in enumerate(mx_list):
        mx_mean = np.nanmean(mx, axis=0)
        mx_sem = np.nanstd(mx, axis=0) / np.sqrt(mx.shape[0])
        
        # plot mouse groups
        if sem:
            ax.plot(t, mx_mean, color=plot_id[i], label=group_labels[i])
            ax.fill_between(t, mx_mean - mx_sem, mx_mean + mx_sem, color=plot_id[i], 
                            alpha=0.4, edgecolor=None)
        # plot individual mice
        else:
            if len(mice) > 0:
                ms_names = mice[i]
                for row in range(len(mx)):
                    ax.plot(t, mx[row, :], color=mcs[ms_names[row].split('_')[0]], 
                            linestyle=plot_id[i], label=ms_names[row])
            else:
                for row in range(len(mx)):
                    ax.plot(t, mx[row, :], color=mcs[row], linestyle=plot_id[i])
            # set legend for mouse group linestyles
            ax.plot(t[0], min(mx_mean), linestyle=plot_id[i], color='black', 
                    label=group_labels[i])
    # set axis labels and limits
    ax.set_xticks(xticks)
    if len(xlim) < 3:
        ax.set_xticklabels(xticks, rotation=45, ha='right')
    elif len(xlim) == 3:
        ax.set_xticklabels(['']*len(xticks))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # set plot legend
    if sem:
        ax.legend()
    else:
        if len(mice) > 0:
            AS.legend_mice(ax, list(chain.from_iterable(mice)))
            AS.legend_lines(ax, skip=list(chain.from_iterable(mice)), 
                            loc='upper center')
        else:
            AS.legend_lines(ax)
        ax.legend()
    plt.show()


def sleep_timecourse(ppath, recordings, istate, tbin, n, stats, plotMode='0', 
                     tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=False, 
                     p_iso=0, pcluster=0, clus_event='waves', pzscore=False, 
                     exclude_noise=False, mouse_avg='mouse', pplot=True, ylim=[]):
    """
    Plot sleep data for consecutive time bins over the course of sleep recordings
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    tbin - no. of seconds per time bin
    n - no. of time bins
    stats - type of data to plot for each time bin
            'perc' - mean percent time spent in brain state
            'freq' - mean frequency of brain state
            'dur' - mean duration of brain state episodes
            'time' - time (s) spent in brain state (for each time bin)
            'total time' - cumulative time (s) spent in brain state (from 1st time bin)
            'is prob' - percent of transition states followed by REM sleep
            'pwave freq' - mean P-wave frequency during brain state
            'pwave amp - mean P-wave amplitude during brain state
            'dff' - mean DF/F signal during brain state
            'emg twitch' - mean frequency of phasic muscle twitches during brain state
    plotMode - parameters for data bar plot
               '0' - error bar +/-SEM
               '1' - black dots for mice;  '2' - color-coded dots for mice
               '3' - black lines for mice; '4' - color-coded lines for mice
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    pzscore - if True, z-score signal by its mean across the recording
    exclude_noise - if True, exclude time bins containing LFP noise when calculating
                    P-wave frequency
    mouse_avg - method for data averaging; by 'mouse' or 'recording'
    pplot - if True, show plot
    ylim - set y axis limit for plot
    @Returns
    mice - list of mouse names, corresponding to rows of arrays in $TimeCourse
    TimeCourse - data dictionary (key=brain state, value=data matrix of mice x time bins)
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if type(istate) != list:
        istate = [istate]
    
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # determine data type(s) to load
    if stats in ['perc', 'freq', 'dur', 'time', 'total time', 'is prob']:
        MODE = 'STATE'
    elif stats in ['pwave freq', 'pwave amp']:
        MODE = 'PWAVES'
    elif stats in ['dff']:
        MODE = 'DFF'
    elif stats in ['emg twitch']:
        MODE = 'EMG'
    else:
        print('')
        print('ERROR: Invalid "stats" parameter')
        print('')
        sys.exit()
    if stats == 'is prob':
        istate = [1]
        flatten_is = False

    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    
    # create dictionary to store data
    DATA = {s:{} for s in istate}
    for s in istate:
        if mouse_avg == 'mouse':
            for m in mice:
                DATA[s][m] = []
        else:
            for rec in recordings:
                DATA[s][rec] = []
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                                 flatten_is=flatten_is)
        
        # load LFP and P-wave indices, and downsample P-waves
        if MODE == 'PWAVES':
            print('Getting P-waves for', rec, '...')
            
            # load LFP and P-wave indices
            if exclude_noise:
                LFP, idx, noise_idx = load_pwaves(ppath, rec, return_lfp_noise=True)[0:3]
            else:
                LFP, idx = load_pwaves(ppath, rec)[0:2]
            
            # isolate single or clustered P-waves
            if p_iso and pcluster:
                print('ERROR: cannot accept both p_iso and pcluster arguments')
                return
            elif p_iso:
                idx = get_p_iso(idx, sr, win=p_iso)
            elif pcluster:
                idx = get_pclusters(idx, sr, win=pcluster, return_event=clus_event)
            d = downsample_pwaves(LFP, idx, sr=sr, nbin=nbin, rec_bins=len(M))
            p_idx_dn, p_idx_dn_exact, p_count_dn, p_freq_dn = d
            if pzscore:
                p_freq_dn = (p_freq_dn-p_freq_dn.mean()) / p_freq_dn.std()
                LFP[idx] = (LFP[idx]-LFP[idx].mean()) / LFP[idx].std()
            else:
                p_freq_dn *= dt  # freq in units of P-waves / s
        # load DF/F calcium signal
        elif MODE == 'DFF':
            print('Getting DF/F signal for', rec, '...')
            dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd']
            if pzscore:
                dff = (dff-dff.mean()) / dff.std()
            else:
                dff *= 100.0
        elif MODE == 'EMG':
            print('Getting EMG activity for', rec, '...')
            twitch_idx = so.loadmat(os.path.join(ppath, rec, 'emg_twitches.mat'), 
                                    squeeze_me=True)['emg_twitches']
        else:
            print('Getting brain states for', rec, '...')
        
        # define start and end points of analysis
        if tend==-1:
            iend = len(M)
        else:
            iend = int(np.round((1.0*tend) / dt))
        istart = int(np.round((1.0*tstart) / dt))
        ibin = int(np.round(tbin / dt))
        if (iend-istart) < ibin:
            print('ERROR: Desired bin size exceeds total recording time')
            sys.exit()
        
        for s in istate:
            # collect data for each time bin
            BIN_DATA = []
            
            for i in range(n):
                # isolate each time bin 
                M_cut = M[istart+i*ibin:istart+(i+1)*ibin]
                sidx = np.where(M_cut==s)[0]
                sseq = sleepy.get_sequences(sidx)
                
                if MODE =='STATE':
                    if stats == 'perc':
                        # get % time spent in state
                        BIN_DATA.append( (len(sidx) / (1.0*len(M_cut)))*100 )
                    
                    elif stats == 'freq':
                        # get frequency of state
                        if sseq[0].size > 0:
                            BIN_DATA.append(len(sseq) * (3600. / (len(M_cut)*dt)))
                        else:
                            BIN_DATA.append(0.)
                    
                    elif stats == 'dur':
                        # get average duration of state
                        d = np.array([len(j)*dt for j in sseq])
                        if not all(d==0):
                            BIN_DATA.append(d.mean())
                        else:
                            # if no instances of state, store NaN
                            BIN_DATA.append(np.nan)
                    
                    elif 'time' in stats:
                        # get no. of seconds spent in state
                        ssec = len(sidx)*dt
                        if stats == 'time':
                            BIN_DATA.append(ssec)
                        elif stats == 'total time':
                            # get cumulative time (s) spent in state up to this point
                            if i==0:
                                BIN_DATA.append(ssec)
                            else:
                                BIN_DATA.append(ssec + BIN_DATA[i-1])

                    elif stats == 'is prob':
                        # get % of transition states followed by REM sleep
                        numS = len(sleepy.get_sequences(np.where(M_cut==4)[0]))
                        numF = len(sleepy.get_sequences(np.where(M_cut==5)[0]))
                        BIN_DATA.append( numS/(numS+numF)*100 )
                
                if MODE == 'PWAVES':
                    intan_sidx = np.concatenate([np.arange(seq[0]*nbin, seq[-1]*nbin+nbin) for seq in sseq])
                    if exclude_noise:
                        intan_sidx = np.setdiff1d(intan_sidx, noise_idx)
                    
                    # get indices of P-waves in state $s
                    state_p_idx = np.intersect1d(idx, intan_sidx)
                    
                    if stats == 'pwave freq':
                        # divide no. P-waves by time (s) spent in state to get P-waves/s
                        sfreq = len(state_p_idx) / (len(intan_sidx) / sr) * dt
                        BIN_DATA.append(sfreq)
                    
                    elif stats == 'pwave amp':
                        BIN_DATA.append(np.nan)
                    
                if MODE == 'DFF':
                    if stats == 'dff':
                        # get avg DF/F signal during state
                        dff_cut = dff[istart+i*ibin:istart+(i+1)*ibin]
                        BIN_DATA.append(np.nanmean(dff_cut[sidx]))
                        
                if MODE == 'EMG':
                    if stats == 'emg twitch':
                        # get avg frequency of muscle twitches during state
                        twidx = [tw for tw in twitch_idx if tw in sidx]
                        if len(sidx) > 0:
                            BIN_DATA.append( (len(twidx)/len(sidx))*100 )
                        else:
                            BIN_DATA.append( (len(twidx)/len(sidx))*100 )
    
            # store time-binned data for brain state
            if mouse_avg == 'mouse':
                DATA[s][idf].append(BIN_DATA)
            else:
                DATA[s][rec] = BIN_DATA
    
    # create data dictionary (key=brain state, value=matrix of subject x time bins)
    units = mice if mouse_avg=='mouse' else recordings
    TimeCourse = {s:np.zeros((len(units), n)) for s in istate}
    # get data for each recording / avg for each mouse
    for s in istate:
        for i,u in enumerate(units):
            if mouse_avg == 'mouse':
                TimeCourse[s][i,:] = np.nanmean(DATA[s][u], axis=0)
            else:
                TimeCourse[s][i,:] = DATA[s][u]
            
    # plot bar graph for each brain state
    if pplot:
        plot_sleep_timecourse(TimeCourse, units, tstart, tbin, stats, group_colors='states', 
                              stateMap=states, legend='mice', plotMode=plotMode) 
    
    return units, TimeCourse


def plot_sleep_timecourse(TimeCourse_list, mice, tstart, tbin, stats, plotMode='0',
                          compare_within=[], set_key_groups=[], group_colors=[], stateMap={},
                          group_labels=[], ylims=[], titles='auto', legend='', FIG=None, AX=None):
    """
    Plot comparison of time-binned sleep data for one brain state between multiple groups of mice, 
    or between different brain states within one group of mice
    @Params
    TimeCourse_list - list of 1+ data dictionaries (key=brain state, value=matrix of mouse x time bins)
    mice - list of mouse name lists, corresponding to array rows in each $TimeCourse_list dictionary
    tstart - no. of seconds into recordings when data was first collected
    tbin - no. of seconds per time bin/column of arrays in $TimeCourse_list
    stats - type of data collected for each time bin (see sleep_timecourse documentation for full list)
    plotMode - parameters for time-binned bar plot
               '0' - error bar +/-SEM
               '1' - black dots for mice;  '2' - color-coded dots for mice
               '3' - black lines for mice; '4' - color-coded lines for mice
    compare_within - param for comparing data within the same mouse group ($TimeCourse_list must contain only 1 dictionary)
                     if list of integers --> specifies brain states to include in a single plot
                       e.g. [1,2] plots data from REM sleep and wake
                     if empty list --> refer to $set_key_groups param
    set_key_groups - param for comparing data between different mouse group ($TimeCourse_list must contain multiple dictionaries)
                     if list of integers --> create subplot for each brain state included in list
                       e.g. [1,2] generates separate REM sleep and wake graphs, and plots data from each mouse group
                     if empty list --> automatically generate subplots for each brain state appearing in $TimeCourse_list
    group_colors - optional list of lists, specifying colors for each group
                    * no. of items in $group_colors = no. of subplots, no. of elements per item = no. of plotted groups
                    if 'state' - automatically color-code bars by brain state
    group_labels - optional list of legend labels for each group
                    if 'state' - automatically label bars by brain state
    ylims - optional list of tuples specifying y axis limits for each plot
    titles - optional list of titles for each plot
              * if 'auto', use params to automatically generate descriptive title
    legend - info included in legend (''=no legend, 'all'=mouse names & group labels,
                                      'mice'=mouse names, 'groups'=group labels)
    @Returns
    None
    """
    if not stateMap:
        states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    else:
        states = stateMap
    
    # clean data inputs
    if not isinstance(TimeCourse_list, list):
        TimeCourse_list = [TimeCourse_list]
    if not isinstance(mice[0], list):
        mice = [mice] # mice should be a list of lists
    if not isinstance(group_colors, list):
        group_colors = [group_colors]
    if not isinstance(group_labels, list):
        group_labels = [group_labels]

    # check for 1 mouse name list per data dictionary
    if not len(mice) == len(TimeCourse_list):
        print('ERROR: Lists of mouse names and TimeCourse dictionaries must be the same length.')
        sys.exit()
    # check for equal no. of time bins per data matrix
    mx_cols = [ [mx.shape[1] for mx in TC.values()] for TC in TimeCourse_list]
    n = set(chain.from_iterable(mx_cols))
    if len(n) > 1:
        print('ERROR: All TimeCourse matrices must have the same number of time bins.')
        sys.exit()
    else:
        n = min(n)
    
    # compare stats from same set of recordings between different brain states
    if len(compare_within) > 0 and len(TimeCourse_list) == 1:
        compare_within = [ [item] if type(item) not in [list, tuple] else item for item in compare_within ]
        # check for errant state codes
        if set(chain.from_iterable(compare_within)).issubset( set(TimeCourse_list[0].keys()) ):
            plot_keys = compare_within
            nplots = len(plot_keys)
        else:
            print('')
            print('ERROR: Compare within failed, one or more state keys entered were not included the TimeCourse. Plotting states individually ...')
            print('')
            compare_within = []
    
    # compare stats from different sets of recordings for each brain state
    if len(compare_within) == 0 or len(TimeCourse_list) > 1:
        # plot specified brain states
        if len(set_key_groups) > 0:
            nplots = len(set_key_groups)
            plot_keys = set_key_groups
        # automatically plot all brain states
        else:
            # get max number of keys/subplots from the largest TimeCourse dictionary
            nplots = max([len(TC.keys()) for TC in TimeCourse_list])
            plot_keys = []
            kidx = 0
            while kidx < nplots:
                kgroup = [ list(TC.keys())[kidx] if len(list(TC.keys())) > kidx else 'NONE' for TC in TimeCourse_list ]
                plot_keys.append(kgroup)
                kidx+=1

    # automatically color code bars by brain state
    if 'state' in group_colors or 'states' in group_colors:
        group_colors = []
        for pk in plot_keys:
            keynames = [states[k] if type(k) == int else 'NONE' for k in pk]
            keycolors = [AS.colorcode_mice([k])[k] if k != 'NONE' else k for k in keynames]
            group_colors.append(keycolors)
    # automatically label bars by brain state   
    if 'state' in group_labels or 'states' in group_labels:
        title_group = [group_labels[0] if 'state' not in group_labels[0] else '']
        group_labels = []
        for pk in plot_keys:
            group_labels.append( [states[k] if type(k) == int else 'NONE' for k in pk] )
    # format colors and labels
    if len(group_colors) == len(mice):
        group_colors = [group_colors]*nplots
    if len(group_labels) == len(mice):
        group_labels = [group_labels]*nplots
    if len(group_colors) < nplots:
        group_colors = [AS.colorcode_mice('', return_colorlist=True)[-len(mice) : ]]*nplots
    if len(group_labels) < nplots:
        group_labels = [[str(i+1) for i in range(len(mice))]]*nplots
    mcs = {}
    for m in mice:
        mcs.update(AS.colorcode_mice(m))
    
    # set x axis ticks, label time bins
    t = np.arange(0, n)
    dt = 2.5
    start = (tstart + tbin) / 3600.
    tlabel = [start + (i*tbin)/3600. for i in range(n)]
    # set y axis label
    if stats == 'perc': YLABEL = '% time spent'
    elif stats == 'freq': YLABEL = 'Freq (1/h)'
    elif stats == 'dur': YLABEL = 'Dur (s)'
    elif stats == 'time': YLABEL = 'Time spent (s)'
    elif stats == 'total time': YLABEL = 'Total time spent (s)'
    elif stats == 'is prob':
        YLABEL = '% transitions'
        titles = ['% transition states ending in REM sleep']
        # state key is 1, so set different title and color than REM
        if group_colors == [[['cyan']]]:
            group_colors = [[['forestgreen']]]
    elif stats == 'pwave freq': YLABEL = 'P-waves/s'
    elif stats == 'pwave_amp': YLABEL = 'P-wave amplitude'
    elif stats == 'LFP amp': YLABEL = 'LFP amplitude'
    elif stats == 'dff': YLABEL = 'DF/F'
    elif stats == 'emg twitch': YLABEL = 'EMG twitch freq'
    # set y axis limits & plot titles
    if 'auto' in titles:
        titles = ['auto']*nplots
    if len(titles) < nplots:
        titles = ['']*nplots
    if not (set([len(y) for y in ylims]) == {2} and len(ylims) == nplots):
        ylims = ['']*nplots
    
    mouse_warnings = [] 
    
    # create figure
    plt.ion()
    if not FIG or not AX:
        fig, axs = plt.subplots(figsize=(np.sqrt(n)*3, np.sqrt(nplots)*3), 
                                nrows=nplots, ncols=1, constrained_layout=True)
    else:
        fig = FIG; axs = AX
    if type(axs) != np.ndarray:
        axs = np.asarray((axs))
    
    # plot data according to key groupings
    for idx, pk in enumerate(plot_keys):
        ax = axs.reshape(-1)[idx]
        matrices = []
        mouse_names = []
        group_names = []
        bar_colors = []
        timecourse_idx = []
        brainstate_keys = []
        
        # organize plot data
        for ti, key in enumerate(pk):
            if key == 'NONE':
                continue
            if len(compare_within) > 0:
                TC_IDX = 0
            else:
                TC_IDX = ti
            # get data from appropriate TimeCourse dictionary
            matrices.append(TimeCourse_list[TC_IDX][key])
            mouse_names.append(mice[TC_IDX])
            timecourse_idx.append(TC_IDX)
            brainstate_keys.append(key)
            # get bar labels and colors
            group_names.append(group_labels[idx][ti])
            bar_colors.append(group_colors[idx][ti])
            
        # calculate bar width and position
        width = 0.75
        width = width/ len(matrices)
        tick_bar = np.linspace(-((width/2)*(len(matrices)-1)), 
                               (width/2)*(len(matrices)-1), len(matrices))
        # store data plot positions for drawing lines between groups 
        line_coors = pd.DataFrame(index=list(mcs.keys()), dtype = np.float64)
        for i in range(n):
            line_coors[i] = [[] for row in range(len(line_coors))]

        for (iidx, tcidx, k, mx, mnames, gname, bcolor) in zip(range(len(matrices)), 
                                                               timecourse_idx, 
                                                               brainstate_keys, 
                                                               matrices, mouse_names, 
                                                               group_names, bar_colors):
            x = t+tick_bar[iidx]
            data = np.nanmean(mx, axis=0)
            std = np.nanstd(mx, axis=0)
            sem = std / np.sqrt(len(mnames))
            # print warning if no instances of brain state found in a group
            if all(np.isnan(data)):
                mg = title_group[0] if len(compare_within) > 0 else gname
                mouse_warnings.append(f'WARNING: No mice in group {mg} had any instances of brainstate {states[k]}.')

            # plot bars
            if '0' in plotMode:
                ax.bar(x, data, width=width, yerr=sem, color=bcolor, edgecolor='black', 
                       label=gname, error_kw=dict(lw=3, capsize=0))
            else:
                ax.bar(x, data, width=width, color=bcolor, edgecolor='black', label=gname)  
            # plot individual mice  
            for mrow, mname in enumerate(mnames):
                for col in range(n):
                    point = TimeCourse_list[tcidx][k][mrow,col]
                    X_ = x[col]
                    Y_ = TimeCourse_list[tcidx][k][mrow,col]
                    if '1' in plotMode:
                        markercolor = 'black'
                    elif '2' in plotMode:
                        markercolor = mcs[mname]
                    if '3' in plotMode:
                        linecolor = 'black'
                    elif '4' in plotMode:
                        linecolor = mcs[mname]
                    if '1' in plotMode or '2' in plotMode:
                        if not np.isnan(Y_):
                            mh = ax.plot(X_, Y_, color=markercolor, marker='o', 
                                         ms=7, markeredgewidth=1, markeredgecolor='black', 
                                         label=mname, clip_on=False)[0]
                        else:
                             mh = ax.plot(X_, 0, color=markercolor, marker='X', 
                                          ms=7, markeredgewidth=2, markeredgecolor='black', 
                                          label=mname, clip_on=False)[0]
                    if '3' in plotMode or '4' in plotMode:
                        if iidx > 0:
                            xy = line_coors.loc[mname, col][iidx-1]
                            if not np.isnan(xy[1]):
                                ax.plot([xy[0], X_], [xy[1], Y_], color=linecolor, 
                                        linewidth=2, label=mname)
                            else:
                                if not np.isnan(Y_):
                                    ax.plot([xy[0], X_], [0, Y_], color=linecolor, 
                                            linewidth=2, linestyle='dashed', label=mname)
                                else:
                                    ax.plot([xy[0], X_], [0, 0], color=linecolor, 
                                            linewidth=2, linestyle='dotted', label=mname)
                        # add x and y position of plot point to dataframe
                        line_coors.loc[mname, col].append((X_, Y_))
        # set axis ticks & labels
        ax.set_xticks(t)
        ax.set_xticklabels(tlabel)
        ax.set_ylabel(YLABEL)
        if idx == nplots-1:
            ax.set_xlabel('Recording hour')
        # set y axis limits
        if ylims[idx] != '':
            ax.set_ylim(ylims[idx])
        # set plot titles
        if titles[idx] == 'auto':
            if len(compare_within) == 0:
                ax_title = AS.create_auto_title(brainstate_keys, group_names)
            else:
                ax_title = AS.create_auto_title(brainstate_keys, title_group)
            ax.set_title(ax_title)
        elif titles[idx] != '':
            ax.set_title(titles[idx])     
        # set legend
        if idx == 0:
            if 'group' in legend or 'all' in legend:
                AS.legend_bars(ax, loc='upper right')
            if 'mice' in legend or 'all' in legend:
                AS.legend_mice(ax, list(chain.from_iterable(mouse_names)), 
                               symbol='o', loc='upper center')
        sleepy.box_off(ax)
    if not FIG and not AX:
        plt.show()
    
    # print warnings for no brain state instances
    mouse_warnings = list(dict.fromkeys(mouse_warnings).keys())
    print('')
    for mw in mouse_warnings:
        print(mw)


def avg_waveform(ppath, recordings, istate, win=[0.5,0.5], mode='pwaves', tstart=0, 
                 tend=-1, ma_thr=20, ma_state=3, flatten_is=False, mouse_avg='mouse', 
                 exclude_noise=False, plaser=True, post_stim=0.1, lsr_iso=0, pload=False, 
                 psave=False, p_iso=0, pcluster=0, clus_event='waves', wform_std=True, ci='sem',
                 ylim=[], signal_type='LFP', dn=1):
    """
    Calculate average LFP or EMG signal surrounding P-waves and laser pulses in any brain state
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    win - time window (s) to collect data relative to given events
    mode - event-triggered signal(s) to plot
           'pwaves' - plot signal surrounding spontaneous P-waves
                       * if plaser=True, also plot LFP of laser-triggered P-waves
           'lsr' - plot signal surrounding successful and failed laser pulses
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    exclude_noise - if True, "noisy" waveforms collected as NaNs
    plaser - if True, plot LFP surrounding laser-triggered P-waves (if $mode='pwaves') or
             laser pulses (if $mode='lsr')
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    lsr_iso - if > 0, do not analyze laser pulses that are preceded within $lsr_iso s by 1+ P-waves
    pload - optional string specifying a filename to load the data (if False, data is collected from raw recording folders)
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    wform_std - if True, plot average signal +/- st. deviation; if False, plot LFP mean only
    ylim - set y axis limits for graphs
    signal_type - 'LFP' or 'EMG'
    dn - downsample signal by X bins across time axis
    @Returns
    df - dataframe of trial/recording/mouse ID, state, event type, and signal amplitude for each time point
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    
    states = {'total':'total', 1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # load LFP data
    if pload:
        data = AS.load_surround_files(ppath, pload=pload, istate=istate, plaser=plaser, 
                                      null=False, signal_type=signal_type)
        if len(data) > 0:
            if plaser:
                lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data
            elif not plaser:
                p_signal, null_signal, data_shape = data
        else:
            pload = False
    # collect LFP data
    if not pload:
        if plaser:
            data = get_lsr_surround(ppath, recordings, istate=istate, win=win, signal_type=signal_type, tstart=tstart, 
                                    tend=tend, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is, exclude_noise=exclude_noise,
                                    null=False, post_stim=post_stim, lsr_iso=lsr_iso, psave=psave,
                                    p_iso=p_iso, pcluster=pcluster, clus_event='waves')
            lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data
        elif not plaser:
            data = get_surround(ppath, recordings, istate=istate, win=win, signal_type=signal_type, tstart=tstart, 
                                tend=tend, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is, exclude_noise=exclude_noise,
                                null=False, psave=psave, p_iso=p_iso, pcluster=pcluster, clus_event=clus_event)
            p_signal, null_signal, data_shape = data
    
    df = pd.DataFrame(columns=['id','state','event','time','amp'])
    
    if plaser:
        for s in istate:
            # create single matrix (trials x time) of all trials
            if mouse_avg == 'trial':
                lsr_p_wf_mx, lsr_ids = mx2d(lsr_pwaves[s], d1_size = data_shape[0])
                spon_p_wf_mx, spon_ids = mx2d(spon_pwaves[s], d1_size = data_shape[0])
                success_lsr_wf_mx, success_ids = mx2d(success_lsr[s], d1_size = data_shape[0])
                fail_lsr_wf_mx, fail_ids = mx2d(fail_lsr[s], d1_size = data_shape[0])
            
            # create dictionary of matrices (trials x time) for each recording or mouse
            elif mouse_avg == 'recording' or mouse_avg == 'mouse':
                lsr_p_wf_avg = mx2d_dict(lsr_pwaves[s], mouse_avg, d1_size = data_shape[0])
                spon_p_wf_avg = mx2d_dict(spon_pwaves[s], mouse_avg, d1_size = data_shape[0])
                success_lsr_wf_avg = mx2d_dict(success_lsr[s], mouse_avg, d1_size = data_shape[0])
                fail_lsr_wf_avg = mx2d_dict(fail_lsr[s], mouse_avg, d1_size = data_shape[0])
                # get recording/mouse names
                lsr_ids = list(lsr_p_wf_avg.keys())
                spon_ids = list(spon_p_wf_avg.keys())
                success_ids = list(success_lsr_wf_avg.keys())
                fail_ids = list(fail_lsr_wf_avg.keys())
                
                # get single matrix (recording/mouse averages x time)
                lsr_p_wf_mx = np.array(([np.nanmean(lsr_p_wf_avg[k], axis=0) for k in lsr_ids]))
                spon_p_wf_mx = np.array(([np.nanmean(spon_p_wf_avg[k], axis=0) for k in spon_ids]))
                success_lsr_wf_mx = np.array(([np.nanmean(success_lsr_wf_avg[k], axis=0) for k in success_ids]))
                fail_lsr_wf_mx = np.array(([np.nanmean(fail_lsr_wf_avg[k], axis=0) for k in fail_ids]))
            else:
                raise Exception(f'ERROR : "{mouse_avg}" is not a valid averaging method.')
            # downsample signal
            if dn > 1:
                lsr_p_wf_mx = AS.downsample_mx(lsr_p_wf_mx, int(dn), axis='x')
                spon_p_wf_mx = AS.downsample_mx(spon_p_wf_mx, int(dn), axis='x')
                success_lsr_wf_mx = AS.downsample_mx(success_lsr_wf_mx, int(dn), axis='x')
                fail_lsr_wf_mx = AS.downsample_mx(fail_lsr_wf_mx, int(dn), axis='x')
                
            if mode == 'pwaves':
                x = np.linspace(-np.abs(win[0]), win[1], lsr_p_wf_mx.shape[1])
                # create dataframe
                ddf1 = pd.DataFrame({'id':np.repeat(lsr_ids, len(x)),
                                     'state':states[s],
                                     'event':'laser-triggered p-wave',
                                     'time':np.tile(x, len(lsr_ids)),
                                     'amp':lsr_p_wf_mx.flatten()})
                ddf2 = pd.DataFrame({'id':np.repeat(spon_ids, len(x)),
                                     'state':states[s],
                                     'event':'spontaneous p-wave',
                                     'time':np.tile(x, len(spon_ids)),
                                     'amp':spon_p_wf_mx.flatten()})
                df = pd.concat([df,ddf1,ddf2], axis=0, ignore_index=True)
                
                # mx1 for laser-triggered P-waves, mx2 for spontaneous P-waves
                mx1_data = np.nanmean(lsr_p_wf_mx, axis=0) #/ 1000  # convert uV to mV
                mx1_sem = np.nanstd(lsr_p_wf_mx, axis=0) #/ 1000 # +/- std
                if ci == 'sem':
                    mx1_sem /= np.sqrt(lsr_p_wf_mx.shape[0])
                title1 = 'Laser-Triggered P-waves'
                color1 = 'blue'
                mx2_data = np.nanmean(spon_p_wf_mx, axis=0) #/ 1000
                mx2_sem = np.nanstd(spon_p_wf_mx, axis=0) #/ 1000  # +/- std
                if ci == 'sem':
                    mx2_sem /= np.sqrt(spon_p_wf_mx.shape[0])
                title2 = 'Spontaneous P-waves'
                color2 = 'green'
            # mx1 for successful lsr, mx2 for failed lsr
            if mode == 'lsr':
                x = np.linspace(-np.abs(win[0]), win[1], success_lsr_wf_mx.shape[1])
                # create dataframe
                ddf1 = pd.DataFrame({'id':np.repeat(success_ids, len(x)),
                                     'state':states[s],
                                     'event':'successful laser',
                                     'time':np.tile(x, len(success_ids)),
                                     'amp':success_lsr_wf_mx.flatten()})
                ddf2 = pd.DataFrame({'id':np.repeat(fail_ids, len(x)),
                                     'state':states[s],
                                     'event':'failed laser',
                                     'time':np.tile(x, len(fail_ids)),
                                     'amp':fail_lsr_wf_mx.flatten()})
                df = pd.concat([df,ddf1,ddf2], axis=0, ignore_index=True)
                mx1_data = np.nanmean(success_lsr_wf_mx, axis=0) #/ 1000
                mx1_sem = np.nanstd(success_lsr_wf_mx, axis=0) #/ 1000  # +/- std
                if ci == 'sem':
                    mx1_sem /= np.sqrt(success_lsr_wf_mx.shape[0])
                title1 = 'Successful Laser Pulses'
                color1 = 'blue'
                mx2_data = np.nanmean(fail_lsr_wf_mx, axis=0) #/ 1000
                mx2_sem = np.nanstd(fail_lsr_wf_mx, axis=0) #/ 1000 # +/- std
                if ci == 'sem':
                    mx2_sem /= np.sqrt(fail_lsr_wf_mx.shape[0])
                title2 = 'Failed Laser Pulses'
                color2 = 'red'
            
            # plot waveform1 vs waveform2 for each state
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
            fig.suptitle(f'State={s}, averaged by {mouse_avg}')
            ax1.plot(x, mx1_data, color=color1, linewidth=3)
            if wform_std:
                ax1.fill_between(x, mx1_data-mx1_sem, mx1_data+mx1_sem, color=color1, alpha=0.3)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel(signal_type + ' Amp. mV')
            ax1.set_title(title1)
            ax2.plot(x, mx2_data, color=color2, linewidth=3)
            if wform_std:
                ax2.fill_between(x, mx2_data-mx2_sem, mx2_data+mx2_sem, color=color2, alpha=0.3)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel(signal_type + ' Amp. mV')
            ax2.set_title(title2)
            y = [ min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1]) ]
            if len(ylim)==2:
                ax1.set_ylim(ylim); ax2.set_ylim(ylim)
            else:
                ax1.set_ylim(y); ax2.set_ylim(y)
            #plt.show()
            
    if not plaser:
        for s in p_signal.keys():
            # create single matrix (trials x time) of all trials
            if mouse_avg == 'trial':
                wf_mx, wf_ids = mx2d(p_signal[s], d1_size = data_shape[0])
            # create dictionary of matrices (trials x time) for each recording or mouse
            elif mouse_avg == 'recording' or mouse_avg == 'mouse':
                wf_avg = mx2d_dict(p_signal[s], mouse_avg, d1_size = data_shape[0])
                wf_ids = list(wf_avg.keys())
                # get single matrix (recording/mouse averages x time)
                wf_mx = np.array(([np.nanmean(wf_avg[k], axis=0) for k in wf_ids]))
            else:
                raise Exception(f'ERROR : "{mouse_avg}" is not a valid averaging method.')
                
            # downsample signal
            if dn > 1:
                wf_mx = AS.downsample_mx(wf_mx, int(dn), axis='x')
            
            mx_data = np.nanmean(wf_mx, axis=0) #/ 1000  # convert uV to mV
            mx_sem = np.nanstd(wf_mx, axis=0) #/ 1000  # +/- std
            if ci == 'sem':
                mx_sem /= np.sqrt(wf_mx.shape[0])
            
            x = np.linspace(-np.abs(win[0]), win[1], len(mx_data))
            # create dataframe
            ddf = pd.DataFrame({'id':np.repeat(wf_ids, len(x)),
                                'state':states[s],
                                'event':'p-wave',
                                'time':np.tile(x, len(wf_ids)),
                                'amp':wf_mx.flatten()})
            df = pd.concat([df,ddf], axis=0, ignore_index=True)
            
            # plot avg waveform for each brain state
            fig = plt.figure()
            ax = plt.gca()
            ax.plot(x, mx_data, color='black', linewidth=3)
            if wform_std:
                ax.fill_between(x, mx_data-mx_sem, mx_data+mx_sem, color='black', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(signal_type + ' mV')
            #ax.set_title(f'State={s}, averaged by {mouse_avg}')
            if len(ylim) == 2:
                ax.set_ylim(ylim)
            #plt.show()
    return df


def avg_SP(ppath, recordings, istate, win=[5,5], mode='pwaves', recalc_highres=False, 
           tstart=0, tend=-1, pnorm=1, pcalc=0, psmooth=[1,1], vm=[(0,5),(0,5)],
           nsr_seg=2, perc_overlap=0.95, fmax=30, ma_thr=20, ma_state=3, flatten_is=False,
           p_iso=0, pcluster=0, clus_event='waves', mouse_avg='mouse', plaser=False, 
           post_stim=0.1, null=False, null_win=[0.5,0.5], null_match='spon', lsr_iso=0, pload=False, psave=False):
    """
    Calculate average spectrogram(s) surrounding a given event
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    win - time window (s) to collect data relative to given events
    mode - event-triggered spectrograms to plot
           'pwaves' - plot SP surrounding spontaneous P-waves (SP2)
                       * if plaser=True, SP1 is the spectrogram surrounding laser-triggered P-waves
                       * if null=True, SP1 is the spectrogram surrounding random control points
           'lsr' - plot SPs surrounding successful (SP1) and failed (SP2) laser pulses
           'null' - plot SPs surrounding random control points (SP1)
            * if plaser=True, SP2 is the spectrogram surrounding failed laser pulses
            * if plaser=False, SP2 is the spectrogram surrounding spontaneous P-waves
    recalc_highres - if True, recalculate high-resolution spectrogram from EEG, 
                      using $nsr_seg and $perc_overlap params
    tstart, tend - time (s) into recording to start and stop collecting data
    pnorm - method for spectrogram normalization (0=no normalization
                                                  1=normalize SP by recording
                                                  2=normalize SP by collected time window)
    pcalc - method for spectrogram calculation (0=load/calculate full SP and find event by index,
                                                1=calculate SP only for the time window surrounding event)
    psmooth - 2-element list specifying methods for smoothing [SP1, SP2]
               * a $psmooth element of 1 number specifies convolution along X axis
               * a $psmooth element of 2 numbers defines a box filter for smoothing
    vm - 2-element list controlling saturation for [SP1, SP2]
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    fmax - maximum frequency plotted in spectrogram
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    plaser - if True, plot SPs surrounding laser-triggered P-waves (if $mode='pwaves') or
             laser pulses (if $mode='lsr')
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    null - if True, plot SPs surrounding randomized control points in $istate
    null_win - if > 0, qualifying "null" points must be free of P-waves and laser pulses in surrounding $null_win interval (s)
               if = 0, "null" points are randomly selected from all state indices
    null_match - the no. of random control points is matched with the no. of some other event type
                 'spon'    - # control points equals the # of spontaneous P-waves
                 'lsr'     - # control points equals the # of laser-triggered P-waves
                 'success' - # control points equals the # of successful laser pulses
                 'fail'    - # control points equals the # of failed laser pulses
                 'all lsr' - # control points equals the # of total laser pulses
    lsr_iso - if > 0, do not analyze laser pulses that are preceded within $lsr_iso s by 1+ P-waves
    pload - optional string specifying a filename to load the data (if False, data is collected from raw recording folders)
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    @Returns
    None
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    if mode == 'null':
        null=True
    if vm == []:
        vm = [[],[]]
    if pcalc == 0:
        if pnorm == 0:
            signal_type = 'SP'  # pcalc=0 & pnorm=0 --> load SP from raw highres_SP, no normalization
            pnorm = False
        elif pnorm == 1:
            signal_type = 'SP_NORM'  # pcalc=0 & pnorm=1 --> load SP from normalized highres_SP
            pnorm = False
        elif pnorm == 2:
            signal_type = 'SP'  # pcalc=0 & pnorm=2 --> load SP from raw highres_SP, normalize within time window
            pnorm = True
    elif pcalc == 1:
        if pnorm == 0:
            signal_type = 'SP_CALC'  # pcalc=1 & pnorm=0 --> calculate SP from EEG, do not normalize
            pnorm = False
        elif pnorm == 1:
            signal_type = 'SP_CALC_NORM'  # pcalc=1 & pnorm=0 --> calculate SP from EEG, normalize by recording
            pnorm = False
        elif pnorm == 2:
            signal_type = 'SP_CALC'  # pcalc=1 & pnorm=2 --> calculate SP from EEG, normalize within time window
            pnorm = True
    
    states = {'total':'total', 1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
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
    # collect SPs
    if not pload:
        if plaser:
            data = get_lsr_surround(ppath, recordings, istate=istate, win=win, signal_type=signal_type, recalc_highres=recalc_highres,
                                    tstart=tstart, tend=tend, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is,
                                    nsr_seg=nsr_seg, perc_overlap=perc_overlap, null=null, null_win=null_win, null_match=null_match,
                                    post_stim=post_stim, lsr_iso=lsr_iso, psave=psave, p_iso=p_iso, pcluster=pcluster, clus_event=clus_event)
            lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data
        elif not plaser:
            p_signal, null_pts, data_shape = get_surround(ppath, recordings, istate=istate, win=win, signal_type=signal_type, recalc_highres=recalc_highres,
                                                             tstart=tstart, tend=tend, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is,
                                                             nsr_seg=nsr_seg, perc_overlap=perc_overlap, null=null, null_win=null_win, 
                                                             psave=psave, p_iso=p_iso, pcluster=pcluster, clus_event=clus_event)
    ifreq = np.arange(0, fmax*2+1)  # frequency idxs
    freq = np.arange(0, fmax+0.5, 0.5)  # frequencies
    x = np.linspace(-np.abs(win[0]), win[1], data_shape[1])
    
    if plaser:
        # for each brainstate, create a data matrix of freq x time bins x subject
        for s in istate:
            # mx1 for laser-triggered P-waves, mx2 for spontaneous P-waves
            if mode == 'pwaves':
                mx1, labels = mx3d(lsr_pwaves[s], mouse_avg, data_shape)
                mx2, labels = mx3d(spon_pwaves[s], mouse_avg, data_shape)
                title1 = 'Laser-triggered P-waves'
                title2 = 'Spontaneous P-waves'
            # mx1 for successful lsr, mx2 for failed lsr
            elif mode == 'lsr':
                mx1, labels = mx3d(success_lsr[s], mouse_avg)
                mx2, labels = mx3d(fail_lsr[s], mouse_avg)
                title1 = 'Successful laser pulses'
                title2 = 'Failed laser pulses'
            # mx1 for randomized control points, mx2 for failed lsr
            elif mode == 'null':
                mx1, labels = mx3d(null_pts[s], mouse_avg)
                mx2, labels = mx3d(fail_lsr[s], mouse_avg)
                title1 = 'Null points'
                title2 = 'Failed laser pulses'
            
            # average, normalize and smooth SPs
            mx1_plot = AS.adjust_spectrogram(np.nanmean(mx1, axis=2)[ifreq, :], pnorm, psmooth[0])
            mx2_plot = AS.adjust_spectrogram(np.nanmean(mx2, axis=2)[ifreq, :], pnorm, psmooth[1])
            
            # plot SP1 and SP2
            fig, (lax, sax) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
            fig.suptitle('SPs averaged by ' + mouse_avg)
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
            p_mx, labels = mx3d(p_signal[s], mouse_avg, data_shape)
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
                n_mx, labels = mx3d(null_pts[s], mouse_avg)
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
                fig = plt.figure(figsize=(4,5))
                ax = plt.gca()
                im = ax.pcolorfast(x, freq, p_mx_plot, cmap='jet')
                if len(vm[0]) == 2:
                    im.set_clim(vm[0])
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Freq (Hz)')
                ax.set_title(f'SP surrounding P-waves ({states[s]})')
                plt.colorbar(im, ax=ax, pad=0.0)
            #plt.show()
            #fig.suptitle('SPs averaged by ' + mouse_avg)


def avg_band_power(ppath, recordings, istate, bands, band_labels=[], band_colors=[], win=[5,5], 
                   mode='pwaves', recalc_highres=False, tstart=0, tend=-1, pnorm=1, pcalc=0, 
                   psmooth=0, sf=0, nsr_seg=2, perc_overlap=0.95, fmax=0, ma_thr=20, ma_state=3, 
                   flatten_is=False, mouse_avg='mouse', p_iso=0, pcluster=0, clus_event='waves', 
                   plaser=False, post_stim=0.1, null=False, null_win=[0.5,0.5], null_match='spon', 
                   lsr_iso=0, pload=False, psave=False, ylim=[]):
    """
    Calculate timecourses of frequency band power surrounding events, using collected spectrograms
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    bands - list of tuples with min and max frequencies in each power band
            e.g. [ [0.5,4], [6,10], [11,15], [55,100] ]
    band_labels - optional list of descriptive names for each freq band
            e.g. ['delta', 'theta', 'sigma', 'gamma']
    band_colors - optional list of colors to plot each freq band
            e.g. ['firebrick', 'limegreen', 'cyan', 'purple']
    win - time window (s) to collect SP data relative to given events
    mode - event(s) to analyze surrounding freq band power
           'pwaves' - plot power bands (PB) surrounding spontaneous P-waves (PB2)
                       * if plaser=True, PB1 is the power bands surrounding laser-triggered P-waves
                       * if null=True, PB1 is the power bands surrounding random control points
           'lsr' - plot power bands surrounding successful (PB1) and failed (PB2) laser pulses
           'null' - plot power bands surrounding random control points (PB1)
            * if plaser=True, PB2 is the power bands surrounding failed laser pulses
            * if plaser=False, PB2 is the power bands surrounding spontaneous P-waves
    recalc_highres - if True, recalculate high-resolution spectrogram from EEG, 
                      using $nsr_seg and $perc_overlap params
    tstart, tend - time (s) into recording to start and stop collecting data
    pnorm - method for spectrogram normalization (0=no normalization
                                              1=normalize SP by recording
                                              2=normalize SP by collected time window)
    pcalc - method for spectrogram calculation (0=load/calculate full SP and find event by index,
                                                1=calculate SP only for the time window surrounding event)
    psmooth, sf - smoothing params for spectrograms/vectors of freq band power
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    fmax - maximum frequency in collected spectrogram
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    plaser - if True, plot freq band power surrounding laser-triggered P-waves (if $mode='pwaves') or
             laser pulses (if $mode='lsr')
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    null - if True, plot freq band power surrounding randomized control points in $istate
    null_win - if > 0, qualifying "null" points must be free of P-waves and laser pulses in surrounding $null_win interval (s)
               if = 0, "null" points are randomly selected from all state indices
    null_match - the no. of random control points is matched with the no. of some other event type
                 'spon'    - # control points equals the # of spontaneous P-waves
                 'lsr'     - # control points equals the # of laser-triggered P-waves
                 'success' - # control points equals the # of successful laser pulses
                 'fail'    - # control points equals the # of failed laser pulses
                 'all lsr' - # control points equals the # of total laser pulses
    lsr_iso - if > 0, do not analyze laser pulses that are preceded within $lsr_iso s by 1+ P-waves
    pload - optional string specifying a filename to load the data (if False, data is collected from raw recording folders)
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    ylim - set y axis limits for graphs
    @Returns
    labels - list of mouse/recording names, or integers to number trials
    bdict1, bdict2 - dictionaries with timecourses of frequency band power surrounding specified events
                     (key = freq band label, value = subject x time bin matrix containing averaged
                      freq band power over time for each mouse/trial )
    x - vector of time points corresponding to mx columns (spanning time window specified by $win)
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    if len(band_labels) != len(bands):
        band_labels = bands
    if len(band_colors) != len(bands):
        band_colors = ['firebrick', 'limegreen', 'cyan', 'purple', 'darkgray', 'black'][0:len(bands)]
            
    if pcalc == 0:
        if pnorm == 0:
            signal_type = 'SP'  # pcalc=0 & pnorm=0 --> load SP from raw highres_SP, no normalization
            pnorm = 0
        elif pnorm == 1:
            signal_type = 'SP_NORM'  # pcalc=0 & pnorm=1 --> load SP from normalized highres_SP
            pnorm = 0
        elif pnorm == 2:
            signal_type = 'SP'  # pcalc=0 & pnorm=2 --> load SP from raw highres_SP, normalize within time window
            pnorm = 1
    elif pcalc == 1:
        if pnorm == 0:
            signal_type = 'SP_CALC'  # pcalc=1 & pnorm=0 --> calculate SP from EEG, do not normalize
            pnorm = 0
        elif pnorm == 1:
            signal_type = 'SP_CALC_NORM'  # pcalc=1 & pnorm=0 --> calculate SP from EEG, normalize by recording
            pnorm = 0
        elif pnorm == 2:
            signal_type = 'SP_CALC'  # pcalc=1 & pnorm=2 --> calculate SP from EEG, normalize within time window
            pnorm = 1
    
    states = {'total':'total', 1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # load SP data
    if pload:
        data = AS.load_surround_files(ppath, pload=pload, istate=istate, plaser=plaser, 
                                      null=null, signal_type=signal_type)
        if len(data) > 0:
            if plaser:
                lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data
            else:
                p_signal, null_signal, data_shape = data
        else:
            pload = False
    # collect SP data
    if not pload:
        if plaser:
            data = get_lsr_surround(ppath, recordings, istate=istate, win=win, signal_type=signal_type, 
                                    recalc_highres=recalc_highres, tstart=tstart, tend=tend, ma_thr=ma_thr,
                                    ma_state=ma_state, flatten_is=flatten_is, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                    null=null, null_win=null_win, null_match=null_match, post_stim=post_stim, 
                                    lsr_iso=lsr_iso, psave=psave, p_iso=p_iso, pcluster=pcluster, clus_event=clus_event)
            lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data
        elif not plaser:
            p_signal, null_signal, data_shape = get_surround(ppath, recordings, istate=istate, win=win, signal_type=signal_type, 
                                                             recalc_highres=recalc_highres, tstart=tstart, tend=tend, ma_thr=ma_thr,
                                                             ma_state=ma_state, flatten_is=flatten_is, nsr_seg=nsr_seg, 
                                                             perc_overlap=perc_overlap, null=null, null_win=null_win, psave=psave, 
                                                             p_iso=p_iso, pcluster=pcluster, clus_event=clus_event)
            
    freq = np.arange(0, data_shape[0]/2-0.5, 0.5) # frequencies
    x = np.linspace(-np.abs(win[0]), win[1], data_shape[1])

    if plaser:
        # for each brainstate, create dictionary with freq bands as keys and matrices
        # containing avg band power per time bin for each trial or mouse
        for s in istate:
            # dict1 for laser-triggered P-waves, dict2 for spontaneous P-waves
            if mode == 'pwaves':
                bdict1, labels = get_band_pwr(lsr_pwaves[s], freq, bands, mouse_avg=mouse_avg, 
                                              pnorm=pnorm, psmooth=psmooth, sf=sf, fmax=fmax)
                bdict2, labels = get_band_pwr(spon_pwaves[s], freq, bands, mouse_avg=mouse_avg, 
                                              pnorm=pnorm, psmooth=psmooth, sf=sf, fmax=fmax)
                title1 = 'Laser-triggered P-waves'
                title2 = 'Spontaneous P-waves'
            # dict1 for successful lsr, dict2 for failed lsr
            elif mode == 'lsr':
                bdict1, labels = get_band_pwr(success_lsr[s], freq, bands, mouse_avg=mouse_avg, 
                                              pnorm=pnorm, psmooth=psmooth, sf=sf, fmax=fmax)
                bdict2, labels = get_band_pwr(fail_lsr[s], freq, bands, mouse_avg=mouse_avg, 
                                              pnorm=pnorm, psmooth=psmooth, sf=sf, fmax=fmax)
                title1 = 'Successful laser pulses'
                title2 = 'Failed laser pulses'
            # dict1 for random control points, dict2 for failed lsr
            elif mode == 'null':
                bdict1, labels = get_band_pwr(null_pts[s], freq, bands, mouse_avg=mouse_avg, 
                                              pnorm=pnorm, psmooth=psmooth, sf=sf, fmax=fmax)
                bdict2, labels = get_band_pwr(fail_lsr[s], freq, bands, mouse_avg=mouse_avg, 
                                              pnorm=pnorm, psmooth=psmooth, sf=sf, fmax=fmax)
                title1 = 'Null points'
                title2 = 'Failed laser pulses'
                
            # plot timecourses of avg power for each frequency band
            fig, (lax, sax) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
            fig.suptitle('Band power averaged by ' + mouse_avg + ', state=' + str(s))
            for b,bl,bc in zip(bands, band_labels, band_colors):
                # PB1 - plot laser-triggered P-waves OR successful laser pulses
                bp1_avg = np.nanmean(bdict1[b], axis=0)
                bp1_yerr = np.nanstd(bdict1[b], axis=0) / np.sqrt(bdict1[b].shape[0])
                lax.plot(x, bp1_avg, color=bc, label=bl, linewidth=2)
                lax.fill_between(x, bp1_avg-bp1_yerr, bp1_avg+bp1_yerr, color=bc, alpha=0.3)
                # PB2 - plot spontaneous P-waves OR failed laser pulses
                bp2_avg = np.nanmean(bdict2[b], axis=0)
                bp2_yerr = np.nanstd(bdict2[b], axis=0) / np.sqrt(bdict2[b].shape[0])
                sax.plot(x, bp2_avg, color=bc, label=bl, linewidth=2)
                sax.fill_between(x, bp2_avg-bp2_yerr, bp2_avg+bp2_yerr, color=bc, alpha=0.3)
            sax.legend()
            lax.set_xlabel('Time (s)')
            sax.set_xlabel('Time (s)')
            [lax.set_ylabel('Power uV^2s') if (pnorm==0 and 'NORM' not in signal_type) else lax.set_ylabel('Rel. power')]
            lax.set_title(title1)
            sax.set_title(title2)
            # set y axes equal to each other
            if len(ylim) == 2:
                y=ylim
            else:
                y = [min(lax.get_ylim()[0], sax.get_ylim()[0]), max(lax.get_ylim()[1], sax.get_ylim()[1])]
            lax.set_ylim(y)
            sax.set_ylim(y)
    
    elif not plaser:
        # for each brainstate, create dictionary (keys=freq bands, values=subject x time bins)
        for s in istate:
            bdict, labels = get_band_pwr(p_signal[s], freq, bands, mouse_avg=mouse_avg, 
                                         pnorm=pnorm, psmooth=psmooth, sf=sf, fmax=fmax)
            # plot timecourse of avg freq band power surrounding spontaneous P-waves
            fig = plt.figure()
            fig.suptitle('Band power averaged by ' + mouse_avg + ', state=' + str(s))
            ax = plt.gca()
            for b,bl,bc in zip(bands, band_labels, band_colors):
                bp_avg = np.nanmean(bdict[b], axis=0)
                bp_yerr = np.nanstd(bdict[b], axis=0) / np.sqrt(bdict[b].shape[0])
                ax.plot(x, bp_avg, color=bc, label=bl, linewidth=2)
                ax.fill_between(x, bp_avg-bp_yerr, bp_avg+bp_yerr, color=bc, alpha=0.3)
            ax.legend()
            ax.set_xlabel('Time (s)')
            [ax.set_ylabel('Power uV^2s') if pnorm==0 else ax.set_ylabel('Rel. Power')]
            if len(ylim) == 2:
                ax.set_ylim(ylim)
    plt.show()
    if plaser:
        return labels, bdict1, bdict2, x
    else:
        return labels, bdict, x


def sp_profiles(ppath, recordings, spon_win=[1,1], frange=[6,15], recalc_highres=False, 
                tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=False, nsr_seg=2, 
                perc_overlap=0.95, pnorm=0, psmooth=0, pcalc=0, null=True, null_win=[-0.5,0.5], 
                null_match='spon', p_iso=0, pcluster=0, clus_event='waves', plaser=True, 
                lsr_win=[1,1], collect_win=[], post_stim=0.1, ci=68, 
                mouse_avg='mouse', lsr_iso=0, pload=False, psave=False):
    """
    Plot EEG spectral profiles associated with laser-triggered P-waves, spontaneous
    P-waves, failed laser pulses, and random control points
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    spon_win - time window (s) to collect data relative to P-waves and randomized points
    frange - [min, max] frequency in power spectral density plot
    recalc_highres - if True, recalculate high-resolution spectrogram from EEG, 
                      using $nsr_seg and $perc_overlap params
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram 
    pnorm - method for spectrogram normalization (0=no normalization
                                                  1=normalize SP by recording
                                                  2=normalize SP by collected time window)
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing) 
    pcalc - method for spectrogram calculation (0=load/calculate full SP and find event by index,
                                                1=calculate SP only for the time window surrounding event)
    null - if True, also collect data surrounding randomized control points in $istate
    null_win - if > 0, qualifying "null" points must be free of P-waves and laser pulses in surrounding $null_win interval (s)
               if = 0, "null" points are randomly selected from all state indices
    null_match - the no. of random control points is matched with the no. of some other event type
                     'spon'    - # control points equals the # of spontaneous P-waves
                     'lsr'     - # control points equals the # of laser-triggered P-waves
                     'success' - # control points equals the # of successful laser pulses
                     'fail'    - # control points equals the # of failed laser pulses
                     'all lsr' - # control points equals the # of total laser pulses
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    plaser - if True, plot laser-related events (laser pulses and triggered P-waves)
    lsr_win - time window (s) to collect data relative to successful and failed laser pulses
    collect_win - time window (s) to initally collect and normalize, before isolating the intervals specified by $spon_win and $lsr_win
               e.g. if collect_win = [-3,3], spon_win = [-0.5,0.5], and lsr_win=[0,1] --> SPs are collected as 6 s windows
               surrounding an event, then each frequency is normalized by its mean power within the 6 s window, then a smaller
               1 s window immediately surrounding the event is isolated and plotted
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                              integer between 0 and 100=confidence interval)
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    lsr_iso - if > 0, do not analyze laser pulses that are preceded within $lsr_iso s by 1+ P-waves
    pload - optional string specifying a filename to load the data (if False, data is collected from raw recording folders)
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    @Returns
    df2 -  dataframe containing the power values for frequency band(s) that were statistically compared between events 
    """
    states = {'total':'total', 1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if pcalc == 0:
        if pnorm == 1:
            signal_type = 'SP_NORM'; pnorm=0
        else:
            signal_type = 'SP'; pnorm=True if pnorm==2 else False
    if pcalc == 1:
        if pnorm == 1:
            signal_type = 'SP_CALC_NORM'
        else:
            signal_type = 'SP_CALC'; pnorm=True if pnorm==2 else False
    if ci == 'sem':
        ci = 68
    
    istate=[1]
    ifreq = np.arange(frange[0]*2, frange[1]*2+1)   # theta freq idx in SP
    freq = np.linspace(frange[0], frange[1], len(ifreq))  # theta frequencies
    
    if plaser:
        # load laser data
        if pload:
            filename = pload if isinstance(pload, str) else f'lsrSurround_{signal_type}'
            lsr_pwaves = {}
            spon_pwaves = {}
            success_lsr = {}
            fail_lsr = {}
            null_pts = {}
            try:
                for s in istate:
                    # load .mat files with stored SPs
                    lsr_pwaves[s] = so.loadmat(os.path.join(ppath, f'{filename}_lsr_pwaves_{s}.mat'))
                    spon_pwaves[s] = so.loadmat(os.path.join(ppath, f'{filename}_spon_pwaves_{s}.mat'))
                    success_lsr[s] = so.loadmat(os.path.join(ppath, f'{filename}_success_lsr_{s}.mat'))
                    fail_lsr[s] = so.loadmat(os.path.join(ppath, f'{filename}_fail_lsr_{s}.mat'))
                    if null:
                        null_pts[s] = so.loadmat(os.path.join(ppath, f'{filename}_null_pts_{s}.mat'))
                    # remove MATLAB keys so later functions can get recording list
                    for mat_key in ['__header__', '__version__', '__globals__']:
                        _ = lsr_pwaves[s].pop(mat_key)
                        _ = spon_pwaves[s].pop(mat_key)
                        _ = success_lsr[s].pop(mat_key)
                        _ = fail_lsr[s].pop(mat_key)
                        if null:
                            _ = null_pts[s].pop(mat_key)
                print('\nLoading data dictionaries ...\n')
            except:
                print('\nUnable to load .mat files - calculating new theta power values ...\n')
                pload = False
        
        # collect SPs
        if not pload:
            # use spon_win to collect SP data SURROUNDING spontaneous and laser-triggered P-waves and random null REM indices
            # use lsr_win to collect SP data FOLLOWING successful/failed laser events
            if len(collect_win) != 2:
                data = get_lsr_surround(ppath, recordings, istate=istate, win=spon_win, signal_type=signal_type, recalc_highres=recalc_highres,
                                    tstart=tstart, tend=tend, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is,
                                    nsr_seg=nsr_seg, perc_overlap=perc_overlap, null=null, null_win=null_win, null_match=null_match,
                                    post_stim=post_stim, lsr_iso=lsr_iso, lsr_win=lsr_win)
            # if using collect_win parameter, collect a larger window of SP data, normalize, and cut out spon_win/lsr_win
            else:
                data = get_lsr_surround(ppath, recordings, istate=istate, win=collect_win, signal_type=signal_type, recalc_highres=recalc_highres,
                                    tstart=tstart, tend=tend, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is,
                                    nsr_seg=nsr_seg, perc_overlap=perc_overlap, null=null, null_win=null_win, null_match=null_match,
                                    post_stim=post_stim, lsr_iso=lsr_iso, lsr_win=collect_win)
            lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape = data
            # save files
            if psave:
                filename = psave if isinstance(psave, str) else f'lsrSurround_{signal_type}'
                s = 1
                so.savemat(os.path.join(ppath, f'{filename}_lsr_pwaves_{s}.mat'), lsr_pwaves[s])
                so.savemat(os.path.join(ppath, f'{filename}_spon_pwaves_{s}.mat'), spon_pwaves[s])
                so.savemat(os.path.join(ppath, f'{filename}_success_lsr_{s}.mat'), success_lsr[s])
                so.savemat(os.path.join(ppath, f'{filename}_fail_lsr_{s}.mat'), fail_lsr[s])
                so.savemat(os.path.join(ppath, f'{filename}_null_pts_{s}.mat'), null_pts[s])
    elif not plaser:
        # load non-laser data
        if pload:
            filename = pload if isinstance(pload, str) else f'Surround_{signal_type}'
            spon_pwaves = {}
            null_pts = {}
            try:
                for s in istate:
                    # load .mat files with stored SPs
                    spon_pwaves[s] = so.loadmat(os.path.join(ppath, f'{filename}_pwaves_{s}.mat'))
                    if null:
                        null_pts[s] = so.loadmat(os.path.join(ppath, f'{filename}_null_{s}.mat'))
                    # remove MATLAB keys so later functions can get recording list
                    for mat_key in ['__header__', '__version__', '__globals__']:
                        _ = spon_pwaves[s].pop(mat_key)
                        if null:
                            _ = null_pts[s].pop(mat_key)
                data_shape = so.loadmat(os.path.join(ppath, f'{filename}_data_shape.mat'))['data_shape'][0]
                print('\nLoading data dictionaries ...\n')
            except:
                print('\nUnable to load .mat files - calculating new spectrograms ...\n')
                pload = False
        if not pload:
            p_signal, null_pts, data_shape = get_surround(ppath, recordings, istate=istate, win=spon_win, signal_type=signal_type, 
                                                             recalc_highres=recalc_highres, tstart=tstart, tend=tend, ma_thr=ma_thr,
                                                             ma_state=ma_state, flatten_is=flatten_is, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                                             null=null, null_win=null_win, psave=psave, p_iso=p_iso, pcluster=pcluster, clus_event=clus_event)

    # normalize SP with larger time window and later isolate smaller interval
    if len(collect_win) == 2:
        spon_pwaves = get_SP_subset(spon_pwaves[1], win=collect_win, sub_win=spon_win, pnorm=pnorm)
        null_pts = get_SP_subset(null_pts[1], win=collect_win, sub_win=spon_win, pnorm=pnorm)
        if plaser:
            lsr_pwaves = get_SP_subset(lsr_pwaves[1], win=collect_win, sub_win=spon_win, pnorm=pnorm)
            success_lsr = get_SP_subset(success_lsr[1], win=collect_win, sub_win=lsr_win, pnorm=pnorm)
            fail_lsr = get_SP_subset(fail_lsr[1], win=collect_win, sub_win=lsr_win, pnorm=pnorm)
    else:
        spon_pwaves = spon_pwaves[1]; null_pts = null_pts[1]
        if plaser:
            lsr_pwaves = lsr_pwaves[1]; success_lsr = success_lsr[1]; fail_lsr = fail_lsr[1];
    
    # collect spectral power for each event
    spon_p_thetapwr = {rec:[] for rec in recordings}
    null_thetapwr = {rec:[] for rec in recordings}
    if plaser:
        lsr_p_thetapwr = {rec:[] for rec in recordings}
        success_lsr_thetapwr = {rec:[] for rec in recordings}
        fail_lsr_thetapwr = {rec:[] for rec in recordings}

    for rec in recordings:
        # isolate mean power of theta frequencies
        spon_p_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in spon_pwaves[rec]]
        null_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in null_pts[rec]]
        if plaser:
            lsr_p_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in lsr_pwaves[rec]]
            success_lsr_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in success_lsr[rec]]
            fail_lsr_thetapwr[rec] = [np.mean(sp[ifreq, :], axis=1) for sp in fail_lsr[rec]]
    
    # create matrix of subject x frequency
    spon_mx, mice = mx2d(spon_p_thetapwr, mouse_avg)
    null_mx, _ = mx2d(null_thetapwr, mouse_avg)
    if plaser:
        lsr_mx, _ = mx2d(lsr_p_thetapwr, mouse_avg)
        success_mx, _ = mx2d(success_lsr_thetapwr, mouse_avg)
        fail_mx, _ = mx2d(fail_lsr_thetapwr, mouse_avg)
    # smooth spectrogram
    if psmooth:
        spon_mx = AS.convolve_data(spon_mx, psmooth, axis='x')
        null_mx = AS.convolve_data(null_mx, psmooth, axis='x')
        if plaser:
            lsr_mx = AS.convolve_data(lsr_mx, psmooth, axis='x')
            success_mx = AS.convolve_data(success_mx, psmooth, axis='x')
            fail_mx = AS.convolve_data(fail_mx, psmooth, axis='x')
            
    # store data in dataframe
    spon_data = [(mice[i], freq[j], spon_mx[i,j], 'spon') for j in range(len(freq)) for i in range(spon_mx.shape[0])]
    null_data = [(mice[i], freq[j], null_mx[i,j], 'null') for j in range(len(freq)) for i in range(null_mx.shape[0])]
    if plaser:
        lsr_data = [(mice[i], freq[j], lsr_mx[i,j], 'lsr') for j in range(len(freq)) for i in range(lsr_mx.shape[0])]
        success_data = [(mice[i], freq[j], success_mx[i,j], 'success') for j in range(len(freq)) for i in range(success_mx.shape[0])]
        fail_data = [(mice[i], freq[j], fail_mx[i,j], 'fail') for j in range(len(freq)) for i in range(fail_mx.shape[0])]
    if plaser:
        df = pd.DataFrame(columns=['mouse', 'freq', 'pow', 'group'], data=spon_data+null_data+lsr_data+fail_data+success_data)
    else:
        df = pd.DataFrame(columns=['mouse', 'freq', 'pow', 'group'], data=spon_data+null_data)
        
    # plot power spectrum
    plt.figure()
    sns.lineplot(data=df, x='freq', y='pow', hue='group', ci=ci, hue_order=['spon','null','lsr','fail'], 
                 palette={'null':'darkgray', 'fail':'red', 'spon':'green', 'lsr':'blue','success':'blue'})
    plt.show()
    
    # choose frequency band to statistically compare between events
    stat_bands = [(8, 20)]
    bins = [-1, 8, 20]

    stat_band_labels = [str(band) for band in stat_bands]
    b = np.digitize(df['freq'], bins, right=True)
    if len(stat_bands) > 1:
        df['band'] = [stat_band_labels[i-1] for i in b]
    else:
        df['band'] = [stat_band_labels[0] if i==2 else 'na' for i in b]
    
    # get power averages within frequency bands
    band_avgs = []
    for band in stat_band_labels:
        for group in ['spon', 'null', 'lsr', 'fail']:
            for m in mice:
                idx = np.intersect1d(np.intersect1d(np.where(df['mouse']==m)[0], np.where(df['group']==group)[0]), np.where(df['band']==band)[0])
                band_avgs.append((m, df['pow'].iloc[idx].mean(), group, band))
    df2 = pd.DataFrame(band_avgs, columns=['mouse', 'pow', 'group', 'band'])
    
    # boxplot
    plt.figure()
    sns.boxplot(x='group', y='pow', order=['null','fail','spon','lsr'], data=df2, whis=np.inf, fliersize=0, 
                palette={'null':'gray', 'fail':'red','spon':'green','lsr':'blue'})
    sns.pointplot(x='group', y='pow', hue='mouse', order=['null','fail','spon','lsr'], data=df2, ci=None, markers='', color='black')
    plt.ylim((0.8,1.4)); plt.gca().get_legend().remove()
    
    # stats
    res_anova = AnovaRM(data=df2, depvar='pow', subject='mouse', within=['group']).fit()
    mc = MultiComparison(df2['pow'], df2['group']).allpairtest(scipy.stats.ttest_rel, method='bonf')
    print(res_anova); print('p = ' + str(float(res_anova.anova_table['Pr > F']))); print(''); print(mc[0])
    
    return df, df2


def theta_pfreq(ppath, recordings, istate=2, r_theta=[6,10], r_delta=[0.5,4], 
                thres=[40,60], ma_thr=20, ma_state=3, flatten_is=4, 
                mouse_avg='mouse', exclude_noise=True, pplot=True):
    """
    Calculate average P-wave frequency during high-theta and low-theta time bins
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state to analyze
    r_theta, r_delta - [min, max] frequencies in theta and delta ranges
    thres - percentiles to use as thresholds for [low, high] theta power
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    exclude_noise - if False, ignore manually annotated LFP noise indices
                    if True, exclude time bins containing LFP noise from analysis
    pplot - if True, plot P-wave frequencies
    @Returns
    DF - dataframe containing P-wave frequencies for high-theta and low-theta bins
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if isinstance(istate, list):
        istate = istate[0]
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
        
    data_dict = {rec:[] for rec in recordings}
    
    for i,rec in enumerate(recordings):
        print(f'Getting P-waves for {rec} ({i+1}/{len(recordings)})')
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load spectrogram
        P = so.loadmat(os.path.join(ppath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        df = freq[1]-freq[0]
        # get ratio of theta:delta power
        itheta = np.where((freq>=r_theta[0]) & (freq<=r_theta[1]))[0]
        pow_theta = SP[itheta,:].sum(axis=0)*df
        idelta = np.where((freq>=r_delta[0]) & (freq<=r_delta[1]))[0]
        pow_delta = SP[idelta,:].sum(axis=0)*df
        thd_ratio = np.divide(pow_theta, pow_delta)
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt=dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
        
        # load LFP and P-wave indices
        if exclude_noise:
            # load noisy LFP indices, make sure no P-waves are in these regions
            LFP, p_idx, noise_idx = load_pwaves(ppath, rec, return_lfp_noise=True)[0:3]
            noise_dn = np.floor(np.divide(noise_idx, nbin)).astype('int')
        else:
            LFP, p_idx = load_pwaves(ppath, rec)[0:2]
            noise_dn = np.array((), dtype='int')
        
        # get indices of bins in $istate with high and low theta power
        sidx = np.where(M==istate)[0]
        sidx = np.setdiff1d(sidx, noise_dn)
        thres_hi = np.percentile(thd_ratio[sidx], thres[1])
        thres_lo = np.percentile(thd_ratio[sidx], thres[0])
        itheta_hi = np.intersect1d(sidx, np.where(thd_ratio >= thres_hi)[0])
        itheta_lo = np.intersect1d(sidx, np.where(thd_ratio <= thres_lo)[0])
        
        # collect P-wave frequency in each bin
        d = downsample_pwaves(LFP, p_idx, sr=sr, nbin=nbin, rec_bins=len(M))
        p_idx_dn, p_idx_dn_exact, p_count_dn, p_freq_dn = d
        if len(p_freq_dn) == len(M)-1:
            p_freq_dn = np.append(p_freq_dn, 0.)
        pfreq_hi = p_freq_dn[itheta_hi]
        pfreq_lo = p_freq_dn[itheta_lo]
        
        data_dict[rec] = [pfreq_hi, pfreq_lo]
        
    # create dataframe with P-wave frequency for each 'hi' theta and 'lo' theta time bin
    DF = pd.DataFrame(columns=['mouse', 'recording', 'theta', 'pfreq'])
    for rec in recordings:
        idf = rec.split('_')[0]
        df_hi = pd.DataFrame({'mouse':idf, 'recording':rec, 'theta':'hi', 'pfreq':data_dict[rec][0]})
        df_lo = pd.DataFrame({'mouse':idf, 'recording':rec, 'theta':'lo', 'pfreq':data_dict[rec][1]})
        DF = pd.concat([DF, df_hi, df_lo], axis=0, ignore_index=True)
    # average P-wave frequency by recording or mouse
    if mouse_avg in ['recording', 'mouse']:
        DF = DF.groupby([mouse_avg, 'theta']).mean().reset_index()
        
    if pplot:
        # plot P-wave frequency for each trial/recording/mouse
        fig = plt.figure()
        ax = plt.gca()
        if mouse_avg == 'trial':
            sns.stripplot(x='theta', y='pfreq', data=DF)
        else:
            sns.barplot(x='theta', y='pfreq', data=DF, ci=68, ax=ax)
            sns.pointplot(x='theta', y='pfreq', hue=mouse_avg, data=DF, ci=None, 
                          markers='', color='black', legend=False, ax=ax)
        ax.legend().remove()
        plt.show()
    
    if mouse_avg in ['recording', 'mouse']:
        # paired t-test comparing P-wave frequency between high and low-theta bins
        p = scipy.stats.ttest_rel(DF.loc[DF.theta=='hi', 'pfreq'], DF.loc[DF.theta=='lo', 'pfreq'])
        print('')
        print(f'P-wave frequency during high-theta vs. low-theta bins (state={istate})')
        print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}')


def lsr_hilbert(ppath, recordings, istate, bp_filt, stat='perc', mode='pwaves',
                min_state_dur=5, mouse_avg='mouse', ma_thr=20, ma_state=3, 
                flatten_is=False, post_stim=0.1, bins=9, pload=False, psave=False):
    """
    Plot P-wave or laser events in each phase of a filtered EEG signal
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state to analyze
    bp_filt - 2-element list specifying the lowest and highest frequencies to use
              for bandpass filtering
    stat - statistic to compute in each bin ('perc' = % of events in each bin,
                                             'count' = # of events in each bin)
    mode - type of data to plot ('pwaves' = distribution of spontaneous & laser-triggered P-waves,
                                 'lsr' = distribution of laser pulses and success rates)
    min_state_dur - minimum brain state duration (min) to be included in analysis
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
                  (all others are "spontaneous")
                Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    bins - no. histogram bins in the phase distribution plot
    pload - optional string specifying a filename to load the data (if False, data is collected from raw recording folders)
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    @Returns
    None
    """
     # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if isinstance(istate, list):
        istate = istate[0]
    
    # load data file
    if pload:
        filename = psave if isinstance(psave, str) else f'lsrSurround_LFP'
        try:
            lsr_p_rec = so.loadmat(os.path.join(ppath, f'{filename}_lsr_phase_{istate}.mat'))
            spon_p_rec = so.loadmat(os.path.join(ppath, f'{filename}_spon_phase_{istate}.mat'))
            success_p_rec = so.loadmat(os.path.join(ppath, f'{filename}_success_phase_{istate}.mat'))
            fail_p_rec = so.loadmat(os.path.join(ppath, f'{filename}_fail_phase_{istate}.mat'))

            for rec in recordings:
                lsr_p_rec[rec] = lsr_p_rec[rec][0]
                spon_p_rec[rec] = spon_p_rec[rec][0]
                success_p_rec[rec] = success_p_rec[rec][0]
                fail_p_rec[rec] = fail_p_rec[rec][0]
        except:
            print('Cannot load .mat files, calculating phases ...')
            pload = False
    # collect phase data
    if not pload:
        data = get_lsr_phase(ppath, recordings, istate, bp_filt=bp_filt, min_state_dur=min_state_dur, 
                             post_stim=post_stim, psave=psave)
        lsr_p_rec, spon_p_rec, success_p_rec, fail_p_rec = data[0:4]

    mice = list({rec.split('_')[0]:[] for rec in recordings})
    # create dictionary of recording data
    if mouse_avg == 'recording':
        idfs = recordings
        lsr_p_dict = lsr_p_rec
        spon_p_dict = spon_p_rec
        success_p_dict = success_p_rec
        fail_p_dict = fail_p_rec
    # create dictionary of mouse data
    elif mouse_avg == 'mouse':
        idfs = mice
        lsr_p_dict = {m:[] for m in mice}
        spon_p_dict = {m:[] for m in mice}
        success_p_dict = {m:[] for m in mice}
        fail_p_dict = {m:[] for m in mice}
        for rec in recordings:
            idf = rec.split('_')[0]
            lsr_p_dict[idf].append(lsr_p_rec[rec])
            spon_p_dict[idf].append(spon_p_rec[rec])
            success_p_dict[idf].append(success_p_rec[rec])
            fail_p_dict[idf].append(fail_p_rec[rec])
        for m in mice:
            lsr_p_dict[m] = list(chain.from_iterable(lsr_p_dict[m]))
            spon_p_dict[m] = list(chain.from_iterable(spon_p_dict[m]))
            success_p_dict[m] = list(chain.from_iterable(success_p_dict[m]))
            fail_p_dict[m] = list(chain.from_iterable(fail_p_dict[m]))
            
    # create matrix of subjects x phase bins
    if mouse_avg == 'mouse' or mouse_avg == 'recording':
        lsr_p_mx = np.zeros((len(idfs), bins))
        spon_p_mx = np.zeros((len(idfs), bins))
        lsr_success_mx = np.zeros((len(idfs), bins))
        lsr_fail_mx = np.zeros((len(idfs), bins))
        lsr_all_mx = np.zeros((len(idfs), bins))
        lsr_prob_mx = np.zeros((len(idfs), bins))
        df = pd.DataFrame()
        for row, idf in enumerate(idfs):
            # get P-wave stats
            lsr_p_hist = np.histogram(lsr_p_dict[idf], bins=bins, range=(-3.14, 3.14))[0]           # no. laser-triggered P-waves per phase
            spon_p_hist = np.histogram(spon_p_dict[idf], bins=bins, range=(-3.14, 3.14))[0]         # no. spontaneous P-waves per phase
            lsr_p_prob = [(i/sum(lsr_p_hist))*100 for i in lsr_p_hist]                              # probability of a laser-triggered P-wave occurring in each phase
            spon_p_prob = [(i/sum(spon_p_hist))*100 for i in lsr_p_hist]                            # probability of a spontaneous P-wave occurring in each phase
            # get laser stats
            success_p_hist = np.histogram(success_p_dict[idf], bins=bins, range=(-3.14, 3.14))[0]   # no. successful laser pulses per phase
            fail_p_hist = np.histogram(fail_p_dict[idf], bins=bins, range=(-3.14, 3.14))[0]         # no. failed laser pulses per phase
            lsr_per_phase = np.sum((success_p_hist, fail_p_hist), axis=0)                           # no. total laser pulses per phase
            lsr_all_prob = (lsr_per_phase / sum(lsr_per_phase))*100                                 # probability of a random laser pulse occurring in each phase
            lsr_success_prob = [(i/sum(success_p_hist))*100 for i in success_p_hist]                # probability of a given successful laser pulse occurring in each phase
            lsr_fail_prob = [(i/sum(fail_p_hist))*100 for i in fail_p_hist]                         # probability of a given failed laser pulse occurring in each phase
            lsr_raw_prob = [(i/j)*100 for i,j in zip(success_p_hist, lsr_per_phase)]                # in each phase, probability of a given laser pulse being successful
            lsr_prob_mx[row, :] = lsr_raw_prob
            
            # fill in data matrix with P-wave counts or percentages
            if stat == 'count':
                lsr_p_mx[row, :] = lsr_p_hist
                spon_p_mx[row, :] = spon_p_hist
                lsr_success_mx[row, :] = success_p_hist
                lsr_fail_mx[row, :] = fail_p_hist
                lsr_all_mx[row, :] = lsr_per_phase
            elif stat == 'perc':
                lsr_p_mx[row, :] = lsr_p_prob
                spon_p_mx[row, :] = spon_p_prob
                lsr_success_mx[row, :] = lsr_success_prob
                lsr_fail_mx[row, :] = lsr_fail_prob
                lsr_all_mx[row, :] = lsr_all_prob
    
    # create histograms from arrays of phase values
    elif 'trial' in mouse_avg:
        df = pd.DataFrame()
        for rec in recordings:
            for event,ddict in zip(['lsr','spon','success','fail'],
                                   [lsr_p_rec, spon_p_rec, success_p_rec, fail_p_rec]):
                df = pd.concat([df, pd.DataFrame({'mouse' : rec.split('_')[0],
                                                  'recording' : rec,
                                                  'state' : istate,
                                                  'event' : event,
                                                  'phase' : ddict[rec]})], axis=0, ignore_index=True)
        # get all phases of events across mice/recordings
        lsr_phases = list(chain.from_iterable([lsr_p_rec[rec] for rec in recordings]))
        spon_phases = list(chain.from_iterable([spon_p_rec[rec] for rec in recordings]))
        success_phases = list(chain.from_iterable([success_p_rec[rec] for rec in recordings]))
        fail_phases = list(chain.from_iterable([fail_p_rec[rec] for rec in recordings]))
        
        # calculate histograms and probabilities
        lsr_p_hist = np.histogram(lsr_phases, bins=bins, range=(-3.14, 3.14))[0]
        spon_p_hist = np.histogram(spon_phases, bins=bins, range=(-3.14, 3.14))[0]
        lsr_p_prob = [(i/sum(lsr_p_hist))*100 for i in lsr_p_hist]
        spon_p_prob = [(i/sum(spon_p_hist))*100 for i in spon_p_hist]  
        success_p_hist = np.histogram(success_phases, bins=bins, range=(-3.14, 3.14))[0]
        fail_p_hist = np.histogram(fail_phases, bins=bins, range=(-3.14, 3.14))[0]
        lsr_per_phase = np.sum((success_p_hist, fail_p_hist), axis=0)
        lsr_all_prob = (lsr_per_phase / sum(lsr_per_phase))*100 
        lsr_success_prob = [(i/sum(success_p_hist))*100 for i in success_p_hist]
        lsr_fail_prob = [(i/sum(fail_p_hist))*100 for i in fail_p_hist]
        lsr_raw_prob = [(i/j)*100 for i,j in zip(success_p_hist, lsr_per_phase)] 
        lsr_prob_mx = np.array((lsr_raw_prob), ndmin=2)
        if stat == 'count':
            lsr_p_mx = np.reshape(lsr_p_hist, (1,bins))
            spon_p_mx = np.reshape(spon_p_hist, (1,bins))
            lsr_success_mx = np.reshape(success_p_hist, (1,bins))
            lsr_fail_mx = np.reshape(fail_p_hist, (1,bins))
            lsr_all_mx = np.reshape(lsr_per_phase, (1,bins))
        elif stat == 'perc':
            lsr_p_mx = np.array((lsr_p_prob), ndmin=2)
            spon_p_mx = np.array((spon_p_prob), ndmin=2)
            lsr_success_mx = np.array((lsr_success_prob), ndmin=2)
            lsr_fail_mx = np.array((lsr_fail_prob), ndmin=2)
            lsr_all_mx = np.array((lsr_all_prob), ndmin=2)
        
        # Rayleigh's test for non-uniform distributions
        from astropy.stats import rayleightest
        print('')
        print('LASER-TRIGGERED P-WAVE PHASES')
        print(f'Rayleigh test of uniformity - p-value={round(rayleightest(np.array((lsr_phases))), 5)}')
        print('')
        print('SPONTANEOUS P-WAVE PHASES')
        print(f'Rayleigh test of uniformity - p-value={round(rayleightest(np.array((spon_phases))), 5)}')
        print('')

    # get total number of events in each phase
    if stat == 'count':
        if mouse_avg == 'recording' or mouse_avg == 'mouse':
            print('WARNING : since stat "count" was selected, averaging within {mouse_avg} was not performed.')
        lsr_p_data = np.nansum(lsr_p_mx, axis=0)
        lsr_p_yerr = 0
        spon_p_data = np.nansum(spon_p_mx, axis=0)
        spon_p_yerr = 0
        lsr_success_data = np.nansum(lsr_success_mx, axis=0)
        lsr_success_yerr = 0
        lsr_fail_data = np.nansum(lsr_fail_mx, axis=0)
        lsr_fail_yerr = 0
        lsr_all_data = np.nansum(lsr_all_mx, axis=0)
        lsr_all_yerr = 0
        title = 'Number of'
    # get probability of events in each phase
    elif stat == 'perc':
        lsr_p_data = np.nanmean(lsr_p_mx, axis=0)
        lsr_p_yerr = np.nanstd(lsr_p_mx, axis=0) / np.sqrt(lsr_p_mx.shape[0])
        spon_p_data = np.nanmean(spon_p_mx, axis=0)
        spon_p_yerr = np.nanstd(spon_p_mx, axis=0) / np.sqrt(spon_p_mx.shape[0])
        lsr_success_data = np.nanmean(lsr_success_mx, axis=0)
        lsr_success_yerr = np.nanstd(lsr_success_mx, axis=0) / np.sqrt(lsr_success_mx.shape[0])
        lsr_fail_data = np.nanmean(lsr_fail_mx, axis=0)
        lsr_fail_yerr = np.nanstd(lsr_fail_mx, axis=0) / np.sqrt(lsr_fail_mx.shape[0])
        lsr_all_data = np.nanmean(lsr_all_mx, axis=0)
        lsr_all_yerr = np.nanstd(lsr_all_mx, axis=0) / np.sqrt(lsr_all_mx.shape[0])
        title = '%'
    lsr_prob_data = np.nanmean(lsr_prob_mx, axis=0)
    lsr_prob_yerr = np.nanstd(lsr_prob_mx, axis=0) / np.sqrt(lsr_prob_mx.shape[0])
    
    # plot figure
    
    x = np.linspace(-3.14, 3.14, bins)
    if mode == 'pwaves':
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
        # spontaneous P-waves
        ax1.bar(x, lsr_p_data, color='blue', edgecolor='black', yerr=lsr_p_yerr)
        ax1.set_title(f'{title} laser-triggered P-waves')
        # laser-triggered P-waves
        ax2.bar(x, spon_p_data, color='green', edgecolor='black', yerr=spon_p_yerr)
        ax2.set_title(f'{title} spontaneous P-waves')
        ax2.set_xlabel('Phase')
    elif mode == 'lsr':
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
        # successful lsr
        ax1.bar(x, lsr_success_data, color='springgreen', edgecolor='black', yerr=lsr_success_yerr)
        ax1.set_title(f'{title} successful laser pulses')
        # total lsr
        ax2.bar(x, lsr_all_data, color='gray', edgecolor='black', yerr=lsr_all_yerr)
        ax2.set_title(f'{title} total laser pulses')
        # probability of laser success
        ax3.bar(x, lsr_prob_data, color='purple', edgecolor='black', yerr=lsr_prob_yerr)
        ax3.set_title('Probablity of successful laser pulse')
        ax3.set_xlabel('Phase')
    fig.suptitle(f'State={istate}, averaged by {mouse_avg}, stat = {stat}, bp_filt = {bp_filt}')
    plt.show()
    return df
    

def lsr_pwaves_sumstats(ppath, recordings, istate, tstart=0, tend=-1, ma_thr=20, 
                        ma_state=3, flatten_is=False, exclude_noise=False, post_stim=0.1, 
                        p_iso=0, pcluster=0, clus_event='waves', lsr_iso=0, mouse_avg='mouse'):
    """
    Get summary stats for optogenetically triggered P-wave recordings
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    exclude_noise - if False, ignore manually annotated LFP noise indices
                    if True, exclude time bins containing LFP noise from analysis
    post_stim - P-waves within $post_stim s of laser onset are "laser-triggered"
    p_iso, pcluster - inter-P-wave interval thresholds for detecting single and clustered P-waves
    clus_event - type of info to return for P-wave clusters
    lsr_iso - if > 0, do not analyze laser pulses that are preceded within $lsr_iso s by 1+ P-waves
    @Returns
    df - dataframe with # and % of laser-triggered vs spontaneous P-waves and successful
         vs. failed laser pulses
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if not isinstance(istate, list):
        istate = [istate]
    
    # create dataframe
    cols = ['mouse', 'recording', 'state', '# lsr p-waves', '# spon p-waves', '# total p-waves', 
            '% lsr p-waves', '# success lsr', '# fail lsr', '# total lsr', '% success lsr']
    df = pd.DataFrame(columns=cols)
    
    lsr_elim_counts = {'success':0, 'fail':0}
    
    for rec in recordings:
        idf = rec.split('_')[0]
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M, _ = sleepy.load_stateidx(ppath, rec)
        M = AS.adjust_brainstate(M, dt, ma_thr, ma_state, flatten_is)
        
        # load LFP and P-wave indices
        if exclude_noise:
            # load noisy LFP indices, make sure no P-waves are in these regions
            LFP, p_idx, noise_idx = load_pwaves(ppath, rec, return_lfp_noise=True)[0:3]
            p_idx = np.setdiff1d(p_idx, noise_idx)
        else:
            LFP, p_idx = load_pwaves(ppath, rec)[0:2]
        # isolate single or clustered P-waves
        if p_iso and pcluster:
            print('ERROR: cannot accept both p_iso and pcluster arguments')
            return
        elif p_iso:
            p_idx = get_p_iso(p_idx, sr, win=p_iso)
        elif pcluster:
            p_idx = get_pclusters(p_idx, sr, win=pcluster, return_event=clus_event)
        
        # load laser and find laser-triggered P=waves
        lsr = sleepy.load_laser(ppath, rec)
        data = list(get_lsr_pwaves(p_idx, lsr, post_stim, sr))
        
        # define start and end points of analysis
        istart = int(np.round(tstart*sr))
        iend = len(LFP)-1 if tend==-1 else int(np.round(tend*sr))
        iidx = np.intersect1d(np.where(p_idx >= istart)[0], np.where(p_idx <= iend)[0])
        p_idx = p_idx[iidx]
        for i,d in enumerate(data):
            d = np.array(d)
            iidx = np.intersect1d(np.where(d >= istart)[0], np.where(d <= iend)[0])
            data[i] = d[iidx]
        lsr_pi, spon_pi, success_lsr, fail_lsr = data
        if exclude_noise:
            fail_lsr = np.setdiff1d(fail_lsr, noise_idx)
        all_lsr = np.concatenate((success_lsr, fail_lsr))
        
        if lsr_iso > 0:
            # check for P-waves within the $lsr_iso window
            iwin = int(lsr_iso*sr)
            iso_elim = [l for l in all_lsr if np.intersect1d(p_idx, np.arange(l - iwin, l)).size > 0]
            lsr_elim_counts['success'] += len(np.intersect1d(success_lsr, iso_elim))
            success_lsr = np.setdiff1d(success_lsr, iso_elim)
            lsr_elim_counts['fail'] += len(np.intersect1d(fail_lsr, iso_elim))
            fail_lsr = np.setdiff1d(fail_lsr, iso_elim)
            all_lsr = np.concatenate((success_lsr, fail_lsr))
        
        # get no. of successful laser pulses / total pulses for each state
        lsr_states = np.array([M[int(round(l/nbin))] for l in all_lsr if l < (len(M)-1)*nbin])
        num_lsr_success = np.array([len(np.intersect1d(success_lsr, all_lsr[np.where(lsr_states==s)[0]])) for s in istate])
        num_lsr_fail = np.array([len(np.intersect1d(fail_lsr, all_lsr[np.where(lsr_states==s)[0]])) for s in istate])
        
        # get no. laser-triggered P-waves / total P-waves for each state
        pi_states = np.array([M[int(round(i/nbin))] for i in p_idx if i < (len(M)-1)*nbin])
        num_lsr_pi = np.array([len(np.intersect1d(lsr_pi, p_idx[np.where(pi_states==s)[0]])) for s in istate])
        num_spon_pi = np.array([len(np.intersect1d(spon_pi, p_idx[np.where(pi_states==s)[0]])) for s in istate])
        
        # collect stats
        ddf = pd.DataFrame({'mouse':[idf]*len(istate),
                            'recording':[rec]*len(istate),
                            'state':istate,
                            '# lsr p-waves' : num_lsr_pi,
                            '# spon p-waves' : num_spon_pi,
                            '# total p-waves' : num_lsr_pi + num_spon_pi,
                            '% lsr p-waves' : 0,
                            '# success lsr' : num_lsr_success,
                            '# fail lsr' : num_lsr_fail,
                            '# total lsr' : num_lsr_success + num_lsr_fail,
                            '% success lsr' : 0})
        df = pd.concat([df, ddf], axis=0, ignore_index=True)
        
    if mouse_avg == 'mouse':
        df = df.iloc[:, np.where(df.columns != 'recording')[0]]
        df = df.groupby(['mouse','state']).sum().reset_index()
        
    df['% lsr p-waves'] = (df['# lsr p-waves'] / df['# total p-waves']) * 100
    df['% success lsr'] = (df['# success lsr'] / df['# total lsr']) * 100
        
    
    if lsr_iso > 0:
        print(f"{lsr_elim_counts['success']} successful laser pulses and {lsr_elim_counts['fail']} failed laser pulses were eliminated due to closely preceding P-waves.")
        
    return df
        

def lsr_state_success(stats_df, istate, jstate=[], flatten_is=4, ci='sem'):
    """
    Plot success rate of laser pulses in each brain state
    @Params
    stats_df - dataframe containing the brain state of each successful and
               failed laser pulse (output of get_lsr_stats)
    istate - brain state(s) to analyze for success rate of true laser pulses
    jstate - brain state(s) to analyze for "success rate" of sham (jittered) laser pulses
    flatten_is - brain state for transition sleep
    ci - error bars ('sd'=standard deviation, 'sem'=standard error, 
                     integer between 0 and 100=confidence interval)
    @Returns
    df - dataframe containing laser success rate for each mouse in each brain state
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    dfstates = np.unique(stats_df.state)
    if 5 not in dfstates:
        states[4] = 'IS'
    mice = list(stats_df.mouse.unique())
    
    d = {'mouse':[], 'state':[], 'perc':[]}
    
    for s in istate:
        # calculate laser success by mouse for each brain state
        state_df = stats_df.iloc[np.where(stats_df['state'] == s)[0], :]
        for m in mice:
            mouse_df = state_df.iloc[np.where(state_df['mouse'] == m)[0], :]
            num_success_lsr = len(np.where(mouse_df['event'] == 'successful lsr')[0])
            num_fail_lsr = len(np.where(mouse_df['event'] == 'failed lsr')[0])
            num_total_lsr = num_success_lsr + num_fail_lsr
            if num_total_lsr > 0:
                perc_success = round((num_success_lsr / num_total_lsr)*100, 2)
            else:
                perc_success = np.nan
            d['mouse'].append(m); d['state'].append(states[s]); d['perc'].append(perc_success)
    for s in jstate:
        # calculate "success rate" for jittered laser pulses
        state_df = stats_df.iloc[np.where(stats_df['state'] == s)[0], :]
        for m in mice:
            mouse_df = state_df.iloc[np.where(state_df['mouse'] == m)[0], :]
            num_success_jlsr = len(np.where(mouse_df['event'] == 'jitter success')[0])
            num_fail_jlsr = len(np.where(mouse_df['event'] == 'jitter fail')[0])
            num_total_jlsr = num_success_jlsr + num_fail_jlsr
            if num_total_jlsr > 0:
                perc_success = round((num_success_jlsr / num_total_jlsr)*100, 2)
            else:
                perc_success = np.nan
            d['mouse'].append(m); d['state'].append('sham '+states[s]); d['perc'].append(perc_success);
    
    # collect state-dependent success rates and plot in bar graph
    df = pd.DataFrame(d)
    if ci == 'sem':
        ci = 'se'
    elif type(ci) == int:
        ci = ('ci',ci)
    plt.figure()
    sns.barplot(x='state', y='perc', data=df, errorbar=ci, palette={'REM':'cyan',
                                                                    'Wake':'darkviolet',
                                                                    'NREM':'darkgray', 
                                                                    'IS':'darkblue',
                                                                    'sham REM':'cadetblue',
                                                                    'sham wake':'rebeccapurple',
                                                                    'sham NREM':'dimgray', 
                                                                    'sham IS':'darkgreen'})
    #sns.pointplot(x='state', y='perc', hue='mouse', data=df, ci=None, markers='', color='black')
    lines = sns.lineplot(x='state', y='perc', hue='mouse', data=df, errorbar=None, 
                         markersize=0, linewidth=2, legend=False)
    _ = [l.set_color('black') for l in lines.get_lines()]
    plt.ylabel('% probability of successful laser pulse')
    plt.title('State-dependent probability of laser-triggered P-wave')
    plt.show()
    
    # stats
    state_names = list(set(df['state']))
    # compare 2 brain states with paired t-test
    if len(state_names) == 2:
        data1 = df['perc'].iloc[np.where(df['state'] == state_names[0])[0]]
        data2 = df['perc'].iloc[np.where(df['state'] == state_names[1])[0]]
        p = scipy.stats.ttest_rel(data1, data2, nan_policy='omit')
        sig='yes' if p.pvalue < 0.05 else 'no'
        print('')
        print(f'Laser success {state_names[0]} vs {state_names[1]}  -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}, sig={sig}')
        print('')
    # compare >2 brain states with repeated measures ANOVA
    elif len(state_names) > 2:
        res_anova = AnovaRM(data=df, depvar='perc', subject='mouse', within=['state']).fit()
        mc = MultiComparison(df['perc'], df['state']).allpairtest(scipy.stats.ttest_rel, method='bonf')
        print(res_anova); print('p = ' + str(float(res_anova.anova_table['Pr > F']))); print(''); print(mc[0])
    return df


def lsr_pwave_latency(df, istate, jitter=False, bins='auto', binrange=(0,100)):
    """
    Plot distribution of latencies of laser-triggered P-waves from laser onset
    @Params
    df - dataframe containing the latency of each laser-triggered P-wave from
         the laser onset (output of get_lsr_stats)
    istate - brain state to analyze
    jitter - if True, also plot latencies of "laser-triggered P-waves" from the
              onset of sham (jittered) laser pulses
    bins - number of bins in histogram
    binrange - (low, high) value for histogram
    @Returns
    None
    """
    # clean data inputs
    if type(bins) != int:
        bins = 'auto'
    if type(binrange) not in [list, tuple] or len(binrange) != 2:
        binrange = None
    
    state_df = df.iloc[np.where(df['state'] == istate)[0], :].reset_index(drop=True)
    
    plt.figure()
    # plot P-wave latencies from both true and sham laser pulses
    if jitter:
        sns.histplot(x='latency', hue='event', data=state_df, stat='count', bins=bins,
                     binrange=binrange, hue_order=['sham pwave', 'lsr-triggered pwave'],
                     palette={'lsr-triggered pwave':'blue', 'sham pwave':'gray'})
    # plot P-wave latencies from true laser pulses only
    else:
        sns.histplot(x='latency', hue='event', data=state_df, stat='count', bins=bins,
                     binrange=binrange, hue_order=['lsr-triggered pwave'],
                     palette={'lsr-triggered pwave':'blue'})
    plt.xlabel('Latency from laser (ms)')
    plt.ylabel('Number of laser pulses')
    plt.show()
    return state_df
    

def lsr_pwave_size(df, istate, stat='amp2', plotMode='0', mouse_avg='mouse', nbins=12):
    """
    Compare mean amplitudes/half-widths of spontaneous and laser-triggered P-waves
    @Params
    df - dataframe containing the amplitude and half-width of each spontaneous and
          laser-triggered P-wave (output of get_lsr_stats)
    istate - brain state to analyze
    stat -  measurement to plot
           'amp'       - plot raw LFP values of P-wave at the negative peak
           'amp2'      - plot vertical distance between negative peak and beginning of waveform
           'halfwidth' - plot width of P-wave at 1/2 max amplitude
    plot_type -  type of plot to show
           '0' - bar plot of mean $stat of spontaneous vs laser-triggered P-waves
                '1' = color-coded dots for mice; '2' = black dots for mice
                '3' = color-coded lines for mice; '4' = black lines for mice
           'h' - histogram plots of the distributions of spontaneous vs laser-triggered P-waves by $stat
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    nbins - no. bins in histogram distribution plots
    @Returns
    None
    """
    # drop rows with NaN values in stat column (eliminates laser pulses and super-small P-waves)
    df = df.dropna(axis=0, subset=[stat])
    
    if istate != 0:
        state_df = df.iloc[np.where(df['state'] == istate)[0], :]
    else:
        state_df = df
    lsr_p_idx = np.where(state_df['event'] == 'lsr-triggered pwave')[0]
    spon_p_idx = np.where(state_df['event'] == 'spontaneous pwave')[0]
    
    mice = list(state_df.mouse.unique())
    
    # bar plot
    if '0' in plotMode:
        plt.figure()
        # average within mice
        if mouse_avg == 'mouse':
            spon_amps = np.zeros((len(mice),))
            lsr_amps = np.zeros((len(mice),))
            for i,m in enumerate(mice):
                ms_state_df = state_df.iloc[np.where(state_df['mouse'] == m)[0], :]
                spon_amps[i] = (ms_state_df[stat].iloc[np.where(ms_state_df['event'] == 'spontaneous pwave')[0]]).mean()
                lsr_amps[i] = (ms_state_df[stat].iloc[np.where(ms_state_df['event'] == 'lsr-triggered pwave')[0]]).mean()
            # plot mouse-averaged $stat
            plt.bar(['spon', 'lsr'], [np.nanmean(spon_amps), np.nanmean(lsr_amps)], 
                    yerr=[np.nanstd(spon_amps)/np.sqrt(len(spon_amps)), np.nanstd(lsr_amps)/np.sqrt(len(spon_amps))], 
                    color=['gray','blue'])
            # plot individual mice
            mcs = {}
            for m in mice:
                mcs.update(AS.colorcode_mice(m))
            for midx, mname in enumerate(mice):
                points = [spon_amps[midx], lsr_amps[midx]]
                if '1' in plotMode: markercolor = 'black'
                elif '2' in plotMode: markercolor = mcs[mname]
                if '3' in plotMode: linecolor = 'black'
                elif '4' in plotMode: linecolor = mcs[mname]
                if '1' in plotMode or '2' in plotMode:
                    plt.plot(['spon', 'lsr'], points, color=markercolor, marker='o', ms=7, markeredgewidth=1, 
                             linewidth=0, markeredgecolor='black', label=mname, clip_on=False)[0]
                    if markercolor != 'black': plt.legend()
                if '3' in plotMode or '4' in plotMode:
                    plt.plot(['spon', 'lsr'], points, color=linecolor, linewidth=2, label=mname)
                    if linecolor != 'black': plt.legend()
        
        # plot individual P-wave trials       
        elif 'trial' in mouse_avg:
            spon_amps = state_df[stat].iloc[spon_p_idx]
            lsr_amps = state_df[stat].iloc[lsr_p_idx]
            sns.barplot(x='event', y=stat, data=state_df, ci='sd', order=['spontaneous pwave', 'lsr-triggered pwave'], 
                        palette={'spontaneous pwave':'gray', 'lsr-triggered pwave':'blue'})
            if '5' in plotMode:
                sns.stripplot(x='event', y=stat, data=state_df, order=['spontaneous pwave', 'lsr-triggered pwave'], color='black')
        if 'amp' in stat: plt.ylabel(f'P-wave amplitude ({stat})')
        elif stat=='halfwidth': plt.ylabel('P-wave halfwidth (ms)')
        plt.title(f'Spontaneous vs laser-triggered P-waves (state={istate})')
        plt.show()

        # stats
        if mouse_avg == 'mouse':
            p = scipy.stats.ttest_rel(spon_amps, lsr_amps, nan_policy='omit')
        elif 'trial' in mouse_avg:
            p = scipy.stats.ttest_ind(spon_amps, lsr_amps, equal_var=False, nan_policy='omit')
        print('')
        print(f'###   Spontaneous vs laser-triggered P-wave {stat} (state={istate})')
        if mouse_avg == 'mouse':
            print('paired t-test')
        elif 'trial' in mouse_avg:
            print("unpaired Welch's 't-test")
        print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}')
        print('')
    
    # histogram plots
    if 'h' in plotMode:
        fig, (lax, sax) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        lsr_amps = state_df[stat].iloc[lsr_p_idx]
        spon_amps = state_df[stat].iloc[spon_p_idx]
        
        # # get percent of spontaneous and laser-triggered P-waves in each $stat bin
        lsr_data, b1 = np.histogram(lsr_amps, bins=nbins)
        spon_data, b2 = np.histogram(spon_amps, bins=nbins)
        lsr_data = lsr_data/sum(lsr_data)*100
        spon_data = spon_data/sum(spon_data)*100
        ylabel = '% pwaves'
        lsr_x = np.linspace(b1[0], b1[-1], nbins)
        spon_x = np.linspace(b2[0], b2[-1], nbins)
        x = np.sort(np.concatenate((lsr_x, spon_x)))
        if 'amp' in stat:
            xlabel = f'P-wave amplitude ({stat})'
        elif stat=='halfwidth':
            xlabel = 'P-wave half-width (ms)'
            
        # plot laser pulse distributions
        width = (x[-1] - x[0])/len(x)
        lax.bar(lsr_x, lsr_data, width=width, color='blue', edgecolor='black', label = 'lsr-triggered P-waves')
        sax.bar(spon_x, spon_data, width=width, color='gray', edgecolor='black', label = 'spontaneous P-waves')
        fig.suptitle(f'P-Wave {stat} (state={istate})')
        lax.set_xlabel(xlabel)
        sax.set_xlabel(xlabel)
        xlm = [min(lax.get_xlim()[0], sax.get_xlim()[0]), max(lax.get_xlim()[1], sax.get_xlim()[1])]
        ylm = [min(lax.get_ylim()[0], sax.get_ylim()[0]), max(lax.get_ylim()[1], sax.get_ylim()[1])]
        lax.set_xlim(xlm); sax.set_xlim(xlm)
        lax.set_ylim(ylm); sax.set_ylim(ylm)
        lax.set_ylabel(ylabel); sax.set_ylabel(ylabel)
        lax.legend(); sax.legend()
        plt.show()
        
        # kolmogorov-smirnov test of distributions
        p = scipy.stats.ks_2samp(spon_amps, lsr_amps)
        print('')
        print(f'###   K-S test for spontaneous vs laser-triggered P-wave {stat} (state={istate})')
        print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}')
        print('')


def lsr_prev_theta_success(ppath, recordings, win=[-2,0], theta_band=[6,15], mode='power', 
                           post_stim=0.1, ma_thr=20, ma_state=3, flatten_is=False, 
                           nsr_seg=2, perc_overlap=0.95, pnorm=0, psmooth=0, pcalc=0,
                           mouse_avg='mouse', ci=68, nbins=10, prange1=(0,5), prange2=(0,5),
                           pload=False, psave=False):
    """
    Correlate success rate of the laser during REM sleep with power or frequency of EEG 
    theta band in the preceding seconds
    @Params
    ppath - base folder
    recordings - list of recordings
    win - time window (s) to collect data relative to laser pulses 
           e.g. [-2,0] --> 2 s preceding pulse onset
    theta_band - [min, max] frequency in power spectral density plot
    mode - type of analysis/plot to execute
           'spectrum'  - plot normalized power spectral densities
           'power'     - plot normalized theta power
           'mean freq' - plot mean theta frequency
    post_stim - Laser pulses followed by 1+ P-waves within $post_stim s are "successful"
                  (all others are "failed")
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    pnorm - method for spectrogram normalization (0=no normalization
                                                  1=normalize SP by recording
                                                  2=normalize SP by collected time window)
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing) 
    pcalc - method for spectrogram calculation (0=load/calculate full SP and find event by index,
                                                1=calculate SP only for $win time interval surrounding event)
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                              integer between 0 and 100=confidence interval)
    nbins, prange1, prange2 - no. of histogram bins and histogram ranges for overlapping
                              laser pulse distributions and success probability distribution
    pload - optional string specifying a filename to load the data (if False, data is collected from raw recording folders)
    psave - optional string specifying a filename to save the data (if False, data is not saved)
    @Returns
    None
    """
    # clean data inputs
    if not isinstance(recordings, list):
        recordings = [recordings]
    if pcalc == 0:
        if pnorm == 0:
            signal_type = 'SP'  # pcalc=0 & pnorm=0 --> load SP from raw highres_SP, no normalization
            pnorm = False
        elif pnorm == 1:
            signal_type = 'SP_NORM'  # pcalc=0 & pnorm=1 --> load SP from normalized highres_SP
            pnorm = False
        elif pnorm == 2:
            signal_type = 'SP'  # pcalc=0 & pnorm=2 --> load SP from raw highres_SP, normalize within time window
            pnorm = True
    elif pcalc == 1:
        if pnorm == 0:
            signal_type = 'SP_CALC'  # pcalc=1 & pnorm=0 --> calculate SP from EEG, do not normalize
            pnorm = False
        elif pnorm == 1:
            signal_type = 'SP_CALC_NORM'  # pcalc=1 & pnorm=0 --> calculate SP from EEG, normalize by recording
            pnorm = False
        elif pnorm == 2:
            signal_type = 'SP_CALC'  # pcalc=1 & pnorm=2 --> calculate SP from EEG, normalize within time window
            pnorm = True
    if ci == 'sem':
        ci = 68
    
    states = {'total':'total', 1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    istate=[1]
    ifreq = np.arange(theta_band[0]*2, theta_band[1]*2+1)   # theta freq idx in SP
    freq = np.linspace(theta_band[0], theta_band[1], len(ifreq))  # theta frequencies
    sp_ifreq = np.arange(0,61)  # indices of all frequencies up to 30Hz
    sp_freq = np.linspace(0,30, len(sp_ifreq))  # all frequencies up to 30 Hz
    
    # load data file
    if pload:
        filename = pload if isinstance(pload, str) else f'lsrSurround_{signal_type}'
        success_lsr = {}
        fail_lsr = {}
        try:
            for s in istate:
                success_lsr[s] = so.loadmat(os.path.join(ppath, f'{filename}_success_lsr_{s}.mat'))
                fail_lsr[s] = so.loadmat(os.path.join(ppath, f'{filename}_fail_lsr_{s}.mat'))
                # remove MATLAB keys so later functions can get recording list
                for mat_key in ['__header__', '__version__', '__globals__']:
                    _ = success_lsr[s].pop(mat_key)
                    _ = fail_lsr[s].pop(mat_key)
            print('\nLoading data dictionaries ...\n')
        except:
            print('\nUnable to load .mat files - calculating new theta power values ...\n')
            pload = False
    
    # collect SP data
    if not pload:
        data_lsr = get_lsr_surround(ppath, recordings, istate=istate, win=win, signal_type=signal_type,
                            recalc_highres=False, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is, 
                            nsr_seg=nsr_seg, perc_overlap=perc_overlap, null=False, post_stim=post_stim, lsr_iso=0)
        success_lsr = data_lsr[2]
        fail_lsr = data_lsr[3]
        
        if psave:
            filename = psave if isinstance(psave, str) else f'lsrSurround_{signal_type}'
            s = 1
            so.savemat(os.path.join(ppath, f'{filename}_success_lsr_{s}.mat'), success_lsr[s])
            so.savemat(os.path.join(ppath, f'{filename}_fail_lsr_{s}.mat'), fail_lsr[s])
    
    # collect mean power spectrums preceding successful & failed laser pulses
    success_spectrums = {rec:[] for rec in recordings}
    fail_spectrums = {rec:[] for rec in recordings}
        
    for rec in recordings:
        success_spectrums[rec] = [np.mean(sp, axis=1) for sp in success_lsr[1][rec]]
        fail_spectrums[rec] = [np.mean(sp, axis=1) for sp in fail_lsr[1][rec]]
    
    # get single matrix (trials x frequency) of all trials across all recordings
    success_spectrum_mx, success_labels = mx2d(success_spectrums, mouse_avg)
    fail_spectrum_mx, fail_labels = mx2d(fail_spectrums, mouse_avg)
    if 'trial' in mouse_avg:
        fail_labels += len(success_labels)
        
    if psmooth:
        success_spectrum_mx = AS.convolve_data(success_spectrum_mx, psmooth, axis='x')
        fail_spectrum_mx = AS.convolve_data(fail_spectrum_mx, psmooth, axis='x')
    
    # store data in dataframe (1 power value per frequency per trial)
    success_df = pd.DataFrame({'id' : np.repeat(success_labels, len(sp_freq)),
                               'freq' : np.tile(sp_freq, len(success_labels)),
                               'pow' : success_spectrum_mx[:,sp_ifreq].flatten(),
                               'group' : 'success',
                               'theta' : np.nan})
    fail_df = pd.DataFrame({'id' : np.repeat(fail_labels, len(sp_freq)),
                            'freq' : np.tile(sp_freq, len(fail_labels)),
                            'pow' : fail_spectrum_mx[:,sp_ifreq].flatten(),
                            'group' : 'fail',
                            'theta' : np.nan})
    
    for labels,ddf in zip([success_labels, fail_labels], [success_df, fail_df]):
        if mode == 'spectrum':
            ddf['theta'] = -1
            continue
        
        for l in labels:
            # get power of each frequency in trial
            rows = np.where((ddf.id == l) &
                            (ddf.freq >= freq[0]) & 
                            (ddf.freq <= freq[-1]))[0]
            f_pows = np.array(ddf.loc[rows, 'pow'])
            
            # calculate trial mean theta power
            if mode == 'power':
                #success_vec[trial] = f_pows.mean()
                d = f_pows.mean()
            # calculate trial mean theta frequency
            elif mode == 'mean freq':
                # normalize each theta frequency by total power in the theta range
                num = np.array([f*fp for (f,fp) in zip(freq,f_pows)]).sum()
                denom = np.sum(f_pows)
                #success_vec[trial] = round(num/denom, 3)
                d = round(num/denom, 3)
            ddf.loc[rows,'theta'] = d
    
    df = pd.concat([success_df, fail_df], axis=0, ignore_index=True)
    df = df.dropna(axis=0, subset=['theta']).reset_index(drop=True)
            
    ###   GRAPHS   ###
    
    if mode == 'spectrum':
        # plot power spectral densities preceding successful and failed laser pulses
        if ci == 'sem':
            ci = 'se'
        elif type(ci) == int:
            ci = ('ci', ci)
        plt.figure()
        sns.lineplot(x='freq', y='pow', hue='group', data=df, errorbar=ci,
                     palette={'success':'blue', 'fail':'red'})
        tmp = 'Raw' if signal_type == 'SP' and pnorm == False else 'Normalized'
        plt.title(f'{tmp} power spectrum during {np.abs(win[0])}s preceding laser')
        plt.show()

    else:
        df = df.groupby(['id','group']).mean().reset_index()

        plt.figure()
        tmp = 'Raw' if signal_type == 'SP' and pnorm == False else 'Normalized'
        if mode == 'power':
            title = f'{tmp} theta power ({theta_band[0]}Hz - {theta_band[1]}Hz) during {np.abs(win[0])}s preceding laser'
        elif mode == 'peak freq':
            title = f'Peak theta frequency during {np.abs(win[0])}s preceding laser'
        elif mode == 'mean freq':
            title = f'Mean theta frequency during {np.abs(win[0])}s preceding laser'
        plt.title(title)
        sns.boxplot(x='group', y='theta', data=df, whis=np.inf, color='lightgray', fliersize=0)
        sns.swarmplot(x='group', y='theta', data=df, size=5, edgecolor='black', linewidth=0.8, 
                      palette={'success':'blue', 'fail':'red'})
        
        # plot overlapping distributions of successful vs failed laser pulses
        plt.figure()
        if len(prange1)==2:
            sns.histplot(x='theta', hue='group', data=df, stat='probability', common_norm=False, 
                         binrange=prange1, kde=True, kde_kws={'bw_adjust':2}, fill=True, 
                         palette={'success':'blue', 'fail':'red'}, **{'linewidth':0, 'alpha':0.4})
        else:
            sns.histplot(x='theta', hue='group', data=df, stat='probability', common_norm=False, 
                         kde=True, kde_kws={'bw_adjust':2}, fill=True, palette={'success':'blue', 'fail':'red'}, 
                         **{'linewidth':0, 'alpha':0.4})
        if mode == 'power': plt.xlabel(f'{tmp} theta power') 
        elif mode == 'peak freq': plt.xlabel('Peak frequency')
        elif mode == 'mean freq': plt.xlabel('Mean theta frequency')
        plt.ylabel('Proportion of lsr trials')
        plt.title(title)
        
        success_vec = np.array(df.loc[np.where(df.group=='success')[0], 'theta'])
        fail_vec = np.array(df.loc[np.where(df.group=='fail')[0], 'theta'])
        # plot probability of laser success for each theta power/frequency bin
        success_hist, b = np.histogram(success_vec, bins=nbins, range=prange2)
        fail_hist, b = np.histogram(fail_vec, bins=nbins, range=prange2)
        total_hist = success_hist + fail_hist
        prob_hist = [i/j*100 if j>0 else 0 for i,j in zip(success_hist, total_hist)]
        x = np.linspace(b[0], b[-1], nbins)
        width = (x[-1] - x[0])/len(x)
        plt.figure()
        plt.bar(x, prob_hist, width=width, color='purple')
        if mode == 'power': plt.xlabel(f'{tmp} theta power') 
        elif mode == 'peak freq': plt.xlabel('Peak frequency')
        elif mode == 'mean freq': plt.xlabel('Mean frequency')
        plt.ylabel('Percent successful laser pulses (%)')
        if mode == 'power':
            plt.title(f'Probability of laser success vs. preceding {tmp.lower()} theta power')
        elif mode == 'peak freq':
            plt.title('Probability of laser success vs. preceding peak theta frequency')
        elif mode == 'mean freq':
            plt.title('Probability of laser success vs. preceding mean theta frequency')
        plt.show()
        
        # stats
        x = np.concatenate((np.array((success_vec)), np.array((fail_vec))), axis=0)
        y = np.concatenate((np.ones((len(success_vec),)), np.zeros((len(fail_vec),))))
        corr = scipy.stats.pointbiserialr(x,y)
        print('')
        if mode == 'power': print(f'PRECEDING REM THETA POWER CORRELATION')
        elif mode == 'peak freq': print(f'PRECEDING REM THETA PEAK FREQUENCY CORRELATION')
        elif mode == 'mean freq': print(f'PRECEDING REM MEAN THETA FREQUENCY CORRELATION')
        #print(f'rpb={np.round(corr.correlation,5)}, p-value={round(corr.pvalue,5)}')
        print(f'rpb={np.round(corr.correlation,5)}, p-value={AS.pp(corr.pvalue)[0]}')
        print('')
    
    return df



###############           STATISTICAL FUNCTIONS           ################

def bonferroni_signtest(df, alpha=0.05):
    """
    Bonferroni correction for Wilcoxon ranked-sign test
    @Params
    df - dataframe (columns = dependent samples (groups) that are compared with each other)
    alpha - significance level; corrected alpha is divided by no. of pairwise comparisons
    @Returns
    results - stats dataframe with test statistics and p-values
    """
    groups = df.columns
    n = len(groups)
    diffs = []
    s = []
    p = []
    labels = []
    ntest = 0
    # perform pairwise comparisons
    for i in range(n):
        for j in range(i+1, n):
            g1 = groups[i]
            g2 = groups[j]
            label = str(g1) + '<>' + str(g2)
            val = scipy.stats.wilcoxon(df[g1], df[g2])
            s.append(val[0])
            p.append(val[1])

            diff = df[g1].mean() - df[g2].mean()
            diffs.append(diff)
            labels.append(label)
            ntest += 1
    # collect statistical sig. result
    reject = []
    for sig in p:
        if sig < alpha / ntest:
            reject.append(True)
        else:
            reject.append(False)
    # store stats results in dataframe
    results = pd.DataFrame(index = labels, columns=['diffs', 'statisics', 'p-values', 'reject'])
    results['diffs'] = diffs
    results['statistics'] = s
    results['p-values'] = p
    results['reject'] = reject
    return results


def cohen_d(x,y):
    """
    Correct if the population S.D. is expected to be equal for the two groups
    @Params
    x - 1D np.array or list of values in group 1
    y - 1D np.array or list of values in group 2
    @Returns
    es - effect size
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    es = (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    return es


def pairT_from_df(df, cond_col, cond1, cond2, test_cols, c1_label='', c2_label='', 
                  test_col_labels=[], nan_policy='omit', print_stats=True, print_notice=[]):
    """
    Perform paired t-test on labeled data within a column of a dataframe
    @Params
    df - dataframe with info column for experimental condition labels, and
         data columns for raw values
    cond_col - name of the column containing experimental condition labels (e.g. 'Lsr')
               * If cond_col=[], test_cols is a 2-element list specifying columns between
                 which to compare all data
    cond1, cond2 - names of conditions to compare within dataframe column (e.g. 1 and 0)
    test_cols - list of column names containing data to compare between conditions (e.g. ['R', 'N', 'W'])
    c1_label, c2_label - optional strings describing cond1 and cond2 (e.g. 'Lsr ON and Lsr OFF')
    test_col_labels - optional list of strings describing the compared data columns (e.g. ['REM', 'NREM', 'Wake'])
    nan_policy - method for handling Nans in the data ('omit' = ignore,
                 'propogate' = return NaN, 'raise' = throw an error)
    print_stats - if True, print stats summary with T statistics and p-values
    print_notice - optional message to print instead of stats
    @Returns
    stats_df - summary dataframe (rows=data column labels, columns=T statistic, 
                p-value, and significance with alpha=0.05)
    """
    # perform paired t-test with 2 data columns specified by $test_cols
    if cond_col == []:
        p = scipy.stats.ttest_rel(df[test_cols[0]], df[test_cols[1]], nan_policy=nan_policy)
        sig = 'yes' if p.pvalue < 0.05 else 'no'
        if print_stats:
            print(test_cols[0] + ' vs ' + test_cols[1])
            print('T=' + str(p.statistic) + ', p=' + str(p.pvalue) + ' sig:' + sig)   
    else:
        data = []
        # create labels for data columns and experimental conditions
        C1 = c1_label if c1_label else f'{cond_col}={cond1}'
        C2 = c2_label if c2_label else f'{cond_col}={cond2}'
        col_labels = test_col_labels if len(test_col_labels)==len(test_cols) else test_cols
        # for each test column, compare between $cond1 and $cond2
        for col, col_label in zip(test_cols, col_labels):
            d1 = df[col].iloc[np.where(df[cond_col]==cond1)[0]]
            d2 = df[col].iloc[np.where(df[cond_col]==cond2)[0]]
            p = scipy.stats.ttest_rel(d1, d2, nan_policy=nan_policy)
            sig = 'yes' if p.pvalue < 0.05 else 'no'
            data.append([col_label, p.statistic, p.pvalue, sig])
        # create stats dataframe and print summary
        stats_df = pd.DataFrame(data=data, columns = ['', 'T', 'p-value', 'sig'])
        if print_stats:
            print('')
            if len(print_notice) > 0:
                print(print_notice)
            else:
                print(f'   ### STATISTICS - {C1} vs {C2} ###   ')
            print(stats_df)
        return stats_df

    
def stats_timecourse(mx, pre, post, sr, base_int, baseline_start=0, baseline_end=-1, 
                     nan_policy='omit', print_stats=True):
    """
    Determine when timecourse data significantly differs from baseline, using 
    repeated paired t-tests with Bonferroni correction
    @Params
    mx - matrix of subjects (rows) x time bins (columns)
    pre, post - time window (s) spanned by the matrix columns. Can be relative
                 to an event (e.g. -5 and +5 s) or absolute (e.g. 100 and 120 s)
    sr - time bins per second in $mx
    base_int - size of consecutive time bins (s) to compare
    baseline_start - no. of bins into timecourse to start "baseline" bin
    baseline_end - no. of bins into timecourse to end comparisons
    nan_policy - method for handling Nans in the data ('omit' = ignore,
                 'propogate' = return NaN, 'raise' = throw an error)
    print_stats - if True, print timecourse stats with T statistics and p-values
    @Returns
    df - summary dataframe (rows=time bins, columns=T statistics, raw p-values, 
          adjusted p-values, and significance with CORRECTED alpha=0.05)
    """
    if baseline_end == -1:
        baseline_end = mx.shape[1]+1
    # get no. of time points per bin, and total no. of bins
    ibin = int(np.round(base_int * sr))
    nbin = int(np.floor((baseline_end-baseline_start)/ibin))
    # eliminate NaNs in matrix
    data = []
    ctrans = []
    for row in range(0, mx.shape[0]):
        if all(np.isnan(mx[row,:])):
            continue
        else:
            ctrans.append(mx[row,:])
    ctrans = np.array((ctrans)).reshape(len(ctrans), len(ctrans[0]))
    
    # get mean baseline value for each subject
    base = np.nanmean(ctrans[: , baseline_start:baseline_start+ibin], axis=1)
    x = np.linspace(pre, post, ctrans.shape[1])
    cur_i = baseline_start
    for i in range(1,nbin):
        # get indices of current time bin
        si = baseline_start + i*ibin
        ei = baseline_start + (i+1)*ibin
        if ei >= len(x):
            ei = len(x)-1
        if si==ei or si==ei+1:
            continue
        cur_i += ibin
        if baseline_end > cur_i:
            tbin_mx = ctrans[:, si : ei]
            # paired t-test to compare subject means between baseline and current time bin
            p = scipy.stats.ttest_rel(base, np.nanmean(tbin_mx, axis=1), nan_policy=nan_policy)
            # adjust p-value by total number of comparisons (Bonferroni correction)
            p_adj = p.pvalue * (nbin-1)
            sig = 'no'
            if p_adj < 0.05:
                sig = 'yes'
            tpoint = f'{round(x[si], 1)} - {round(x[ei], 1)}'
            # save raw and adjusted stats in dataframe
            data.append([tpoint, p.statistic, p.pvalue, p_adj, sig])
    # create stats dataframe
    df = pd.DataFrame(data = data, columns = ['time', 'T', 'p-value', 'p-adj', 'sig'])
    df['T'] = np.round(np.array(df['T']), 2)
    df['p-adj'] = AS.pp(df['p-adj'])
    if print_stats:
        print(df)
    return df