#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:21:14 2020

@author: fearthekraken
"""
import os
import re
import scipy
import scipy.io as so
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from functools import reduce
import math
import pingouin as ping
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
import h5py
import pdb
import warnings
warnings.simplefilter('error', pd.errors.DtypeWarning)
# custom modules
import sleepy
import pwaves


#################            SUPPORT FUNCTIONS            #################

def pp(data, sci_sig=2, dec_sig=3):
    try:
        iter(data)
    except:
        data = [data]
    data = np.array(data)
    y = []
    for x in data:
        if isinstance(x, str):
            y.append(x)
        elif x < 0.001:
            y.append(f"{x:.{sci_sig}E}")
        else:
            y.append(f"{x:.{dec_sig}f}")
    return y


def get_snr_pwaves(ppath, name, default='NP'):
    """
    Load sampling rate of recording $ppath/$name from info file
    @Returns
    SR - sampling rate (Hz)
    """
    
    fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)
    lines = fid.readlines()
    fid.close()
    
    sr = None
    np_sr = None
    for l in lines:
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)  # find Intan SR
        if a:
            sr = float(a.group(1))
        b = re.search("^" + 'SR_NP' + ":" + "\s+(.*)", l)  # find Neuropixel SR
        if b:
            np_sr = float(b.group(1))
            
    if np_sr is None:  # if Neuropixel SR not in info file, return Intan SR
        return sr
    if sr is None:     # unlikely - every recording protocol saves Intan SR
        return np_sr
    if default == 'Intan':  # if file contains both sampling rates, return 'default' SR
        return sr
    if default == 'NP':
        return np_sr
    return sr


def load_eeg_emg(ppath, rec, dname='EEG'):
    data_path = os.path.join(ppath, rec, f'{dname}.mat')
    if os.path.exists(data_path):
        try:
            data = so.loadmat(data_path, squeeze_me=True)[dname]
        except ValueError:
            try:
                with h5py.File(data_path, 'r') as f:
                    data = np.squeeze(f[dname])
            except OSError:
                return OSError(dname + ' file must be saved in .mat or h5py format')
    else:
        return FileNotFoundError('No saved ' + dname + ' file found')
    return data


def upsample_mx(x, nbin, axis=1):
    """
    Upsample input data by duplicating each element (if $x is a vector)
    or each row/column (if $x is a 2D array) $nbin times
    @Params
    x - input data
    nbin - factor by which to duplicate
    axis - specifies dimension to upsample for 2D input data
           if 0 or 'y' - duplicate rows
           if 1 or 'x' - duplicate columns
    @Returns
    y - upsampled data
    """
    if nbin == 1:
        return x
    # get no. elements in vector to be duplicated
    if axis == 0 or axis == 'y' or x.ndim == 1:
        nelem = x.shape[0]
    elif axis == 1 or axis == 'x':
        nelem = x.shape[1]
    # upsample 1D input data
    if x.ndim == 1:
        y = np.zeros((nelem * nbin,))
        for k in range(nbin):
            y[k::nbin] = x
    # upsample 2D input data
    else:
        if axis == 0 or axis == 'y':
            y = np.zeros((nelem * nbin, x.shape[1]))
            for k in range(nbin):
                y[k::nbin, :] = x
        elif axis == 1 or axis == 'x':
            y = np.zeros((x.shape[0], nelem * nbin))
            for k in range(nbin):
                y[:, k::nbin] = x
    return y


def downsample_vec(x, nbin):
    """
    Downsample input vector by replacing $nbin consecutive bins by their mean
    @Params
    x - input vector
    bin - factor by which to downsample
    @Returns
    y - downsampled vector
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))
    for i in range(nbin) :
        idx = list(range(i, int(n_down*nbin), int(nbin)))
        x_down += x[idx]
    y = x_down / nbin
    return y


def downsample_nanvec(x, nbin):
    n_down = int(np.floor(len(x) / nbin))  # no. bins in downsampled vector
    x = x[0:n_down*nbin]                   # cut $x to evenly divide into $n_down bins
    x_down = np.zeros((n_down,))  # sum of all numerical elements in each bin
    x_num = np.zeros((n_down,))   # no. numerical elements summed in corresponding $x_down bin
    for i in range(nbin):
        idx = list(range(i, int(n_down*nbin), int(nbin)))       # $index of ith element in each bin
        x_down = np.nansum(np.vstack((x_down,x[idx])), axis=0)  # add all numerical elements to corresponding bin sum
        x_num += np.invert(np.isnan(x[idx])).astype('int')      # keep track of numerical elements added
    x_num[np.where(x_num==0)[0]] = np.nan
    y = np.divide(x_down, x_num)            # get mean of all numerical elements in each bin
    return y


def downsample_mx(x, nbin, axis=1):
    """
    Downsample input matrix by replacing $nbin consecutive rows or columns by their mean
    @Params
    x - input matrix
    nbin - factor by which to downsample
    axis - specifies dimension to downsample
           if 0 or 'y' - downsample rows
           if 1 or 'x' - downsample columns
    @Returns
    y - downsampled matrix
    """
    # downsample rows
    if axis == 0 or axis == 'y':
        n_down = int(np.floor(x.shape[0] / nbin))
        x = x[0:n_down * nbin, :]
        x_down = np.zeros((n_down, x.shape[1]))
        for i in range(nbin):
            idx = list(range(i, int(n_down * nbin), int(nbin)))
            x_down += x[idx, :]
    # downsample columns
    elif axis == 1 or axis == 'x':
        n_down = int(np.floor(x.shape[1] / nbin))
        x = x[:, 0:n_down * nbin]
        x_down = np.zeros((x.shape[0], n_down))
        for i in range(nbin):
            idx = list(range(i, int(n_down * nbin), int(nbin)))
            x_down += x[:, idx]
    y = x_down / nbin
    return y


def time_morph(X, nstates):
    """
    Set size of input data to $nstates bins
    X - input data; if 2D matrix, resample columns
    nstates - no. elements/columns in returned vector/matrix
    @Returns
    Y - resampled data
    """
    # upsample elements/columns by $nstates
    if X.ndim == 1:
        m = X.shape[0]
    else:
        m = X.shape[1]
    A = upsample_mx(X, nstates)
    
    # downsample A to desired size
    if X.ndim == 1:
        Y = downsample_vec(A, int((m * nstates) / nstates))
    else:
        Y = downsample_mx(A, int((m * nstates) / nstates))
    return Y


def smooth_data(x, sig):
    """
    Smooth data vector using Gaussian kernel
    @Params
    x - input data
    sig - standard deviation for smoothing (larger STD = more smoothing)
    @Returns
    sm_data - smoothed data
    
    filter = [f1, f2, f3, f4, f5]
    res = [ ... ... ... (i-2)*f1 + (i-1)*f2 + (i)*f3 + (i+1)*f4 + (i+2)*f5 ... ... ... ]
    """
    
    sig = float(sig)
    if sig == 0.0:
        return x
    
    # create gaussian
    #gauss = lambda x, sig : (1/(sig*np.sqrt(2.*np.pi)))*np.exp(-(x*x)/(2.*sig*sig))
    
    # create gaussian
    def gauss(x, sig):
        # get maximum of PDF function (or, probability of observing the most likely outcome)
        func_max = 1 / (sig*np.sqrt(2.*np.pi))  # larger standard deviation = smaller probability of each outcome
        
        # find no. of standard deviations between $x and a mean of zero
        zsq = -0.5 * (x/sig)**2  # large negative = element far above/below mean; small negative = element close to mean
        zsq_exp = np.exp(zsq)    # scale distance between 0 (inf. far from mean) and 1 (equal to mean)
        
        # get probability of observing value $x (0 to $func_max)
        pdf = zsq_exp * func_max
        
        return pdf
        
    bound = 1.0/10000   # 0.0001
    # random variable $L has a normal distribution with standard deviation $sig
    L = 10.
    # what is the probability of a random observation from this distribution having the value 10.0?
    p = gauss(L, sig)
    # if probability is too high, make the test value larger/further away from zero
    while (p > bound):
        L = L+10       # larger $sig = smaller value of $L = shorter filter
        p = gauss(L, sig)

    # create smoothing filter
    # for each value between -L and L, get probability of observing that value
    # min probability at tails is just under $bound, max probability at peak is $func_max
    F = [gauss(x, sig) for x in np.arange(-L, L+1.)]
    F = F / np.sum(F)
    
    # perform Gaussian smoothing on $x vector
    res = scipy.ndimage.convolve1d(np.array(x), np.array(F))
    return res
    

def smooth_data2(x, nstep):
    """
    Smooth data by replacing each of $nstep consecutive bins with their mean
    @Params
    x - input data
    nstep - no. consecutive samples to average
    @Returns
    x2 - smoothed data
    """
    x2 = [[np.mean(x[i:i+nstep])]*nstep for i in np.arange(0, len(x), nstep)]
    #x2 = list(chain.from_iterable(x2))
    #x2 = np.array((x2))
    x2 = np.concatenate(x2)
    x2 = x2[0:len(x)]
    
    return x2


def convolve_data(x, psmooth, axis=2):
    """
    Smooth data by convolving with filter defined by $psmooth
    @Params
    x - input data
    psmooth - integer or 2-element tuple describing filter for convolution
              * for 2-element $psmooth param, idx1 smooths across rows and idx2 smooths 
                 across columns
    axis - specifies filter if $psmooth is an integer
    	    if 0 or 'y' - convolve across rows
            if 1 or 'x' - convolve across columns
            if 2 or 'xy' - convolve using box filter
    @Returns
    smooth - smoothed data
    """
    if not psmooth:
        return x
    # if np.isnan(x).any():
    #     raise KeyError('ERROR: NaN(s) found in data')
    # smooth across 1D data vector
    if x.ndim == 1:
        if type(psmooth) in [int, float]:
            filt = np.ones(int(psmooth)) / np.sum(np.ones(int(psmooth)))
        elif type(psmooth) in [list, tuple] and len(psmooth)==1:
            filt = np.ones(int(psmooth[0])) / np.sum(np.ones(int(psmooth[0])))
        else:
            raise KeyError('ERROR: incorrect number of values in $psmooth parameter for 1-dimensional data')
        xsmooth = scipy.signal.convolve(x, filt, mode='same')
    # smooth 2D data matrix
    elif x.ndim == 2:
        if type(psmooth) in [int, float]:
            if axis == 0 or axis == 'y':
                filt = np.ones((int(psmooth),1))
            elif axis == 1 or axis == 'x':
                filt = np.ones((1, int(psmooth)))
            elif axis == 2 or axis == 'xy':
                filt = np.ones((int(psmooth), int(psmooth)))
        elif type(psmooth) in [list, tuple] and len(psmooth)==2:
            filt = np.ones((int(psmooth[0]), int(psmooth[1])))
        else:
            raise KeyError('ERROR: incorrect number of values in $psmooth parameter for 2-dimensional data')
        filt = filt / np.sum(filt)
        xsmooth = scipy.signal.convolve2d(x, filt, boundary='symm', mode='same')
    else:
        raise KeyError('ERROR: inputted data must be a 1 or 2-dimensional array')
    
    return xsmooth


def exponential_smooth(x, exponent=2, min_weight=0.5, max_weight=1, avg_mode=0):
    """
    Smooth data vector using the weighted average of each point with its two neighbors. 
    The weight of each point is inversely proportional to its exponentially scaled 
    deviation from the surrounding point(s).
    @Params
    data - input vector
    exponent - power by which to raise the deviation of each data point from its
               neighbor(s). Bigger numbers = steeper exponential curve for scaling
               deviations (i.e. proportionally greater impact on outlier values)
    min_weight, max_weight - minimum and maximum values (0-1) used to weight data 
                             points in weighted average. Smaller numbers = larger
                             adjustments for each smoothed data element
    avg_mode - calculate deviation of each data point compared to the preceding
               element (-1), the following element (1), or the mean of both (0)
    @Returns
    x - smoothed data
    """
    # clean data inputs
    if type(x) != np.ndarray:
        x = np.array(x)
    if len(x) < 3:
        return x
    # calculate deviations between each data point and its neighbor(s)
    if avg_mode == -1:   # element minus preceding element
        difs = np.abs(x[1:-1] - x[0:-2])
    elif avg_mode == 1:  # element minus following element
        difs = np.abs(x[1:-1] - x[2:])
    elif avg_mode == 0:  # element minus mean of 2 surrounding elements
        difs = np.abs(x[1:-1] - np.mean([x[0:-2], x[2:]], axis=0))
        
    # exponentiate deviations and rescale between 0 and 1
    exp_devs = np.power(difs, exponent)
    weights = np.interp(exp_devs, (exp_devs.min(), exp_devs.max()), (max_weight, min_weight))
    # average each data point with its 2 neighbors, weighting inversely proportional to variance
    wt_avg = np.average([x[1:-1], x[0:-2], x[2:]], axis=0, 
                        weights=[weights, (1-weights)/2, (1-weights)/2])
    x[1:-1] = wt_avg
    
    return x


def sort_df(df, column, sequence, id_sort='mouse'):
    """
    Sort dataframe rows in order of $sequence
    @Params
    df - dataframe to be sorted
    column - name of the column containing items in $sequence
    sequence - ordered list of items to sort by
    mouse_sort - optionally sort dataframe by 'mouse' or 'recording' as well
    @Returns
    sorted_df - dataframe with rows in order specified by $sequence
    """
    # create temporary ranking column
    sorterIndex = dict(zip(sequence, range(len(sequence))))
    df['tmp_rank'] = df[column].map(sorterIndex)
    if id_sort in df.columns:
        # sort by mouse/recording name and $sequence
        sorted_df = df.sort_values([id_sort,'tmp_rank'], ascending=True)
    else:
        # sort by $sequence only
        sorted_df = df.sort_values('tmp_rank', ascending=True)
    # remove ranking column from sorted dataframe  
    sorted_df = sorted_df.iloc[:,0:-1]
    sorted_df.reset_index(drop=True, inplace=True)
    return sorted_df


def fit_dff(a465, a405, sr, nskip=5, wcut=2, wcut405=0, perc=0, shift_only=False):
    """
    Calculate optimal linear fit of isobestic signal to calcium signal. Output
    is DF/F signal; difference in fluorescence from baseline divided by baseline
    @Params
    a465 - calcium signal
    a405 - isobestic signal
    sr - sampling rate (Hz)
    nskip - ignore the first $nskip seconds when fitting signals
    wcut - lowpass filter (Hz) for calcium signal
    wcut405 - lowpass filter (Hz) for isobestic signal
    perc - if > 0, fit using only lower Xth percentile of DF/F signal
    shift_only - if True, only shift (do not scale) 405 signal to fit 465 signal
    @Returns
    dff - fitted DF/F signal
    """
    # low-pass filter 405 and 465 signals
    if wcut405 == 0 or perc > 0:
        wcut405 = wcut
    w0 = wcut    / (0.5*sr)
    w1 = wcut405 / (0.5*sr)
    if w0>0:
        a465 = sleepy.my_lpfilter(a465, w0, N=4)
        a405 = sleepy.my_lpfilter(a405, w1, N=4)
    nstart = int(np.round(nskip*sr))  # discard initial $nskip seconds
    # shift and/or scale 405 signal
    if shift_only and perc == 0:
        X = np.vstack([np.ones(len(a405))]).T
    else:
        X = np.vstack([a405, np.ones(len(a405))]).T
    
    # least squares regression to fit 405 signal to 465 signal
    p = np.linalg.lstsq(X[nstart:,:], a465[nstart:], rcond=-1)[0]
    a465_fit = np.dot(X, p)
    # calculate DF/F (difference from baseline divided by the baseline)
    dff = np.divide((a465 - a465_fit), a465_fit)
    
    if perc > 0:
        # calculate lowest Xth percentile of DF/F signal
        pc = np.percentile(dff[nstart:], perc)
        idx = np.where(dff<pc)[0]
        idx = idx[np.where(idx>nstart)[0]]
        # fit baseline for points where calcium signal < Xth percentile of DF/F
        X2 = np.vstack([a405[idx], np.ones(len(idx))]).T
        p = np.linalg.lstsq(X2, a465[idx])[0]
        a465_fit = a405*p[0] + p[1]
        dff = np.divide((a465 - a465_fit), a465_fit)
        
    return dff


def calculate_dff(ppath, name, nskip=5, wcut=2, wcut405=0, perc=0, shift_only=False):
    """
    Calculate DF/F signal for fiber photometry recording and save as DFF.mat
    @Params
    ppath - base folder
    name - recording folder
    nskip - ignore the first $nskip seconds when fitting DF/F signal
    wcut - lowpass filter for calcium (465 nm) signal
    wcut405 - lowpass filter for isobestic (405 nm) signal
    perc - if > 0, use only the lower Xth percentile of DF/F signal; avoids 
           largely negative values
    shift_only - if True, only shift (do not scale) 405 signal to fit 465 signal
    @Returns
    None
    """
    # load sampling rate
    sr = sleepy.get_snr(ppath, name)
    dt = 1.0/sr
    # load 405 (isobestic) and 465 (calcium) signals
    D = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)
    a465 = D['465']
    a405 = D['405']
    n = len(a465)
    
    # calculate fitted DF/F signal
    dff = fit_dff(a465, a405, sr=sr, nskip=nskip, wcut=wcut, wcut405=wcut405,
                  perc=perc, shift_only=shift_only)

    # downsample DF/F signal to 2.5 s bins
    nbins = int(np.round(sr)*2.5)
    k = int(np.ceil((1.0 * n) / nbins))
    dff2 = np.zeros((k*nbins,))
    dff2[:len(dff)]=dff
    dff = dff2
    dffd = downsample_vec(dff, nbins)
    t = np.linspace(0.0, n*dt, n+1)
    
    # save signals in .mat file
    so.savemat(os.path.join(ppath, name, 'DFF.mat'), {'t':t, '405':a405, '465':a465, 
                                                      'dff':dff, 'dffd':dffd})


def load_recordings(ppath, trace_file, dose, pwave_channel=False):
    """
    Load recording names, drug doses, and P-wave channels from .txt file
    @Params
    ppath - path to $trace_file
    trace_file - .txt file with recording information
    dose - if True, load drug dose info (e.g. '0' or '0.25' for DREADD experiments)
    pwave_channel - if True, load P-wave detection channel ('X' for mice without clear P-waves)
    @Returns
    ctr_list - list of control recordings
    * if $dose=True:  exp_dict - dictionary of experimental recordings (keys=doses, values=list of recording names)
    * if $dose=False: exp_list - list of experimental recordings
    """
    
    # read in $trace_file
    rfile = os.path.join(ppath, trace_file)
    f = open(rfile, newline=None)
    lines = f.readlines()
    f.close()
    
    # list of control recordings
    ctr_list = []
    # if file includes drug dose info, store experimental recordings in dictionary
    if not dose:
        exp_list = []
    # if no dose info, store exp recordings in list
    else:
        exp_dict = {}
    
    for l in lines :
        # if line starts with $ or #, skip it
        if re.search('^\s+$', l) :
            continue
        if re.search('^\s*#', l) :
            continue
        # a is any line that doesn't start with $ or #
        a = re.split('\s+', l)
        
        # for control recordings
        if re.search('C', a[0]) :
            # if file includes P-wave channel info, collect recording name and P-wave channel
            if pwave_channel:
                if a[-2] != 'X':
                    ctr_list.append(a[1])
                #ctr_list.append([a[1], a[-2]])
            # if no P-wave channel info, collect recording name
            else:
                ctr_list.append(a[1])
        
        # for experimental recordings
        elif re.search('E', a[0]) :
            #  if no dose info, collect exp recordings in list
            if not dose:
                if pwave_channel:
                    if a[-2] != 'X':
                        exp_list.append(a[1])
                    #exp_list.append([a[1], a[-2]])
                else:
                    exp_list.append(a[1])
            # if file has dose info, collect exp recordings in dictionary
            # (keys=doses, values=lists of recording names)
            else:
                if a[2] in exp_dict:
                    if pwave_channel:
                        if a[-2] != 'X':
                            exp_dict[a[2]].append(a[1])
                        #exp_dict[a[2]].append([a[1], a[-2]])
                    else:
                        exp_dict[a[2]].append(a[1])
                else:
                    if pwave_channel:
                        if a[-2] != 'X':
                            exp_dict[a[2]] = [a[1]]
                    else:
                        exp_dict[a[2]] = [a[1]]
    # returs 1 list and 1 dict if file has drug dose info, or 2 lists if not
    if dose:
        return ctr_list, exp_dict
    else:
        return ctr_list, exp_list
    
    
def load_surround_files(ppath, pload, istate, plaser, null, signal_type=''):
    """
    Load raw data dictionaries from saved .mat files
    @Params
    ppath - folder with .mat files
    pload - base filename
    istate - brain state(s) to load data files
    plaser - if True, load files for laser-triggered P-waves, spontaneous P-waves,
                      successful laser pulses, and failed laser pulses
             if False, load file for all P-waves
    null - if True, load file for randomized control points
    signal_type - string indicating type of data loaded (e.g. SP, LFP), completes 
                  default filename
    @Returns
    *if plaser --> lsr_pwaves  - dictionaries with brain states as keys, and sub-dictionaries as values
                                 Sub-dictionaries have mouse recordings as keys, with lists of 2D or 3D signals as values
                                 Signals represent the time window surrounding each laser-triggered P-wave
                   spon_pwaves - signals surrounding each spontaneous P-wave
                   success_lsr - signals surrounding each successful laser pulse
                   fail_lsr    - signals surrounding each failed laser pulse
                   null_pts    - signals surrounding each random control point
                   data_shape  - tuple with shape of the data from one trial 
    
    *if not plaser --> p_signal    - signals surrounding each P-wave
                       null_signal - signals surrounding each random control point
                       data_shape  - tuple with shape of the data from one trial 
    """
    if plaser:
        filename = pload if isinstance(pload, str) else f'lsrSurround_{signal_type}'
        lsr_pwaves = {}
        spon_pwaves = {}
        success_lsr = {}
        fail_lsr = {}
        null_pts = {}
        try:
            for s in istate:
                # load .mat files with stored data dictionaries
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
            data_shape = so.loadmat(os.path.join(ppath, f'{filename}_data_shape.mat'))['data_shape'][0]
            print('\nLoading data dictionaries ...\n')
            return lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape
        except:
            print('\nUnable to load .mat files - collecting new data ...\n')
            return []
        
    elif not plaser:
        filename = pload if isinstance(pload, str) else f'Surround_{signal_type}'
        p_signal = {}
        null_signal = {}
        try:
            for s in istate:
                # load .mat files with stored data
                p_signal[s] = so.loadmat(os.path.join(ppath, f'{filename}_pwaves_{s}.mat'))
                if null:
                    null_signal[s] = so.loadmat(os.path.join(ppath, f'{filename}_null_{s}.mat'))
                # remove MATLAB keys so later functions can get recording list
                for mat_key in ['__header__', '__version__', '__globals__']:
                    _ = p_signal[s].pop(mat_key)
                    if null:
                        _ = null_signal[s].pop(mat_key)
            data_shape = so.loadmat(os.path.join(ppath, f'{filename}_data_shape.mat'))['data_shape'][0]
            print('\nLoading data dictionaries ...\n')
            return p_signal, null_signal, data_shape
        except:
            print('\nUnable to load .mat files - calculating new spectrograms ...\n')
            return []


def emg_amplitude(ppath, rec, emg_source='raw', recalc_amp=False, nsr_seg=2, 
                  perc_overlap=0.75, recalc_highres=False, r_mu=[10,500], 
                  w0=-1, w1=-1, dn=1, smooth=0, exclude_noise=False, pemg2=False):
    """
    Load or calculate EMG amplitude for a recording
    @Params
    ppath - base folder
    rec - name of recording
    emg_source - use downsampled raw EMG ('raw') or summed EMG spectrogram ('msp')
    recalc_amp - if True, recalculate EMG amplitude using given params
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for mSP calculation
    recalc_highres - if True, recalculate mSP using $nsr_seg and $perc_overlap params
    r_mu - [min,max] frequencies summed to get EMG amplitude from mSP
    w0, w1 - min, max frequencies for raw EMG filtering
             w0=-1 and w1=-1, no filtering; w0=-1, low-pass filter; w1=-1, high-pass filter 
    dn - no. samples per bin for downsampling raw EMG
    smooth - smoothing factor for raw EMG
    exclude_noise - if True, noise indices in raw EMG signal will be loaded; 
                    corresponding indices in calculated EMG amplitude vector will be NaNs
    pemg2 - if True, calculate EMG amplitude from EMG channel 2
    @Returns
    EMG_amp - EMG amplitude vector
    mnbin - no. Intan samples per EMG amp bin
    mdt - no. seconds per EMG amp bin
    """
    # get name of EMG amp file and raw signal in recording folder
    fname = 'emg2' if pemg2 else 'emg'
    
    # try loading EMG amp from file
    fpath = os.path.join(ppath, rec, f'{fname}_amp_{rec}.mat')
    if not recalc_amp:
        if os.path.exists(fpath):
            EA = so.loadmat(fpath)
            EMG_amp = np.array(EA['EMG_amp'][0])
            mnbin = float(EA['mnbin'][0])
            mdt = float(EA['mdt'][0])
        else:
            recalc_amp = True
    
    if recalc_amp:
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        # calculate EMG amplitude from raw signal
        if emg_source == 'raw':
            # load EMG signal
            if pemg2 and os.path.exists(os.path.join(ppath, rec, 'EMG2.mat')):
                EMG = so.loadmat(os.path.join(ppath, rec, 'EMG2.mat'), squeeze_me=True)['EMG2']
            else:
                EMG = so.loadmat(os.path.join(ppath, rec, 'EMG.mat'), squeeze_me=True)['EMG']
            # filter EMG
            if w0 != -1 or w1 != -1:
                if w1 == -1:    # highpass
                    EMGfilt = np.abs(sleepy.my_hpfilter(EMG, w0))
                elif w0 == -1:  # lowpass
                    EMGfilt = np.abs(sleepy.my_lpfilter(EMG, w1))
                else:           # bandpass
                    EMGfilt = np.abs(sleepy.my_bpfilter(EMG, w0, w1))
            else:
                EMGfilt = np.abs(np.array(EMG))
            # replace noise values with NaNs
            if exclude_noise:
                npath = os.path.join(ppath, rec, 'p_idx.mat')
                try:
                    ni = np.array(so.loadmat(npath, squeeze_me=True)['emg_noise_idx']).astype('int')
                except:
                    print('###   WARNING: Unable to exclude noise - no saved noise indices found   ###')
                    ni = []
                EMGfilt[ni] = np.nan
                #EMGfilt[noise_idx] = np.nan
            # smooth EMG
            if smooth > 0:
                EMGfilt = convolve_data(EMGfilt, smooth, axis='x')
                #EMGdn = sleepy.smooth_data(EMGdn, smooth)
            # downsample EMG
            if dn > 1:
                EMG_amp = sleepy.downsample_vec(EMGfilt, int(dn))
            else:
                EMG_amp = np.array(EMGfilt)
            # highpass filter again (?)
            #EMG_amp = np.abs(sleepy.my_hpfilter(EMG_amp, 0.1))
            # save no. points per bin and no. seconds per bin
            mnbin = float(dn)
            mdt = float(mnbin/sr)
        elif emg_source == 'msp':
            # calculate EMG spectrogram
            MSP = highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                      recalc_highres=recalc_highres, mode='EMG', 
                                      exclude_noise=exclude_noise, pemg2=pemg2)
            if len(MSP) == 0:
                return
            mSP, mfreq, mt, mnbin, mdt = MSP
            i_mu = np.where((mfreq >= r_mu[0]) & (mfreq <= r_mu[1]))[0]
            EMG_amp = np.sqrt(mSP[i_mu, :].sum(axis=0) * (mfreq[1] - mfreq[0]))
        inoise = np.nonzero(np.isnan(EMG_amp))
        ireal = np.setdiff1d(range(len(EMG_amp)), inoise)
        sig = np.abs(sleepy.my_hpfilter(EMG_amp[ireal], 0.1))
        EMG_amp[ireal] = sig
        # save EMG amplitude
        so.savemat(fpath, {'EMG_amp' : EMG_amp, 
                           'mnbin'   : mnbin, 
                           'mdt'     : mdt})
    return EMG_amp, mnbin, mdt
        

def noise_EEGspectrogram(ppath, rec, noise_idx=[], fres=0.5, recalc_sp=False, peeg2=False):
    """
    Load or calculate standard resolution spectrogram from noise-excluded EEG signal
    """
    fname,dname = ['SP2','EEG2'] if peeg2 else ['SP','EEG']
    # try loading mSP from file
    SP_path = os.path.join(ppath, rec, f'{fname.lower()}_nannoise_{rec}.mat')
    if not recalc_sp:
        if os.path.exists(SP_path):
            SPEC = so.loadmat(SP_path)
            SP = SPEC[fname]
            freq = np.array(SPEC['freq'][0])
            t = np.array(SPEC['t'][0])
            nbin = float(SPEC['nbin'][0])
            dt = float(SPEC['dt'][0])
            noise_idx = np.array(SPEC['noise_idx'][0])
        else:
            print(f'###   ERROR: No noise-excluded spectrogram found. Calculating new spectrogram ...')
            recalc_sp = True
    if recalc_sp:
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        swin = round(sr)*5
        fft_win = round(swin/5) # approximate number of data points per second
        if (fres == 1.0) or (fres == 1):
            fft_win = int(fft_win)
        elif fres == 0.5:
            fft_win = 2*int(fft_win)
        else:
            print("Resolution %f not allowed; please use either 1 or 0.5" % fres)
        # load EEG
        EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, dname + '.mat'))[dname])
        noise_idx = np.array(noise_idx).astype('int')
        # convert inputted noise indices to NaNs
        if len(noise_idx) > 0:
            EEG[noise_idx] = np.nan
        # calculate spectrogram
        SP, freq, t = sleepy.spectral_density(EEG, int(swin), int(fft_win), 1/sr)
        dt = t[1] - t[0]
        nbin = dt * sr
        so.savemat(SP_path, {fname       : SP, 
                             'freq'      : freq, 
                             't'         : t, 
                             'dt'        : dt, 
                             'nbin'      : nbin,
                             'noise_idx' : noise_idx})
    return SP, freq, t, nbin, dt, noise_idx
        
    
def highres_spectrogram(ppath, rec, nsr_seg=2, perc_overlap=0.95, recalc_highres=False, 
                        mode='EEG', exclude_noise=False, peeg2=False, pemg2=False, match_params=False):
    """
    Load or calculate high-resolution spectrogram for a recording
    @Params
    ppath - base folder
    rec - name of recording
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    recalc_highres - if True, recalculate high-resolution spectrogram from raw signal, 
                              using $nsr_seg and $perc_overlap params
                      if False, load existing spectrogram from saved file
    mode - use 'EEG' or 'EMG' signal for calculating spectrogram
    exclude_noise - if True, load EEG or EMG noise indices from 'p_idx.mat' file and
                     replace those values with NaNs in raw signal
    peeg2, pemg2 - get SP for prefrontal EEG, EMG channel 2
    match_params - if True, verify that FFT params of loaded SP match input arguments 
    @Returns
    SP - loaded or calculated high-res spectrogram
    freq - list of spectrogram frequencies, corresponding to SP rows
    t - list of spectrogram time bins, corresponding to SP columns
    dt - no. seconds per SP time bin
    nbin - no. samples per SP time bin
    """
    # get name of SP file and raw signal in recording folder
    if mode == 'EEG':
        fname,dname = ('SP2','EEG2') if peeg2 else ('SP','EEG')
    elif mode == 'EMG':
        fname,dname = ('mSP2','EMG2') if pemg2 else ('mSP','EMG')
        
    # try loading mSP from file
    SP_path = os.path.join(ppath, rec, f'{fname.lower()}_highres_{rec}.mat')
    if not recalc_highres:
        if os.path.exists(SP_path):
            SPEC = so.loadmat(SP_path)
            # if $nsr_seg and $perc_overlap arguments don't match loaded params, recalculate SP
            if match_params:
                if 'nsr_seg' in SPEC.keys() and 'perc_overlap' in SPEC.keys():
                    ns = float(SPEC['nsr_seg'][0])
                    po = float(SPEC['perc_overlap'][0])
                
                    if ns != nsr_seg or po != perc_overlap:
                        print('###   WARNING: Loaded FFT params for spectrogram calculation do not match current params. Recalculating spectrogram ...')
                        recalc_highres = True
                else:
                    print('###   WARNING: No FFT parameter values found. Recalculating spectrogram ...')
                    recalc_highres = True
            if not recalc_highres:
                SP = SPEC[fname]
                freq = np.array(SPEC['freq'][0])
                t = np.array(SPEC['t'][0])
                nbin = float(SPEC['nbin'][0])
                dt = float(SPEC['dt'][0])
                try:
                    ns = SPEC['nsr_seg'][0]
                    po = SPEC['perc_overlap'][0]
                except KeyError:
                    print('###   WARNING: No saved FFT params found in spectrogram file.')
                    ns = None
                    po = None
        else:
            print(f'###   ERROR: No saved {dname} spectrogram found. Calculating new spectrogram ...')
            recalc_highres = True
            
    # calculate high-resolution SP using $nsr_seg and $perc_overlap params
    if recalc_highres:
        
        sr = sleepy.get_snr(ppath, rec)
        
        data = load_eeg_emg(ppath, rec, dname)
        
        # if exclude_noise:
        #     npath = os.path.join(ppath, rec, 'p_idx.mat')
        #     if os.path.isfile(npath):
        #         nfile = so.loadmat(npath, squeeze_me=True)
        #         k = f'{dname.lower()}_noise_idx'
        #         if k in nfile.keys():
        #             print('###   WARNING: Spectrogram will include NaNs   ###')
        #             ni = np.array(nfile[k]).astype('int')
        #             data[ni] = np.nan
        #         else:
        #             print('###   WARNING: Unable to exclude noise - no saved noise indices found in p_idx.mat   ###')
        #     else:
        #         print('###   WARNING: Unable to exclude noise - no p_idx.mat file found   ###')
            
        # calculate spectrogram
        SPEC = scipy.signal.spectrogram(data, fs=sr, window='hanning', 
                                        nperseg=int(nsr_seg*sr), 
                                        noverlap=int(nsr_seg*sr*perc_overlap))
        freq, t, SP = SPEC[:]
        nbin = len(data) / len(t)
        dt = (1.0 / sr) * nbin
        # save high-res SP
        so.savemat(SP_path, {fname   : SP, 
                             'freq'  : freq, 
                             't'     : t, 
                             'dt'    : dt, 
                             'nbin'  : nbin,
                             'nsr_seg'      : nsr_seg,
                             'perc_overlap' : perc_overlap})
    return SP, freq, t, nbin, dt


def adjust_brainstate(M, dt, ma_thr=20, ma_state=3, flatten_is=False, keep_MA=[1,4,5], noise_state=2):
    """
    Handle microarousals and transition states in brainstate annotation
    @Params
    M - brain state annotation
    dt - s per time bin in M
    ma_thr - microarousal threshold
    ma_state - brain state to assign microarousals (2=wake, 3=NREM, 6=separate MA state)
    flatten_is - specifies handling of transition states, manually annotated
                     as 4 for successful transitions and 5 for failed transitions
                     if False - no change in annotation
                     if integer - assign all transitions to specified brain state (3=NREM, 4=general "transition state")
    keep_MA - microarousals directly following any brain state in $keep_MA are exempt from
              assignment to $ma_state; do not change manual annotation
    noise_state - brain state to assign manually annotated EEG/LFP noise
    @Returns
    M - adjusted brain state annotation
    """
    # assign EEG noise (X) to noise state
    M[np.where(M==0)[0]] = noise_state
    # handle microarousals
    ma_seq = sleepy.get_sequences(np.where(M == 2.0)[0])
    for s in ma_seq:
        if 0 < len(s) < ma_thr/dt:
            if M[s[0]-1] not in keep_MA:
                M[s] = ma_state
    # handle transition states
    if flatten_is:
        M[np.where((M==4) | (M==5))[0]] = flatten_is
    return M


def adjust_spectrogram(SP, pnorm, psmooth, freq=[], fmax=False):
    """
    Normalize and smooth spectrogram
    @Params
    SP - input spectrogram
    pnorm - if True, normalize each frequency in SP its mean power
    psmooth - 2-element tuple describing filter to convolve with SP
               * idx1 smooths across rows/frequencies, idx2 smooths across columns/time
    freq - optional list of SP frequencies, corresponding to rows in SP
    fmax - optional cutoff indicating the maximum frequency in adjusted SP
    @Returns
    SP - adjusted spectrogram
    """
    if psmooth:
        if psmooth == True:  # default box filter
            filt = np.ones((3,3))
        elif isinstance(psmooth, int):  # integer input creates box filter with area $psmooth^2
            filt = np.ones((psmooth, psmooth))
        elif type(psmooth) in [tuple, list] and len(psmooth) == 2:
            filt = np.ones((psmooth[0], psmooth[1]))
        # smooth SP
        filt = filt / np.sum(filt)
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')
        
    # normalize SP
    if pnorm:
        SP_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.repeat([SP_mean], SP.shape[1], axis=0).T)
    
    # cut off SP rows/frequencies above $fmax
    if len(freq) > 0:
        if fmax:
            ifreq = np.where(freq <= fmax)[0]
            if len(ifreq) < SP.shape[0]:
                SP = SP[ifreq, :]
    return SP


def detect_emg_twitches(ppath, name, thres, thres_mode=1, min_dur=30, highres=True, 
                        nsr_seg=2, perc_overlap=0.75, recalc_highres=False):
    """
    Detect phasic muscle twitches during REM sleep
    @Params
    ppath - base folder
    name - recording name
    thres - threshold for detecting muscle twitches (mean+thres*std)
    thres_mode - if 0, set threshold for entire recording
                 if 1, set individual thresholds for each REM sleep episode
    min_dur - minimum duration (s) of REM sleep episodes
    highres - if True, use high-resolution EMG spectrogram to calculate amplitude
              if False, use standard mSP resolution (2.5 s per bin)
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for mSP calculation
    recalc_highres - if True, recalculate mSP from raw EMG; if False, load saved mSP
    @Returns
    twitch_idx - indices of phasic EMG spikes
    mnbin - number of samples per bin in EMG amplitude vector, for binning conversions
            (e.g. round(idx/(2500/mbin)) gives the corresponding index in M)
    """
    # load sampling rate
    sr = sleepy.get_snr(ppath, name)
    nbin = int(np.round(sr) * 2.5)
    dt = (1.0 / sr) * nbin
    EMG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
    
    # load and adjust brain state annotation
    M = sleepy.load_stateidx(ppath, name)[0]
    M = adjust_brainstate(M, dt=2.5, ma_thr=20, ma_state=3, flatten_is=4)
    # get qualifying REM sequences, cut off last bin to eliminate muscle twitch from waking up
    remseq = sleepy.get_sequences(np.where(M==1)[0])
    remseq =  [rs[0:-1] for rs in remseq if len(rs)*dt >= min_dur]
    
    if highres:
        # load/calculate high-res EMG spectrogram
        mSP, mfreq, mt, mnbin, mdt = highres_spectrogram(ppath, name, nsr_seg=nsr_seg, 
                                                               perc_overlap=perc_overlap, 
                                                               recalc_highres=recalc_highres,
                                                               mode='EMG')
        mnbin = float(round(mnbin))
        if not recalc_highres:
            print('Detecting EMG twitches for ' + name + ' ...')
    else:
        # load standard EMG spectrogram
        SPEMG = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name))
        mSP = SPEMG['mSP']
        mfreq = SPEMG['freq'][0]
        mnbin=int(np.round(sr) * 2.5); mdt=(1.0 / sr) * mnbin
        print('Detecting EMG twitches for ' + name + ' ...')
    
    # calculate EMG amplitude
    EMG_amp = get_emg_amp(mSP, mfreq)
    
    # get indices of REM sequences in EMG amp vector
    EMG_remseq = [np.arange(int(round(rs[0]*(nbin/mnbin))), int(round(rs[-1]*(nbin/mnbin)+(nbin/mnbin)))) for rs in remseq]
    if EMG_remseq[-1][-1] == len(EMG_amp):
        EMG_remseq[-1] = EMG_remseq[-1][0:-1]
    
    # threshold all REM bins
    if thres_mode==0:
        ridx =  np.concatenate(EMG_remseq)
        th = np.nanmean(EMG_amp[ridx]) + thres*np.nanstd(EMG_amp[ridx])
        idx = pwaves.spike_threshold(EMG_amp[ridx], th, sign=-1)
        twitch_idx = ridx[idx]
        
    # separately threshold each REM period to look for spikes
    elif thres_mode==1:
        twitch_idx = np.array(())
        for seq_idx in EMG_remseq:
            try:
                th = np.nanmean(EMG_amp[seq_idx]) + thres*np.nanstd(EMG_amp[seq_idx])
            except:
                pdb.set_trace()
            idx = pwaves.spike_threshold(EMG_amp[seq_idx], th, sign=-1)
            twitch_idx = np.concatenate([twitch_idx, seq_idx[idx]])
        twitch_idx = twitch_idx.astype(int)
    
    # get twitch train (1's and 0's)
    twitch_train = np.zeros(len(EMG_amp))
    twitch_train[twitch_idx] = 1
    # upsample twitch train to Intan time
    twitch_train_up = upsample_mx(twitch_train, int(round(nbin/mnbin)))
    twitch_train_up = np.concatenate((twitch_train_up, np.zeros(len(EMG)-len(twitch_train_up))))
    
    # get EMG twitches/min for each bin in M time resolution
    twitch_freq_dn = downsample_vec(twitch_train, int(round(nbin/mnbin)))
    if len(twitch_freq_dn) == len(M)-1:
        twitch_freq_dn = np.concatenate((twitch_freq_dn, np.array((0.0,))))
    twitch_idx_dn = np.where(twitch_freq_dn != 0)[0]
    twitch_freq_dn[twitch_idx_dn] = twitch_freq_dn[twitch_idx_dn] / dt * 60
    
    # save twitch vectors in .mat file
    so.savemat(os.path.join(ppath, name, 'emg_twitches.mat'), {'twitch_train':twitch_train,
                                                               'twitch_train_up':twitch_train_up,
                                                               'twitch_freq':twitch_freq_dn})
    return twitch_idx, mnbin


###############            DATA ANALYSIS FUNCTIONS            ###############

def dff_activity(ppath, recordings, istate, tstart=10, tend=-1, pzscore=0, 
                 ma_thr=20, ma_state=3, flatten_is=4, mouse_avg='mouse', 
                 use405=False, pplot=True, print_stats=True):
    """
    Plot average DF/F signal in each brain state
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    tstart, tend - time (s) into recording to start and stop collecting data
    pzscore - use raw DF/F signal (0) or z-score signal across the recording (1)
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse' or 'recording'
    use405 - if True, analyze 'baseline' 405 signal instead of fluorescence signal
    pplot - if True, show plot
    print_stats - if True, show results of repeated measures ANOVA
    @Returns
    df - dataframe with avg DF/F activity in each brain state for each mouse
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if type(istate) != list:
        istate = [istate]
    
    df = pd.DataFrame(columns=['mouse','recording','state','dff'])
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        print('Getting data for ' + rec + ' ... ')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
        
        # define start and end points of analysis
        istart = int(np.round(tstart/dt))
        if tend == -1:
            iend = M.shape[0]
        else:
            iend = int(np.round(tend/dt))
        M = M[istart:iend]
        
        # calculate DF/F signal using high cutoff frequency for 465 signal
        # and very low cutoff frequency for 405 signal
        if use405:
            # load artifact control 405 signal
            a405 = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['405']
            isobestic = sleepy.my_lpfilter(a405, 2/(0.5*sr), N=4)
            k = int(np.ceil((1.0 * len(isobestic)) / nbin))
            tmp = np.zeros((k*nbin,))
            tmp[:len(isobestic)] = isobestic
            isobestic = tmp
            dff = downsample_vec(isobestic, nbin)
        else:
            # load DF/F signal
            calculate_dff(ppath, rec, wcut=10, wcut405=2, shift_only=False)
            dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd']
        dff = dff[istart:iend]
        if pzscore:
            dff = (dff-dff.mean()) / dff.std()
        else:
            dff *= 100.0
        
        # get DF/F signal in each brain state
        for i,s in enumerate(istate):
            sidx = np.where(M==s)[0]
            sdf = pd.DataFrame({'mouse':idf,
                                'recording':rec,
                                'state':states[s],
                                'dff':dff[sidx]})
            df = pd.concat([df,sdf], axis=0, ignore_index=True)
    state_labels = [states[s] for s in istate]
    if mouse_avg in ['mouse','recording']:
        df = df.groupby([mouse_avg,'state']).mean().reset_index()
        df = sort_df(df, column='state', sequence=state_labels, id_sort=mouse_avg)
        
    if pplot:
        # plot signal in each state
        plt.figure()
        ax = plt.gca()
        pal = {'REM':'cyan', 'Wake':'darkviolet', 'NREM':'darkgray', 'IS':'darkblue', 
               'IS-R':'navy', 'IS-W':'red', 'MA':'magenta'}
        
        sns.barplot(data=df, x='state', y='dff', errorbar='se', palette=pal, ax=ax)
        if mouse_avg in ['mouse','recording']:
            lines = sns.lineplot(data=df, x='state', y='dff', hue=mouse_avg, 
                                 errorbar=None, markersize=0, legend=False, ax=ax)
            _ = [l.set_color('black') for l in lines.get_lines()]
        sns.despine()
        ax.set_xlabel('')
        ylab = '$\Delta$ F/F (z-scored)' if pzscore else '$\Delta$ F/F (%)'
        ax.set_ylabel(ylab)
       
    if mouse_avg in ['mouse','recording'] and print_stats:
        # one-way repeated measures ANOVA
        res = ping.rm_anova(data=df, dv='dff', within='state', subject=mouse_avg)
        ping.print_table(res)
        if float(res['p-GG-corr']) < 0.05:
            res_tt = ping.pairwise_tests(data=df, dv='dff', within='state', 
                                         subject=mouse_avg, padjust='holm')
            ping.print_table(res_tt)
    return df
    

def laser_brainstate(ppath, recordings, pre, post, tstart=0, tend=-1, ma_thr=20, ma_state=3, 
                     flatten_is=4, single_mode=False, sf=0, cond=0, edge=0, ci='sem',
                     offset=0, pplot=True, ylim=[]):
    """
    Calculate laser-triggered probability of REM, Wake, NREM, and IS
    @Params
    ppath - base folder
    recordings - list of recordings
    pre, post - time window (s) before and after laser onset
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    single_mode - if True, plot individual mice
    sf - smoothing factor for vectors of brain state percentages
    cond - if 0, plot all laser trials
           if integer, only plot laser trials where mouse was in brain state $cond 
                       during onset of the laser
    edge - buffer time (s) added to edges of [-pre,post] window, prevents filtering artifacts
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                              integer between 0 and 100=confidence interval)
    offset - shift (s) of laser time points, as control
    pplot - if True, show plots
    ylim - set y axis limits of brain state percentage plot
    @Returns
    BS - 3D data matrix of brain state percentages (mice x time bins x brain state)
    mice - array of mouse names, corresponding to rows in $BS
    t - array of time points, corresponding to columns in $BS
    df - dataframe of brain state percentages in time intervals before, during, and
         after laser stimulation
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    pre += edge
    post += edge

    BrainstateDict = {}
    mouse_order = []
    for rec in recordings:
        # get unique mice
        idf = re.split('_', rec)[0]
        BrainstateDict[idf] = []
        if not idf in mouse_order:
            mouse_order.append(idf)
    nmice = len(BrainstateDict)

    for rec in recordings:
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                              flatten_is=flatten_is)
        
        # define start and end points of analysis
        istart = int(np.round(tstart / dt))
        if tend == -1:
            iend = len(M)
        else:
            iend = int(np.round(tend / dt))
        # define start and end points for collecting laser trials
        ipre  = int(np.round(pre/dt))
        ipost = int(np.round(post/dt))
        
        # load and downsample laser vector
        lsr = sleepy.load_laser(ppath, rec)
        (idxs, idxe) = sleepy.laser_start_end(lsr, offset=offset)
        idxs = [int(i/nbin) for i in idxs]
        idxe = [int(i/nbin) for i in idxe]
        laser_dur = np.mean((np.array(idxe) - np.array(idxs))) * dt
        
        # collect brain states surrounding each laser trial
        for (i,j) in zip(idxs, idxe):
            if i>=ipre and i+ipost<=len(M)-1 and i>istart and i < iend:
                bs = M[i-ipre:i+ipost+1]                
                BrainstateDict[idf].append(bs) 

    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    izero = np.where(t>0)[0][0]  # 1st time bin overlapping with laser
    izero -= 1

    # create 3D data matrix of mice x time bins x brain states
    BS = np.zeros((nmice, len(t), 4))
    Trials = []
    imouse = 0
    for mouse in mouse_order:
        if cond==0:
            # collect all laser trials
            M = np.array(BrainstateDict[mouse])
            Trials.append(M)
            for state in range(1,5):
                C = np.zeros(M.shape)
                C[np.where(M==state)] = 1
                BS[imouse,:,state-1] = C.mean(axis=0)
        if cond>0:
            # collect laser trials during brain state $cond
            M = BrainstateDict[mouse]
            Msel = []
            for trial in M:
                if trial[izero] == cond:
                    Msel.append(trial)
            M = np.array(Msel)
            Trials.append(M)
            for state in range(1,5):
                C = np.zeros(M.shape)
                C[np.where(M==state)] = 1
                BS[imouse,:,state-1] = C.mean(axis=0)
        imouse += 1

    # flatten Trials
    Trials = reduce(lambda x,y: np.concatenate((x,y), axis=0),  Trials)
    # smooth mouse averages
    if sf > 0:
        for state in range(4):
            for i in range(nmice):
                BS[i, :, state] = smooth_data(BS[i, :, state], sf)
    nmice = imouse
    
    ###   GRAPHS   ###
    if pplot:
        state_label = {0:'REM', 1:'Wake', 2:'NREM', 3:'IS'}
        it = np.where((t >= -pre + edge) & (t <= post - edge))[0]
        plt.ion()
        
        if not single_mode:
            # plot average % time in each brain state surrounding laser
            plt.figure()
            ax = plt.axes([0.15, 0.15, 0.6, 0.7])
            colors = ['cyan', 'purple', 'gray', 'darkblue']
            for state in [3,2,1,0]:
                if type(ci) in [int, float]:
                    # plot confidence interval
                    BS2 = BS[:,:,state].reshape(-1,order='F') * 100
                    t2 = np.repeat(t, BS.shape[0])
                    sns.lineplot(x=t2, y=BS2, color=colors[state], ci=ci, err_kws={'alpha':0.4, 'zorder':3}, 
                                 linewidth=3, ax=ax)
                else:
                    # plot SD or SEM
                    tmp = BS[:, :, state].mean(axis=0) *100
                    plt.plot(t[it], tmp[it], color=colors[state], lw=3, label=state_label[state])
                    if nmice > 1:
                        if ci == 'sem':
                            smp = BS[:,:,state].std(axis=0) / np.sqrt(nmice) * 100
                        elif ci == 'sd':
                            smp = BS[:,:,state].std(axis=0) * 100
                        plt.fill_between(t[it], tmp[it]-smp[it], tmp[it]+smp[it], 
                                         color=colors[state], alpha=0.4, zorder=3)
            # set axis limits and labels
            plt.xlim([-pre+edge, post-edge])
            if len(ylim) == 2:
                plt.ylim(ylim)
            ax.add_patch(matplotlib.patches.Rectangle((0,0), laser_dur, 100, 
                                                      facecolor=[0.6, 0.6, 1]))
            sleepy.box_off(ax)
            plt.xlabel('Time (s)')
            plt.ylabel('Brain state (%)')
            plt.legend()
            plt.draw()
        else:
            # plot % brain states surrounding laser for each mouse
            plt.figure(figsize=(7,7))
            clrs = sns.color_palette("husl", nmice)
            for state in [3,2,1,0]:
                ax = plt.subplot('51' + str(5-state))
                for i in range(nmice):
                    plt.plot(t[it], BS[i,it,state]*100, color=clrs[i], label=mouse_order[i])
                # plot laser interval
                ax.add_patch(matplotlib.patches.Rectangle((0, 0), laser_dur, 100, 
                                                          facecolor=[0.6, 0.6, 1], 
                                                          alpha=0.8))
                # set axis limits and labels
                plt.xlim((t[it][0], t[it][-1]))
                if len(ylim) == 2:
                    plt.ylim(ylim)
                plt.ylabel('% ' + state_label[state])
                if state==0:
                    plt.xlabel('Time (s)')
                else:
                    ax.set_xticklabels([])
                if state==3:
                    ax.legend(mouse_order, bbox_to_anchor=(0., 1.0, 1., .102), loc=3, 
                              mode='expand', ncol=len(mouse_order), frameon=False)
            sleepy.box_off(ax)

        # plot brain state surrounding each laser trial
        plt.figure(figsize=(4,6))
        sleepy.set_fontarial()
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        cmap = plt.cm.jet
        my_map = cmap.from_list('ha', [[0,1,1],[0.5,0,1],[0.6, 0.6, 0.6],[0.1,0.1,0.5]], 4)
        x = list(range(Trials.shape[0]))
        im = ax.pcolorfast(t,np.array(x), np.flipud(Trials), cmap=my_map)
        im.set_clim([1,4])
        # plot laser interval
        ax.plot([0,0], [0, len(x)-1], color='white')
        ax.plot([laser_dur,laser_dur], [0, len(x)-1], color='white')
        # set axis limits and labels
        ax.axis('tight')
        plt.draw()
        plt.xlabel('Time (s)')
        plt.ylabel('Trial No.')
        sleepy.box_off(ax)
        plt.show()

    # create dataframe with baseline and laser values for each trial
    ilsr   = np.where((t>=0) & (t<=laser_dur))[0]
    ibase  = np.where((t>=-laser_dur) & (t<0))[0]
    iafter = np.where((t>=laser_dur) & (t<laser_dur*2))[0]
    S = ['REM', 'Wake', 'NREM', 'IS']
    mice = mouse_order + mouse_order + mouse_order
    lsr  = np.concatenate((np.ones((nmice,), dtype='int'), np.zeros((nmice,), dtype='int'), 
                           np.ones((nmice,), dtype='int')*2))
    lsr_char = pd.Series(['LSR']*nmice + ['PRE']*nmice + ['POST']*nmice, 
                         dtype='category')
    df = pd.DataFrame(columns = ['mouse'] + S + ['lsr'])
    df['mouse'] = mice
    df['lsr'] = lsr
    # slightly different dataframe organization
    df2 = pd.DataFrame(columns = ['mouse', 'state', 'perc', 'lsr'])
    for i, state in enumerate(S):
        state_perc = np.concatenate((BS[:,ilsr,i].mean(axis=1), BS[:,ibase,i].mean(axis=1), 
                                     BS[:,iafter,i].mean(axis=1)))*100
        state_label = [state]*len(state_perc)
        df[state]  = state_perc
        df2 = pd.concat([df2,
                         pd.DataFrame({'mouse':mice,
                                       'state':state_label,
                                       'perc':state_perc,
                                       'lsr':lsr_char})],
                        axis=0, ignore_index=True)
    if pplot:
        # plot bar grah of % time in each brain state during pre-laser vs. laser interval
        plt.figure()
        fig, axs = plt.subplots(2,2, constrained_layout=True)
        axs = axs.reshape(-1)
        if ci == 'sem':
            ci = 'se'
        for i in range(len(S)):
            sdf = df2.iloc[np.where(df2['state'] == S[i])[0], :].reset_index(drop=True)
            sdf = pd.concat([sdf.iloc[np.where(sdf.lsr=='PRE')[0]], 
                             sdf.iloc[np.where(sdf.lsr=='LSR')[0]]], 
                             axis=0, ignore_index=True)
            sdf.lsr = sdf.lsr.cat.remove_unused_categories()
            sns.barplot(x='lsr', y='perc', data=sdf, errorbar=ci,
                        palette={'PRE':'gray', 'LSR':'blue'}, ax=axs[i])
            lines = sns.lineplot(x='lsr', y='perc', hue='mouse', data=sdf, errorbar=None, 
                                 markersize=0, legend=False, ax=axs[i])
            _ = [l.set_color('black') for l in lines.get_lines()]
            axs[i].set_title(S[i]); axs[i].set_ylabel('Amount (%)')
        plt.show()

    # stats
    clabs = ['% time spent in ' + state for state in S]
    pwaves.pairT_from_df(df, cond_col='lsr', cond1=1, cond2=0, test_cols=S, 
                         c1_label='during-laser', c2_label='pre-laser', test_col_labels=clabs)
    return BS, mouse_order, t, df2, Trials


def laser_triggered_eeg(ppath, name, pre, post, fmax, pnorm=1, psmooth=0, vm=[], tstart=0,
                        tend=-1, cond=0, harmcs=0, iplt_level=1, peeg2=False, prune_trials=False, 
                        mu=[10,100], offset=0, pplot=True):
    """
    Calculate average laser-triggered spectrogram for a recording
    @Params
    ppath - base folder
    name - recording folder
    pre, post - time window (s) before and after laser onset
    fmax - maximum frequency in spectrogram
    pnorm - method for spectrogram normalization (0=no normalization
                                                  1=normalize SP by recording
                                                  2=normalize SP by pre-lsr baseline interval)
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    vm - controls spectrogram saturation
    tstart, tend - time (s) into recording to start and stop collecting data
    cond - if 0, plot all laser trials
           if integer, only plot laser trials where mouse was in brain state $cond 
                       during onset of the laser
                       
    harmcs - if > 0, interpolate harmonics of base frequency $harmcs
    iplt_level - if 1, interpolate one SP row above/below harmonic frequencies
                 if 2, interpolate two SP rows
    peeg2 - if True, load prefrontal spectrogram 
    prune_trials - if True, automatically remove trials with EEG/EMG artifacts
    mu - [min,max] frequencies summed to get EMG amplitude
    offset - shift (s) of laser time points, as control
    pplot - if True, show plot
    @Returns
    EEGLsr - average EEG spectrogram surrounding laser (freq x time bins)
    EMGLsr - average EMG spectrogram
    freq[ifreq] - array of frequencies, corresponding to rows in $EEGLsr
    t - array of time points, corresponding to columns in $EEGLsr
    """
    def _interpolate_harmonics(SP, freq, fmax, harmcs, iplt_level):
        """
        Interpolate harmonics of base frequency $harmcs by averaging across 3-5 
        surrounding frequencies
        """
        df = freq[2]-freq[1]
        for h in np.arange(harmcs, fmax, harmcs):
            i = np.argmin(np.abs(np.round(freq,1) - h))
            if np.abs(freq[i] - h) < df/2 and h != 60: 
                if iplt_level == 2:
                    SP[i,:] = (SP[i-2:i,:] + SP[i+1:i+3,:]).mean(axis=0) * 0.5
                elif iplt_level == 1:
                    SP[i,:] = (SP[i-1,:] + SP[i+1,:]) * 0.5
                else:
                    pass
        return SP
    
    # load sampling rate
    sr = sleepy.get_snr(ppath, name)
    nbin = int(np.round(sr) * 2.5)
    
    # load laser, get start and end idx of each stimulation train
    lsr = sleepy.load_laser(ppath, name)
    idxs, idxe = sleepy.laser_start_end(lsr, sr, offset=offset)
    laser_dur = np.mean((idxe-idxs)/sr)
    print('Average laser duration: %f; Number of trials %d' % (laser_dur, len(idxs)))
    # downsample laser to SP time    
    idxs = [int(i/nbin) for i in idxs]
    idxe = [int(i/nbin) for i in idxe]
    
    # load EEG and EMG signals
    if peeg2:
        P = so.loadmat(os.path.join(ppath, name,  'sp2_' + name + '.mat'))
    else:
        P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    Q = so.loadmat(os.path.join(ppath, name, 'msp_' + name + '.mat'))
    
    # load spectrogram
    if not peeg2:
        SPEEG = np.squeeze(P['SP'])
    else:
        SPEEG = np.squeeze(P['SP2'])
    SPEMG = np.squeeze(Q['mSP'])
    freq  = np.squeeze(P['freq'])
    t     = np.squeeze(P['t'])
    dt    = float(np.squeeze(P['dt']))
    ifreq = np.where(freq<=fmax)[0]
    # get indices of time window surrounding laser
    ipre  = int(np.round(pre/dt))
    ipost = int(np.round(post/dt))
    speeg_mean = SPEEG.mean(axis=1)
    spemg_mean = SPEMG.mean(axis=1)
    
    # interpolate harmonic frequencies
    if harmcs > 0:
        SPEEG = _interpolate_harmonics(SPEEG, freq, fmax, harmcs, iplt_level)
        SPEMG = _interpolate_harmonics(SPEMG, freq, fmax, harmcs, iplt_level)
    # normalize spectrograms by recording
    if pnorm == 1:
        SPEEG = np.divide(SPEEG, np.repeat(speeg_mean, len(t)).reshape(len(speeg_mean), len(t)))
        SPEMG = np.divide(SPEMG, np.repeat(spemg_mean, len(t)).reshape(len(spemg_mean), len(t)))
    
    # define start and and points of analysis
    if tend > -1:
        i = np.where((np.array(idxs)*dt >= tstart) & (np.array(idxs)*dt <= tend))[0]
    else:
        i = np.where(np.array(idxs)*dt >= tstart)[0]
    idxs = [idxs[j] for j in i]
    idxe = [idxe[j] for j in i]

    # eliminate laser trials with detected EEG/EMG artifacts
    skips = []
    skipe = []
    if prune_trials:
        for (i,j) in zip(idxs, idxe):
            A = SPEEG[0,i-ipre:i+ipost+1] / speeg_mean[0]
            B = SPEMG[0,i-ipre:i+ipost+1] / spemg_mean[0]
            k = np.where(A >= np.median(A)*50)[0]
            l = np.where(B >= np.median(B)*500)[0]
            if len(k) > 0 or len(l) > 0:
                skips.append(i)
                skipe.append(j)
    print("kicking out %d trials" % len(skips))
    prn_lsr = [[i,j] for i,j in zip(idxs, idxe) if i not in skips]
    idxs, idxe = zip(*prn_lsr)
    # collect laser trials starting in brain state $cond
    if cond > 0:
        M = sleepy.load_stateidx(ppath, name)[0]
        cnd_lsr = [[i,j] for i,j in zip(idxs, idxe) if i < len(M) and M[i]==cond]
        idxs, idxe = zip(*cnd_lsr)

    # collect and average spectrograms surrounding each qualifying laser trial
    eeg_sps = []
    emg_sps = []
    for (i,j) in zip(idxs, idxe):
        if i>=ipre and j+ipost < len(t): 
            eeg_sps.append(SPEEG[ifreq,i-ipre:i+ipost+1])
            emg_sps.append(SPEMG[ifreq,i-ipre:i+ipost+1])
    EEGLsr = np.array(eeg_sps).mean(axis=0)
    EMGLsr = np.array(emg_sps).mean(axis=0)
    
    # normalize spectrograms by pre-laser baseline interval
    if pnorm == 2:    
        for i in range(EEGLsr.shape[0]):
            EEGLsr[i,:] = np.divide(EEGLsr[i,:], np.sum(np.abs(EEGLsr[i,0:ipre]))/(1.0*ipre))
            EMGLsr[i,:] = np.divide(EMGLsr[i,:], np.sum(np.abs(EMGLsr[i,0:ipre]))/(1.0*ipre))
    # smooth spectrograms
    EEGLsr = adjust_spectrogram(EEGLsr, pnorm=0, psmooth=psmooth)
    EMGLsr = adjust_spectrogram(EMGLsr, pnorm=0, psmooth=psmooth)
        
    dt = (1.0 / sr) * nbin
    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    f = freq[ifreq]

    if pplot:
        # plot laser-triggered EEG spectrogram
        plt.ion()
        plt.figure(figsize=(10,8))
        ax = plt.axes([0.1, 0.55, 0.4, 0.35])
        im = ax.pcolorfast(t, f, EEGLsr, cmap='jet')
        if len(vm) == 2:
            im.set_clim(vm)
        # plot laser interval
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        # set axis limits and labels
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)
        plt.title('EEG', fontsize=12)
        cbar = plt.colorbar()
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # plot EEG power spectrum during laser vs pre-laser interval
        ax = plt.axes([0.62, 0.55, 0.35, 0.35])
        ilsr = np.where((t>=0) & (t<=120))[0]        
        plt.plot(f,EEGLsr[:,0:ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        sleepy.box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        # plot laser-triggered EMG spectrogram
        ax = plt.axes([0.1, 0.1, 0.4, 0.35])
        im = ax.pcolorfast(t, f, EMGLsr, cmap='jet')
        # plot laser interval
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        # set axis limits and labels
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)    
        plt.title('EMG', fontsize=12)
        cbar = plt.colorbar()
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # plot EMG amplitude surrounding laser trials
        ax = plt.axes([0.62, 0.1, 0.35, 0.35])
        mf = np.where((f>=mu[0]) & (f<= mu[1]))[0]
        df = f[1]-f[0]
        # amplitude is square root of (integral over each frequency)
        avg_emg = np.sqrt(EMGLsr[mf,:].sum(axis=0)*df)    
        m = np.max(avg_emg)*1.5
        plt.plot([0,0], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.plot([laser_dur,laser_dur], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.xlim((t[0], t[-1]))
        plt.ylim((0,m))
        plt.plot(t,avg_emg, color='black', lw=2)
        sleepy.box_off(ax)     
        plt.xlabel('Time (s)')
        plt.ylabel('EMG ampl. (uV)')        
        plt.show()
    return EEGLsr, EMGLsr, freq[ifreq], t


def laser_triggered_eeg_avg(ppath, recordings, pre, post, fmax, laser_dur, pnorm=1, psmooth=0, vm=[],
                            bands=[(0.5,4),(6,10),(11,15),(55,99)], band_labels=[], band_colors=[],
                            tstart=0, tend=-1, cond=0, harmcs=0, iplt_level=1, peeg2=False, sf=0,
                            prune_trials=False, ci='sem', mu=[10,100], offset=0, pplot=True, ylim=[]):
    """
    Calculate average laser-triggered spectrogram and frequency band power for list of recordings
    @Params
    ppath - base folder
    recordings - list of recordings
    pre, post - time window (s) before and after laser onset
    fmax - maximum frequency in spectrogram
    laser_dur - duration (s) of laser stimulation trials
    pnorm - method for spectrogram normalization (0=no normalization
                                                  1=normalize SP by recording
                                                  2=normalize SP by pre-lsr baseline interval)
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    vm - controls spectrogram saturation
    bands - list of tuples with min and max frequencies in each power band
            e.g. [ [0.5,4], [6,10], [11,15], [55,100] ]
    band_labels - optional list of descriptive names for each freq band
            e.g. ['delta', 'theta', 'sigma', 'gamma']
    band_colors - optional list of colors to plot each freq band
            e.g. ['firebrick', 'limegreen', 'cyan', 'purple']
    tstart, tend - time (s) into recording to start and stop collecting data
    cond - if 0, plot all laser trials
           if integer, only plot laser trials where mouse was in brain state $cond 
                       during onset of the laser
    harmcs - if > 0, interpolate harmonics of base frequency $harmcs
    iplt_level - if 1, interpolate one SP row above/below harmonic frequencies
                 if 2, interpolate two SP rows
    peeg2 - if True, load prefrontal spectrogram
    sf - smoothing factor for vectors of frequency band power
    prune_trials - if True, automatically remove trials with EEG/EMG artifacts
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                          integer between 0 and 100=confidence interval)
    mu - [min,max] frequencies summed to get EMG amplitude
    offset - shift (s) of laser time points, as control
    pplot - if True, show plot
    ylim - set y axis limits for frequency band plot
    @Returns
    EEGSpec - dictionary with EEG spectrogram for each mouse
    PwrBands - dictionary with EEG power (mice x time bins surrounding laser) for each freq band
    mice - list of mouse names, corresponding with rows in each $PwrBands array
    t - list of time points, corresponding with columns in each $PwrBands array
    df2 - dataframe with mean power of each freq band during laser vs pre-laser intervals
    """
    # clean data inputs
    if len(band_labels) != len(bands):
        band_labels = [str(b) for b in bands]
    if len(band_colors) != len(bands):
        band_colors = colorcode_mice([], return_colorlist=True)[0:len(bands)]
    if ci == 'sem':
        ci = 68
    
    # collect EEG and EMG spectrograms for each mouse
    EEGSpec = {}
    EMGSpec = {}
    mice = []
    for rec in recordings:
        # get unique mice
        idf = re.split('_', rec)[0]
        if not(idf in mice):
            mice.append(idf)
        EEGSpec[idf] = []
        EMGSpec[idf] = []
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        
        # get laser-triggered EEG and EMG spectrograms for each recording
        EEG, EMG, f, t = laser_triggered_eeg(ppath, rec, pre, post, fmax, pnorm=pnorm, psmooth=psmooth, 
                                             tstart=tstart, tend=tend, cond=cond, prune_trials=prune_trials, peeg2=peeg2,
                                             harmcs=harmcs, iplt_level=iplt_level, mu=mu, offset=offset, pplot=False)
        EEGSpec[idf].append(EEG)
        EMGSpec[idf].append(EMG)
    
    # create dictionary to store freq band power (key=freq band, value=matrix of mice x time bins)
    PwrBands = {b:np.zeros((len(mice), len(t))) for b in bands}
    
    for row, idf in enumerate(mice):
        # get average SP for each mouse
        ms_sp = np.array(EEGSpec[idf]).mean(axis=0)
        ms_emg = np.array(EMGSpec[idf]).mean(axis=0)
        # calculate power of each freq band from averaged SP
        for b in bands:
            ifreq = np.intersect1d(np.where(f >= b[0])[0], np.where(f <= b[1])[0])
            ms_band = np.mean(ms_sp[ifreq, :], axis=0)
            if sf > 0:
                ms_band = smooth_data(ms_band, sf)
            PwrBands[b][row, :] = ms_band
        EEGSpec[idf] = ms_sp
        EMGSpec[idf] = ms_emg
    # get average EEG/EMG spectrogram across all subjects
    EEGLsr = np.array([EEGSpec[k] for k in mice]).mean(axis=0)
    EMGLsr = np.array([EMGSpec[k] for k in mice]).mean(axis=0)
    
    # get indices of harmonic frequencies
    mf = np.where((f >= mu[0]) & (f <= mu[1]))[0]
    if harmcs > 0:
        harm_freq = np.arange(0, f.max(), harmcs)
        for h in harm_freq:
            mf = np.setdiff1d(mf, mf[np.where(f[mf]==h)[0]])
    # remove harmonics and calculate EMG amplitude
    df = f[1] - f[0]
    EMGAmpl = np.zeros((len(mice), EEGLsr.shape[1]))
    i=0
    for idf in mice:
        # amplitude is square root of (integral over each frequency)
        if harmcs == 0:
            EMGAmpl[i,:] = np.sqrt(EMGSpec[idf][mf,:].sum(axis=0)*df)
        else:
            tmp = 0
            for qf in mf:
                tmp += EMGSpec[idf][qf,:] * (f[qf] - f[qf-1])
            EMGAmpl[i,:] = np.sqrt(tmp)
        i += 1
    avg_emg = EMGAmpl.mean(axis=0)
    sem_emg = EMGAmpl.std(axis=0) / np.sqrt(len(mice))

    if pplot:
        plt.ion()
        plt.figure(figsize=(12,10))
        # plot average laser-triggered EEG spectrogram
        ax = plt.axes([0.1, 0.55, 0.4, 0.4])
        im = ax.pcolorfast(t, f, EEGLsr, cmap='jet')
        if len(vm) == 2:
            im.set_clim(vm)
        ax.plot([0,0], [0,f[-1]], color=(1,1,1))
        # plot laser interval
        ax.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)
        plt.title('EEG')
        cbar = plt.colorbar(im, ax=ax, pad=0.0)
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # plot average power spectrum during laser and pre-laser interval
        ax = plt.axes([0.6, 0.55, 0.3, 0.4])
        ipre = np.where(t<0)[0]
        ilsr = np.where((t>=0) & (t<=round(laser_dur)))[0]        
        plt.plot(f,EEGLsr[:,ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        sleepy.box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        # plot average laser-triggered EMG spectrogram
        ax = plt.axes([0.1, 0.05, 0.4, 0.4])
        im = ax.pcolorfast(t, f, EMGLsr, cmap='jet')
        ax.plot([0,0], [0,f[-1]], color=(1,1,1))
        # plot laser interval
        ax.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)    
        plt.title('EMG')
        cbar = plt.colorbar(im, ax=ax, pad=0.0)
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
            
        # plot average laser-triggered power of frequency bands
        ax = plt.axes([0.6, 0.05, 0.3, 0.4])
        for b,l,c in zip(bands, band_labels, band_colors):
            data = PwrBands[b].mean(axis=0)
            yerr = PwrBands[b].std(axis=0) / np.sqrt(len(mice))
            ax.plot(t, data, color=c, label=l)
            ax.fill_between(t, data-yerr, data+yerr, color=c, alpha=0.3)
        ax.set_xlim((t[0], t[-1]))
        sleepy.box_off(ax)
        ax.set_xlabel('Time (s)')
        if pnorm > 0:
            ax.set_ylabel('Avg rel. power')
        else:
            ax.set_ylabel('Avg band power (uV^2)')
        if len(ylim) == 2:
            ax.set_ylim(ylim)
        ax.legend()
        plt.show()
    
    # get indices of pre-laser, laser, and post-laser intervals
    ilsr   = np.where((t>=0) & (t<=laser_dur))[0]
    ibase  = np.where((t>=-laser_dur) & (t<0))[0]
    iafter = np.where((t>=laser_dur) & (t<laser_dur*2))[0]
    m = mice + mice + mice
    lsr  = np.concatenate((np.ones((len(mice),), dtype='int'), 
                           np.zeros((len(mice),), dtype='int'), 
                           np.ones((len(mice),), dtype='int')*2))
    lsr_char = pd.Series(['LSR']*len(mice) + ['PRE']*len(mice) + ['POST']*len(mice), dtype='category')
    # create dataframes with power values for each frequency band
    df = pd.DataFrame(columns = ['mouse'] + band_labels + ['lsr'])
    df['mouse'] = m
    df['lsr'] = lsr
    df2 = pd.DataFrame(columns = ['mouse', 'band', 'pwr', 'lsr'])
    for b,l in zip(bands, band_labels):
        base_data = PwrBands[b][:,ibase].mean(axis=1)
        lsr_data = PwrBands[b][:,ilsr].mean(axis=1)
        post_data = PwrBands[b][:,iafter].mean(axis=1)
        # get mean power of each spectral band before/during/after laser
        b_pwr = np.concatenate((lsr_data, base_data, post_data))
        b_label = [l]*len(b_pwr)
        df[l] = b_pwr
        df2 = pd.concat([df2,pd.DataFrame({'mouse':m, 'band':b_label, 'pwr':b_pwr, 'lsr':lsr_char})], axis=0, ignore_index=True)

    # plot average power of each frequency band during pre-laser vs. laser interval
    if pplot:
        plt.figure()
        fig, axs = plt.subplots(2,2, constrained_layout=True)
        axs = axs.reshape(-1)
        if ci == 'sem':
            ci = 'se'
        elif type(ci) == int:
            ci = ('ci',ci)
        for i in range(len(band_labels)):
            bdf = df2.iloc[np.where(df2['band'] == band_labels[i])[0], :].reset_index(drop=True)
            
            bdf = pd.concat([bdf.iloc[np.where(bdf.lsr=='PRE')[0]], 
                             bdf.iloc[np.where(bdf.lsr=='LSR')[0]]], 
                             axis=0, ignore_index=True)
            bdf.lsr = bdf.lsr.cat.remove_unused_categories()
            sns.pointplot(x='lsr', y='pwr', data=bdf, markers='o', errorbar=ci,
                        palette={'PRE':'gray', 'LSR':'blue'}, ax=axs[i])
            
            lines = sns.lineplot(x='lsr', y='pwr', hue='mouse', data=bdf, errorbar=None, 
                                 markersize=0, legend=False, ax=axs[i])
            _ = [l.set_color('black') for l in lines.get_lines()]
            axs[i].set_title(band_labels[i])
            if pnorm == 0:
                axs[i].set_ylabel('Power uV^2s')
            else:
                axs[i].set_ylabel('Rel. Power')
        plt.show()
    
    # stats - mean freq band power during pre-laser vs laser intervals
    clabs = [l + ' (' + str(b[0]) + '-' + str(b[1]) + ' Hz)' for b,l in zip(bands, band_labels)]
    pwaves.pairT_from_df(df, cond_col='lsr', cond1=1, cond2=0, test_cols=band_labels, 
                         c1_label='during laser', c2_label='pre-laser', test_col_labels=clabs)
    return EEGSpec, PwrBands, mice, t, df2


def laser_transition_probability(ppath, recordings, pre, post, tstart=0, tend=-1,
                                  ma_thr=20, ma_state=3, sf=0, offset=0):
    """
    Calculate laser-triggered likelihood of transition from IS --> REM sleep
    @Params
    ppath - base folder
    recordings - list of recordings
    pre, post - time window (s) before and after laser onset
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    sf - smoothing factor for transition state timecourses
    offset - shift (s) of laser time points, as control
    @Returns
    df - dataframe of IS --> REM transition probabilities for pre-laser, laser,
         and post-laser intervals
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    # get unique mice, create data dictionary
    mice = list({rec.split('_')[0]:[] for rec in recordings})
    BrainstateDict = {rec:[] for rec in recordings}
    # avg probability of transition during/before/after laser
    trans_prob = {m : [] for m in mice}
    
    for rec in recordings:
        print('Getting data for ' + rec + ' ...')
        idf = rec.split('_')[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brainstate annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=False)

        # define start and end points of analysis
        istart = int(np.round(tstart / dt))
        if tend == -1:
            iend = len(M)
        else:
            iend = int(np.round(tend / dt))
        # get indices of time window surrounding laser
        ipre  = int(np.round(pre/dt))
        ipost = int(np.round(post/dt))
        
        # load laser, get start and end idx of each stimulation train
        lsr = sleepy.load_laser(ppath, rec)
        (idxs, idxe) = sleepy.laser_start_end(lsr, sr, offset=offset)
        idxs = [int(i/nbin) for i in idxs]
        idxe = [int(i/nbin) for i in idxe]
        laser_dur = np.mean((np.array(idxe) - np.array(idxs))) * dt
        laser_dn = np.zeros((len(M),))
        for (i,j) in zip(idxs, idxe):
            if i>=ipre and i+ipost<=len(M)-1 and i>istart and i < iend:
                # collect vector of pre-REM (1) and pre-wake (2) IS bouts
                tp = np.zeros((ipre+ipost+1,))
                M_cut = M[i-ipre:i+ipost+1]                
                tp[np.where(M_cut==4)[0]] = 1
                tp[np.where(M_cut==5)[0]] = 2
                BrainstateDict[rec].append(tp)
                # label downsampled indices of laser (1), pre-laser (2), and post-laser (3)
                laser_dn[i:j+1] = 1
                laser_dn[i-int(round(laser_dur/dt)) : i] = 2
                laser_dn[j+1 : j+1+int(round(laser_dur/dt))] = 3
        [laser_idx, pre_laser_idx, post_laser_idx] = [np.where(laser_dn==i)[0] for i in [1,2,3]]
        trans_idx = np.concatenate((np.where(M==4)[0], np.where(M==5)[0]), axis=0)
        trans_seq = sleepy.get_sequences(trans_idx)
        
        # collect total no. of transitions and % transitions ending in REM sleep
        l = {'num_trans':0, 'success_trans':0}
        pre_l = {'num_trans':0, 'success_trans':0}
        post_l = {'num_trans':0, 'success_trans':0}
        for tseq in trans_seq:
            # during laser period ($laser_dur s)
            if tseq[0] in laser_idx:
                l['num_trans'] += 1
                if all(M[tseq] == 4):
                    l['success_trans'] += 1
            # during pre-laser period ($laser_dur s)
            elif tseq[0] in pre_laser_idx:
                pre_l['num_trans'] += 1
                if all(M[tseq] == 4):
                    pre_l['success_trans'] += 1
            # during post-laser period ($laser_dur s)
            elif tseq[0] in post_laser_idx:
                post_l['num_trans'] += 1
                if all(M[tseq] == 4):
                    post_l['success_trans'] += 1
        trans_prob[idf].append(np.array(([pre_l['success_trans']/pre_l['num_trans']*100,
                           l['success_trans']/l['num_trans']*100,
                           post_l['success_trans']/post_l['num_trans']*100])))
    # create mouse-averaged matrix of transition probabilities (mice x pre/lsr/post)
    trans_prob_mx = np.zeros((len(mice), 3))
    for row,m in enumerate(mice):
        trans_prob_mx[row,:] = np.array((trans_prob[m])).mean(axis=0)
    
    conditions = ['pre-laser', 'laser', 'post-laser']
    # create dataframe with transition probability data
    df = pd.DataFrame({'mouse' : np.tile(mice, len(conditions)),
                       'cond' : np.repeat(conditions, len(mice)),
                       'perc' : np.reshape(trans_prob_mx,-1,order='F')})
    
    ###   GRAPHS   ###
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(figsize=(7,10), nrows=2, ncols=1, gridspec_kw={'height_ratios':[2,3]})
    
    # create 3D timecourse data matrix (mice x time bins x pre/lsr/post)
    transitions_dict = pwaves.mx2d_dict(BrainstateDict, mouse_avg='mouse', d1_size=len(tp))
    transitions_mx = np.zeros((len(mice), len(tp), 3))
    for row, m in enumerate(mice):
        tt = np.sum(transitions_dict[m]>0, axis=0)  # no. transition state trials
        st = np.sum(transitions_dict[m]==1, axis=0)  # no. successful transition trials
        ft = np.sum(transitions_dict[m]==2, axis=0)  # no. failed transition trials
        
        st_perc = (st/transitions_dict[m].shape[0])*100  # % successful transition trials
        ft_perc = (ft/transitions_dict[m].shape[0])*100  # % failed transition trials
        bin_prob = [s/t*100 if t>0 else np.nan for s,t in zip(st,tt)]  # prob. of given transition being successful
        if sf > 0:
                st_perc = smooth_data(st_perc, sf)
                ft_perc = smooth_data(ft_perc, sf)
        transitions_mx[row, :, 0] = st_perc
        transitions_mx[row, :, 1] = ft_perc
        transitions_mx[row, :, 2] = bin_prob
    
    # plot timecourses of successful and failed transitions
    t = np.linspace(-ipre*dt, ipost*dt+1, ipre+ipost+1) 
    # % time in successful transitions
    sdata = np.nanmean(transitions_mx[:,:,0], axis=0)
    syerr = np.nanstd(transitions_mx[:,:,0], axis=0) / np.sqrt(len(mice))
    ax1.plot(t, sdata, color='darkblue', lw=3, label='IS-R')
    ax1.fill_between(t, sdata-syerr, sdata+syerr, color='darkblue', alpha=0.3)
    # % time in failed transitions
    fdata = np.nanmean(transitions_mx[:,:,1], axis=0)
    fyerr = np.nanstd(transitions_mx[:,:,1], axis=0) / np.sqrt(len(mice))
    ax1.plot(t, fdata, color='red', lw=3, label='IS-W')
    ax1.fill_between(t, fdata-fyerr, fdata+fyerr, color='red', alpha=0.3)
    ax1.add_patch(matplotlib.patches.Rectangle((0,0), laser_dur, ax1.get_ylim()[1], 
                                               facecolor=[0.6, 0.6, 1], zorder=0))
    sleepy.box_off(ax1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('% time spent')
    ax1.legend()
    plt.draw()
    
    # plot bar graph of avg transition probability
    sns.barplot(x='cond', y='perc', data=df, ci=68, ax=ax2, palette={'pre-laser':'gray',
                                                                     'laser':'lightblue',
                                                                     'post-laser':'gray'})
    sns.pointplot(x='cond', y='perc', hue='mouse', data=df, ci=None, markers='', color='black', ax=ax2)
    ax2.set_ylabel('Transition probability (%)');
    ax2.set_title('Percent IS bouts transitioning to REM')
    ax2.get_legend().remove()
    plt.show()
                
    # stats - transition probability during pre-laser vs laser vs post-laser intervals
    res_anova = AnovaRM(data=df, depvar='perc', subject='mouse', within=['cond']).fit()
    mc = MultiComparison(df['perc'], df['cond']).allpairtest(scipy.stats.ttest_rel, method='bonf')
    print(res_anova)
    print('p = ' + str(float(res_anova.anova_table['Pr > F'])))
    print(''); print(mc[0])
    
    return df
    

def state_online_analysis(ppath, recordings, istate=1, single_mode=False, overlap=0, 
                          ma_thr=20, ma_state=3, flatten_is=False, ylim=[], 
                          pplot=True, print_stats=True):
    """
    Compare duration of laser-on vs laser-off brain states from closed-loop experiments
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state to analyze
    single_mode - if True, plot individual brain state durations
                  if False, plot mean brain state duration for each mouse
    overlap - float between 0 and 100, specifying minimum percentage of overlap
              between detected (online) and annotated (offline) brain states
              required to include episode in analysis
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    ylim - set y axis limits of bar graph
    pplot - if True, show plots
    print_stats - if True, show results of t-test comparison (laser OFF vs laser ON)
    @Returns
    df - dataframe with durations of laser-on and laser-off brain states
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if type(istate) in [list, tuple]:
        istate = istate[0]
    overlap /= 100.0
    
    mice = dict()
    # get unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    if len(mice) == 1:
        single_mode = True
    
    # collect durations of control & experimental brain states
    dur_exp = {m:[] for m in mice}
    dur_ctr = {m:[] for m in mice}
    
    for rec in recordings:
        print('Getting data for ' + rec + ' ...')
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M,_ = sleepy.load_stateidx(ppath, rec)
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                              flatten_is=flatten_is)
        
        # load laser and online brain state detection
        laser = sleepy.load_laser(ppath, rec)
        rem_trig = so.loadmat(os.path.join(ppath, rec, 'rem_trig_%s.mat'%rec), 
                              squeeze_me=True)['rem_trig']
        # downsample to SP time
        laser = downsample_vec(laser, nbin)
        laser[np.where(laser>0)] = 1
        rem_trig = downsample_vec(rem_trig, nbin)
        rem_trig[np.where(rem_trig>0)] = 1
        laser_idx = np.where(laser==1)[0]
        rem_idx = np.where(rem_trig==1)[0]
    
        # get brain state sequences from offline analysis
        seq = sleepy.get_sequences(np.where(M==istate)[0])
        for s in seq:
            # check overlap between online & offline brain state sequences
            isect = np.intersect1d(s, rem_idx)
            if len(np.intersect1d(s, rem_idx)) > 0 and float(len(isect)) / len(s) >= overlap:
                drn = (s[-1]-s[0]+1)*dt
                # collect duration of laser-on or laser-off brain state
                if len(np.intersect1d(isect, laser_idx))>0:
                    dur_exp[idf].append(drn)
                else:
                    dur_ctr[idf].append(drn)
                    
    dataframe = pd.DataFrame(columns=['mouse','lsr','dur'])
    if len(mice) == 1 or single_mode == True:
        # collect duration of each REM period
        for m in mice:
            dataframe = pd.concat([dataframe, pd.DataFrame({'mouse':m,
                                                            'lsr':0,
                                                            'dur':dur_ctr[m]})],
                                  axis=0, ignore_index=True)
            dataframe = pd.concat([dataframe, pd.DataFrame({'mouse':m,
                                                            'lsr':1,
                                                            'dur':dur_exp[m]})],
                                  axis=0, ignore_index=True)
    else:
        # get average REM durations for each mouse
        for m in mice:
            mdata = [np.array(dur_ctr[m]).mean(), np.array(dur_exp[m]).mean()]
            dataframe = pd.concat([dataframe, pd.DataFrame({'mouse':m,
                                                            'lsr':np.array([0,1]),
                                                            'dur':np.array(mdata)})],
                                  axis=0, ignore_index=True)
    if pplot:
        fig = plt.figure()
        ax = plt.gca()
        sns.barplot(data=dataframe, x='lsr', y='dur', errorbar='se', 
                    palette=['gray','blue'], ax=ax)
        if single_mode:
            # plot duration of each laser-off and laser-on REM period
            sns.stripplot(data=dataframe, x='lsr', y='dur', color='black', ax=ax)
            
        else:
            # plot avg duration of laser-off vs laser-on REM periods for each mouse
            lines = sns.lineplot(data=dataframe, x='lsr', y='dur', hue='mouse', 
                                 errorbar=None, markersize=0, legend=False, ax=ax)
            _ = [l.set_color('black') for l in lines.get_lines()]
        # set axis limits and labels
        ax.set_ylabel('REM duration (s)')
        if len(ylim) == 2:
            ax.set_ylim(ylim)
        sns.despine()
        
    if print_stats:
        if single_mode:
            # unpaired t-test
            p = scipy.stats.ttest_ind(np.array(dataframe.dur[np.where(dataframe.lsr==0)[0]]),
                                      np.array(dataframe.dur[np.where(dataframe.lsr==1)[0]]))
            ttype = 'unpaired'
        else:
            # paired t-test
            p = scipy.stats.ttest_rel(np.array(dataframe.dur[np.where(dataframe.lsr==0)[0]]),
                                      np.array(dataframe.dur[np.where(dataframe.lsr==1)[0]]))
            ttype = 'paired'
        # print stats
        print(f'\n###   Laser OFF vs laser ON, state={istate}, {ttype} t-test)')
        print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}\n')
        
    return dataframe


def bootstrap_online_analysis(df, dv, iv, virus='', nboots=1000, resample_mice=True,
                              alpha=0.05, shuffle=True, seed=None, pplot=True):
    """
    Bootstrap mean value for each experimental condition in a dataframe
    @Params
    df - dataframe with mouse name, condition, and numerical value for each data trial
    dv - dependent variable column name (e.g. 'dur' or 'freq')
    iv - independent variable column name (e.g. 'lsr' [0 vs 1] or 'dose' [saline vs cno])
    virus - name of virus expressed in mouse group (e.g. 'chr2' or 'hm3dq' or 'yfp')
    nboots - number of times to resample dataset using bootstrapping
    resample_mice - if True, randomly select new sample of mice for each bootstrap iteration
    alpha - plot shows 1-$alpha confidence intervals
    shuffle - if True, randomly shuffle condition IDs and bootstrap "sham" data
    seed - if integer, set seed for shuffling condition IDs
    pplot - if True, plot bootstrap graph
    @Returns
    boot_df - dataframe with mean condition values for each of $nboots trials
    """
    
    # get unique mice
    mice = [m for i,m in enumerate(list(df.mouse)) if list(df.mouse).index(m)==i]
    nmice = len(mice)
    # get conditions (0 vs 1 for laser experiment, saline vs cno for DREADD experiment)
    conditions = [c for j,c in enumerate(list(df[iv])) if list(df[iv]).index(c)==j]
    
    if shuffle:
        np.random.seed(seed)
        # for each mouse, randomly shuffle condition labels
        iv_shuffle = []
        for m in mice:
            midx = np.where(df.mouse==m)[0]
            n_c0 = len(np.where(df[iv][midx]==conditions[0])[0])
            mshuf = [conditions[1]]*len(midx)
            for ishuf in np.random.choice(range(len(mshuf)), size=n_c0, replace=False):
                mshuf[ishuf] = conditions[0]
            iv_shuffle += mshuf
        df[f'{iv}_shuffle'] = iv_shuffle
    
    np.random.seed(None)

    # collect bootstrapped data (rows = bootstrap trials, cols = conditions)
    # columns: ctrl true, exp true, ctrl shuffled, exp shuffled, true dif, shuffled dif
    boot_mx = np.zeros((nboots,6))
    
    for i in range(nboots):
        # get new sample of mice
        if resample_mice:
            bmice = np.random.choice(mice, size=nmice, replace=True)
        else:
            bmice = list(mice)
        # collect mean resampled values for each mouse (rows = mice)
        btrial = np.zeros((nmice,6))
        if i % 500 == 0:
            print('Bootstrapping trial ' + str(i) + ' of ' + str(nboots) + ' ...')
        for midx,m in enumerate(bmice):
            # for mouse $m, get indices of real and shuffled rows for each condition
            c0 = np.where((df.mouse==m) & (df[iv]==conditions[0]))[0]
            c1 = np.where((df.mouse==m) & (df[iv]==conditions[1]))[0]
            # randomly select a new dataset of trials (same size as original)
            c0_select = np.random.choice(c0, size=len(c0), replace=True)
            c1_select = np.random.choice(c1, size=len(c1), replace=True)
            # calculate mean value for each condition, store in mouse row of trial mx
            btrial[midx,0] = np.nanmean(df[dv][c0_select])
            btrial[midx,1] = np.nanmean(df[dv][c1_select])
            if shuffle:
                # repeat for shuffled rows
                c0_shuf = np.where((df.mouse==m) & (df[f'{iv}_shuffle']==conditions[0]))[0]
                c1_shuf = np.where((df.mouse==m) & (df[f'{iv}_shuffle']==conditions[1]))[0]
                c0_shuf_select = np.random.choice(c0_shuf, size=len(c0_shuf), replace=True)
                c1_shuf_select = np.random.choice(c1_shuf, size=len(c1_shuf), replace=True)
                btrial[midx,2] = np.nanmean(df[dv][c0_shuf_select])
                btrial[midx,3] = np.nanmean(df[dv][c1_shuf_select].mean())
        # for each mouse, get avg difference between conditions
        btrial[:,4] = btrial[:,1] - btrial[:,0]
        btrial[:,5] = btrial[:,3] - btrial[:,2]
        # get "experiment mean" across all sampled mice for each bootstrap trial
        boot_mx[i,:] = np.nanmean(btrial, axis=0)
    
    # store bootstrapped means and condition IDs in dataframe
    IV = np.repeat(conditions + conditions + ['diff', 'diff'], nboots)
    DV = boot_mx.flatten(order='F')
    SHUF = np.repeat([0,0,1,1,0,1], nboots)
    if not shuffle:
        SHUF[:] = 0
    boot_df = pd.DataFrame({iv:IV, dv:DV, 'shuffled':SHUF, 'virus':virus})   
         
    # plot data
    if pplot:
        plot_bootstrap_online_analysis(boot_df, dv=dv, iv=iv, alpha=alpha)

    return boot_df
    

def plot_bootstrap_online_analysis(df, dv, iv, mode=0, plotType='bar', alpha=0.05, 
                                   ylim=[], pload=False):
    """
    Plot mean REM durations for bootstrapped closed-loop data
    @Params
    df - dataframe with mean REM duration for each bootstrap trial
         * Columns include laser ID (0=OFF, 1=ON), (optional) virus type, and 
           (optional) shuffled status (0=real laser IDs, 1=shuffled laser IDs)
    mode - if 0: plot laser-off and laser-on REM duration for each group
           if 1: plot difference in REM duration between laser-off and laser-on trials 
    alpha - plot shows 1-$alpha confidence intervals
    pload - optional string specifying a filename to load the dataframe
    @Returns
    None
    """
    if os.path.isfile(pload):
        df = pd.read_pickle(pload)
        
    # clean data input
    if 'shuffled' not in df.columns:
        df['shuffled'] = 0
    if 'virus' not in df.columns:
        df['virus'] = '--'
        
    # get unique viruses
    viruses = list(set(df.virus))
    vsort = [list(df.virus).index(v) for v in viruses]
    viruses = [v for _,v in sorted(zip(vsort,viruses))]
    # create subplot for each virus
    if len(viruses)==1:
        fig = plt.figure()
        axs = [plt.gca()]
    else:
        fig,axs = plt.subplots(nrows=1, ncols=len(viruses), constrained_layout=True)
    # get conditions (0 vs 1 for laser experiment, saline vs cno for DREADD experiment)
    conditions = list(set(df[iv]))
    csort = [list(df[iv]).index(c) for c in conditions]
    conditions = [c for _,c in sorted(zip(csort,conditions))]
    alpha2 = alpha/2.  # divide alpha by 2 for two-tailed analysis
    
    for virus,ax in zip(viruses,axs):
        vidx = np.where(df.virus==virus)[0]
        # get mean REM durations for real laser-off and laser-on trials
        idx_real = np.intersect1d(vidx, np.where(df.shuffled==0)[0])
        c0_data = np.array(df[dv][np.intersect1d(idx_real, np.where(df[iv]==conditions[0])[0])])
        c1_data = np.array(df[dv][np.intersect1d(idx_real, np.where(df[iv]==conditions[1])[0])])
        
        if mode == 0:
            # get 1-$alpha confidence intervals
            c0_yerr = [np.percentile(c0_data,alpha2*100), np.percentile(c0_data,(1-alpha2)*100)]
            c1_yerr = [np.percentile(c1_data,alpha2*100), np.percentile(c1_data,(1-alpha2)*100)]
            # plot real data
            width = 0.4
            if plotType == 'bar':
                lw = 2; cap = 'projecting'
                ax.bar([0-width/2], [c0_data.mean()], color=['gray'], width=width, label=str(conditions[0]))
                ax.bar([0+width/2], [c1_data.mean()], color=['blue'], width=width, label=str(conditions[1]))
            elif plotType == 'violin':
                lw = 4; cap = 'butt'
                v = ax.violinplot([c0_data, c1_data], positions=[0-width/2, 0+width/2], widths=[width,width], showextrema=False, points=100)
                v['bodies'][0].set_facecolor('gray'); v['bodies'][1].set_facecolor('blue')
                v['bodies'][0].set_alpha(0.8); v['bodies'][1].set_alpha(0.8)
                v['bodies'][0].set_label(str(conditions[0])); v['bodies'][1].set_label(str(conditions[1]))
                ax.plot([0-width/2, 0+width/2], [c0_data.mean(), c1_data.mean()], ls='', marker='o', ms=10, mfc='white', mec='black', mew=1)
            # plot $1-alpha confidence interval
            ax.plot([0-width/2,0-width/2], c0_yerr, color='black', marker=None, lw=lw, solid_capstyle=cap)
            ax.plot([0+width/2,0+width/2], c1_yerr, color='black', marker=None, lw=lw, solid_capstyle=cap)
            ylabel = str(dv)
        elif mode == 1:
            # get mean differences in REM duration between laser-off and laser-on trials
            dif_data = c1_data - c0_data
            # get 1-$alpha confidence interval
            dif_yerr = [np.percentile(dif_data,alpha2*100), np.percentile(dif_data,(1-alpha2)*100)]
            # plot real data
            width = 0.8
            if plotType == 'bar':
                lw = 2; cap = 'projecting'
                ax.bar([0], [dif_data.mean()], color=['darkgreen'], width=width)
            elif plotType == 'violin':
                lw = 4; cap = 'butt'
                v = ax.violinplot([dif_data], positions=[0], widths=[width], showextrema=False, points=100)
                v['bodies'][0].set_facecolor('darkgreen'); v['bodies'][0].set_alpha(0.8)
                ax.plot([0], [dif_data.mean()], ls='', marker='o', ms=10, mfc='white', mec='black', mew=1)
            ax.plot([0,0], dif_yerr, color='black', marker=None, lw=lw, solid_capstyle=cap)
            ylabel = str(dv) + ' difference'
        ax.set_xticks([0])
        ax.set_xticklabels(['TRUE'])
        
        if len(set(df.shuffled)) > 1:
            # repeat for shuffled laser-off and laser-on trials
            idx_shuf = np.intersect1d(vidx, np.where(df.shuffled==1))
            c0_shuf_data = np.array(df[dv][np.intersect1d(idx_shuf, np.where(df[iv]==conditions[0])[0])])
            c1_shuf_data = np.array(df[dv][np.intersect1d(idx_shuf, np.where(df[iv]==conditions[1])[0])])
            if mode == 0:
                # get 1-$alpha confidence intervals
                c0_shuf_yerr = [np.percentile(c0_shuf_data,alpha2*100), np.percentile(c0_shuf_data,(1-alpha2)*100)]
                c1_shuf_yerr = [np.percentile(c1_shuf_data,alpha2*100), np.percentile(c1_shuf_data,(1-alpha2)*100)]
                # plot shuffled data
                if plotType == 'bar':
                    ax.bar([1-width/2], [c0_shuf_data.mean()], color=['gray'], alpha=0.5, width=width, label='shuffled ' + str(conditions[0]))
                    ax.bar([1+width/2], [c1_shuf_data.mean()], color=['blue'], alpha=0.5, width=width, label='shuffled ' + str(conditions[1]))
                elif plotType == 'violin':
                    v = ax.violinplot([c0_shuf_data, c1_shuf_data], positions=[1-width/2, 1+width/2], widths=[width,width], showextrema=False, points=100)
                    v['bodies'][0].set_facecolor('gray'); v['bodies'][1].set_facecolor('blue')
                    v['bodies'][0].set_alpha(0.4); v['bodies'][1].set_alpha(0.4)
                    v['bodies'][0].set_label('shuffled ' + str(conditions[0])); v['bodies'][1].set_label('shuffled ' + str(conditions[1]))
                    ax.plot([1-width/2, 1+width/2], [c0_shuf_data.mean(), c1_shuf_data.mean()], ls='', 
                            marker='o', ms=10, mfc='white', mec='black', mew=1)
                # plot 1-$alpha confidence intervals
                ax.plot([1-width/2,1-width/2], c0_shuf_yerr, color='black', marker=None, lw=lw, solid_capstyle=cap)
                ax.plot([1+width/2,1+width/2], c1_shuf_yerr, color='black', marker=None, lw=lw, solid_capstyle=cap)
            elif mode == 1:
                # get mean REM duration differences, 1-$alpha confidence intervals
                dif_shuf_data = c1_shuf_data - c0_shuf_data
                dif_shuf_yerr = [np.percentile(dif_shuf_data,alpha2*100), np.percentile(dif_shuf_data,(1-alpha2)*100)]
                # plot shuffled data
                if plotType == 'bar':
                    ax.bar([1], [dif_shuf_data.mean()], color=['darkgreen'], alpha=0.5, width=width)
                elif plotType == 'violin':
                    v = ax.violinplot([dif_shuf_data], positions=[1], widths=[width], showextrema=False, points=100)
                    v['bodies'][0].set_facecolor('darkgreen'); v['bodies'][0].set_alpha(0.5)
                    ax.plot([1], [dif_shuf_data.mean()], ls='', marker='o', ms=10, mfc='white', mec='black', mew=1)
                ax.plot([1,1], dif_shuf_yerr, color='black', marker=None, lw=lw, solid_capstyle=cap)
            ax.set_xticks([0,1])
            ax.set_xticklabels(['TRUE','SHUFFLED'])
        if mode==0 and ax==axs[-1]:
            ax.legend()
        ax.set_title(virus)
    
    # set y limits equal to each other
    if len(ylim) == 2:
        y=ylim
    else:
        ymin = min([ax.get_ylim()[0] for ax in axs])
        ymax = max([ax.get_ylim()[1] for ax in axs])
        y = [ymin,ymax]
    _ = [ax.set_ylim(y) for ax in axs]
    axs[0].set_ylabel(ylabel)
    #plt.show()


def compare_boot_stats(df, dv, iv, virus, shuffled, iv_val=[], mode=0, alpha=0.05,
                       grp_names = ['group1','group2'], pload=False):
    """
    Statistically compare mean bootstrapped REM duration between any two groups
    @Params
    df - dataframe with mean REM duration for each bootstrap trial
         * Columns include virus type, laser ID (0=OFF, 1=ON), and 
           (optional) shuffled status (0=real laser IDs, 1=shuffled laser IDs)
    virus - virus type for each group (e.g. ['chr2','yfp']) or both groups (e.g. 'chr2')
    shuffled - real (0) or shuffled (1) laser IDs for each group or both groups
    lsr - laser-off (0) or laser-on (1) ID for each group or both groups (if mode == 0)
    mode - if 0: compare mean REM duration during laser-off or laser-on trials ($lsr param)
                  e.g. non-shuffled laser-on trials in Chr2 mice vs. shuffled laser-on trials in Chr2 mice
           if 1: compare mean REM duration differences (laser-on - laser-off)
                  e.g. non-shuffled dur. difference in Chr2 mice vs. shuffled dur. difference in Chr2 mice
    alpha - report 1-$alpha confidence intervals
    grp_names - optional string labels for [group1, group2]
    pload - optional string specifying a filename to load the dataframe
    @Returns
    None
    """
    if os.path.isfile(pload):
        df = pd.read_pickle(pload)
    # get conditions (0 vs 1 for laser experiment, saline vs cno for DREADD experiment)
    conditions = list(set(df[iv]))
    csort = [list(df[iv]).index(c) for c in conditions]
    conditions = [c for _,c in sorted(zip(csort,conditions))]
    # clean data inputs
    if mode == 1:
        iv_val = [conditions[0], conditions[0]]
    conds = [virus, iv_val, shuffled]
    for i,c in enumerate(conds):
        if type(c) in [str,int,float]:
            conds[i] = [c,c]
        elif type(c)==list:
            if len(c)==1:
                conds[i] = [c[0],c[0]]
            elif len(c)==2:
                conds[i] = c
            else:
                print('Invalid list length for parameter ' + f'{cond=}'.split('=')[0])
                #return
        else:
            print('Invalid data type for parameter ' + f'{cond=}'.split('=')[0])
            return
    virus, iv_val, shuffled = conds
    if 'shuffled' not in df.columns:
        df['shuffled'] = 0
        if 1 in shuffled:
            print('ERROR: no shuffled laser IDs found in dataframe')
            return
    
    # get indices matching criteria for virus type and shuffled designation
    vidx1, vidx2 = [np.where(df.virus==v)[0] for v in virus]
    shidx1, shidx2 = [np.where(df.shuffled==sh)[0] for sh in shuffled]
    
    if mode == 0:
        cidx1, cidx2 = [np.where(df[iv]==c)[0] for c in iv_val]
        # get indices for comparison groups 1 and 2
        idx1 = np.intersect1d(np.intersect1d(vidx1,cidx1),shidx1)
        idx2 = np.intersect1d(np.intersect1d(vidx2,cidx2),shidx2)
        if len(idx1) != len(idx2):
            print('Uh oh, two groups have different number of trials')
            return
        else:
            nboots = len(idx1)
        grp1_data = np.array(df[dv][idx1])
        grp2_data = np.array(df[dv][idx2])
        
    elif mode == 1:
        # get laser-off and laser-on indices for comparison groups 1 and 2
        idx1 = np.intersect1d(vidx1,shidx1)
        c0_1, c1_1 = [np.where(df[iv][idx1]==conditions[0])[0], np.where(df[iv][idx1]==conditions[1])[0]]
        idx2 = np.intersect1d(vidx2,shidx2)
        c0_2, c1_2 = [np.where(df[iv][idx2]==conditions[0])[0], np.where(df[iv][idx2]==conditions[1])[0]]
        if len(idx1) != len(idx2):
            print('Uh oh, two groups have different number of trials')
            return
        elif len(c0_1) != len(c1_1) or len(c0_2) != len(c1_2):
            print('Uh oh, at least one group has a different number of trials in each condition')
            return
        else:
            nboots = len(c0_1)
        grp1_data = np.array(df[dv][idx1])[c1_1] - np.array(df[dv][idx1])[c0_1]
        grp2_data = np.array(df[dv][idx2])[c1_2] - np.array(df[dv][idx2])[c0_2]
        
    # get mean REM duration/difference
    grp1_mean = grp1_data.mean()
    grp1_yerr = [np.percentile(grp1_data,(alpha/2.)*100), np.percentile(grp1_data,(1-alpha/2.)*100)]
    grp2_mean = grp2_data.mean()
    grp2_yerr = [np.percentile(grp2_data,(alpha/2.)*100), np.percentile(grp2_data,(1-alpha/2.)*100)]
    # get difference between mean REM duration/difference between groups, for each bootstrap trial
    difs = grp1_data - grp2_data
    # get no. trials where group1 mean is greater than (p1) or less than/equal to (p2) the group2 mean
    # divide by total no. trials to get probability of group1 REM being longer/shorter than group2
    p1 = len(np.where(difs > 0)[0]) / nboots
    p2 = len(np.where(difs <= 0)[0]) / nboots
    # the p-value is the probability of the less likely condition, multiplied by 2 for two-tailed test 
    pval = 2 * np.min([p1, p2])
    
    # print stats
    txt1 = f'{grp_names[0]} --> mean{" difference " if mode==1 else " "}= {round(grp1_mean,3)}' \
           f', CI = [{round(grp1_yerr[0],3)}, {round(grp1_yerr[1],3)}]'
    txt2 = f'{grp_names[1]} --> mean{" difference " if mode==1 else " "}= {round(grp2_mean,3)}' \
           f', CI = [{round(grp2_yerr[0],3)}, {round(grp2_yerr[1],3)}]'
    print('')
    print(txt1)
    print(txt2)
    print(f'P-value of difference = {round(pval,5)}')
    print('')
    
    return grp1_data, grp2_data, pval

def rem_duration(ppath, recordings, tstart=0, tend=-1):
    """
    Get duration of each REM period in dataset
    ppath - base folder
    recordings - list of recordings
    """
    df = pd.DataFrame(columns=['mouse', 'recording', 'dur'])
    for rec in recordings:
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load brain state annotation, get REM sequences
        M = sleepy.load_stateidx(ppath, rec)[0]
        istart = tstart/dt
        iend = len(M) if tend==-1 else tend/dt
        sseq = sleepy.get_sequences(np.where(M==1)[0])
        sseq = [seq for seq in sseq if seq[0] >= istart and seq[-1] < iend]
        # calculate REM sleep durations
        dur = [len(seq)*dt for seq in sseq]
        df = pd.concat([df, pd.DataFrame({'mouse':rec.split('_')[0], 
                                          'recording':rec, 
                                          'dur':dur})], axis=0, ignore_index=True)
    return df


def compare_online_analysis(ppath, ctr_rec, exp_rec, istate, stat, overlap=0, 
                            ma_thr=20, ma_state=3, flatten_is=False, mouse_avg='mouse', 
                            group_colors=[], ylim=[], pplot=True, print_stats=True):
    """
    Compare overall brain state between control and experimental mouse groups from closed-loop experiments
    @Params
    ppath - base folder
    ctr_rec - list of control recordings
    exp_rec - list of experimental recordings
    istate - brain state to analyze
    stat - statistic to compare
           'perc' - total percent time spent in brain state
           'dur' - mean overall brain state duration (across laser-on + laser-off bouts)
    overlap - float between 0 and 100, specifying minimum percentage of overlap
              between detected (online) and annotated (offline) brain states
              required to include episode in analysis
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse', 'recording', or 'trial'
    group_colors - optional 2-element list of colors for control and experimental groups
    ylim - set y axis limit for bar plot
    @Returns
    None
    """
    # clean data inputs
    if type(ctr_rec) != list:
        ctr_rec = [ctr_rec]
    if type(exp_rec) != list:
        exp_rec = [exp_rec]
    if len(group_colors) != 2:
        group_colors = ['gray', 'blue']
        
    # get control and experimental mouse names
    cmice = dict()
    for crec in ctr_rec:
        idf = re.split('_', crec)[0]
        if not idf in cmice:
            cmice[idf] = 1
    cmice = list(cmice.keys())
    emice = dict()
    for erec in exp_rec:
        idf = re.split('_', erec)[0]
        if not idf in emice:
            emice[idf] = 1
    emice = list(emice.keys())
    
    # collect list of $stat values for each control and experimental recording
    cdict = {crec:[] for crec in ctr_rec}
    edict = {erec:[] for erec in exp_rec}
    
    for rec in ctr_rec + exp_rec:
        if rec == ctr_rec[0]:
            print('\n### GETTING DATA FROM CONTROL MICE ###\n')
        elif rec == exp_rec[0]:
            print('\n### GETTING DATA FROM EXPERIMENTAL MICE ###\n')
        print('Analyzing ' + rec + ' ... ')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
        
        # get total % time spent in brain state
        if stat == 'perc':
            data = [(len(np.where(M==istate)[0]) / len(M)) * 100]
        # get overall brain state duration
        elif stat == 'dur':
            data = []
            # load laser and REM detection, downsample to SP time
            laser = sleepy.load_laser(ppath, rec)
            rem_trig = so.loadmat(os.path.join(ppath, rec, 'rem_trig_%s.mat'%rec), 
                                  squeeze_me=True)['rem_trig']
            laser = downsample_vec(laser, nbin)
            laser[np.where(laser>0)] = 1
            rem_trig = downsample_vec(rem_trig, nbin)
            rem_trig[np.where(rem_trig>0)] = 1
            laser_idx = np.where(laser==1)[0]
            rem_idx = np.where(rem_trig==1)[0]
            # get brain state sequences from offline analysis
            seq = sleepy.get_sequences(np.where(M==istate)[0])
            for s in seq:
                isect = np.intersect1d(s, rem_idx)
                # check overlap between online & offline brain state sequences, collect state duration
                if len(np.intersect1d(s, rem_idx)) > 0 and float(len(isect)) / len(s) >= overlap:
                    data.append((s[-1]-s[0]+1)*dt)
        if rec in ctr_rec:
            cdict[rec] = data
        elif rec in exp_rec:
            edict[rec] = data

    # create dataframes with $stat value for each mouse, recording, or trial
    cdf = pwaves.df_from_rec_dict(cdict, stat); cdf['group'] = 'control'
    edf = pwaves.df_from_rec_dict(edict, stat); edf['group'] = 'exp'
    if mouse_avg in ['mouse', 'recording']:
        cdf = cdf.groupby(mouse_avg, as_index=False)[stat].mean(); cdf['group'] = 'control'
        edf = edf.groupby(mouse_avg, as_index=False)[stat].mean(); edf['group'] = 'exp'
    df = pd.concat([cdf,edf], axis=0)
    
    if pplot:
        # plot bar graph comparing control and experimental mice
        plt.figure()
        sns.barplot(x='group', y=stat, data=df, ci=68, palette={'control':group_colors[0], 
                                                                'exp':group_colors[1]})
        if 'trial' in mouse_avg:
            sns.swarmplot(x='group', y=stat, data=df, palette={'control':group_colors[0], 
                                                                'exp':group_colors[1]})
        elif mouse_avg in ['mouse', 'recording']:
            sns.swarmplot(x='group', y=stat, data=df, color='black', size=8)
        if len(ylim) == 2:
            plt.ylim(ylim)
        if stat == 'perc':
            plt.ylabel('Percent time spent (%)')
        elif stat == 'dur':
            plt.ylabel('Duration (s)')
        plt.title(f'Control vs exp mice - statistic={stat}, state={istate}')
    
    if print_stats:
        # get single vectors of data
        cdata, clabels = pwaves.mx1d(cdict, mouse_avg)
        edata, elabels = pwaves.mx1d(edict, mouse_avg)
        
        # stats - unpaired t-test comparing control & experimental mice
        p = scipy.stats.ttest_ind(np.array((cdata)), np.array((edata)), nan_policy='omit')
        sig='yes' if p.pvalue < 0.05 else 'no'
        dof = len(cmice) + len(emice)
        print('')
        print(f'ctr vs exp, stat={stat}  -- T={round(p.statistic,3)}, DOF={dof}, p-value={round(p.pvalue,3)}, sig={sig}')
        print('')
    return df
    

def avg_sp_transitions(ppath, recordings, transitions, pre, post, si_threshold, sj_threshold, 
                       laser=0, bands=[(0.5,4), (6,10), (11,15), (55,99)], band_labels=[],
                       band_colors=[], tstart=0, tend=-1, fmax=30, pnorm=1, psmooth=0, vm=[], ma_thr=20, 
                       ma_state=3, flatten_is=False, mouse_avg='mouse', sf=0, offset=0):
    """
    Plot average spectrogram and frequency band power at brain state transitions (absolute time)
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
    laser - if True, separate transitions into spontaneous vs laser-triggered
            if False - plot all state transitions
    bands - list of tuples with min and max frequencies in each power band
            e.g. [ [0.5,4], [6,10], [11,15], [55,100] ]
    band_labels - optional list of descriptive names for each freq band
            e.g. ['delta', 'theta', 'sigma', 'gamma']
    band_colors - optional list of colors to plot each freq band
            e.g. ['firebrick', 'limegreen', 'cyan', 'purple']
    tstart, tend - time (s) into recording to start and stop collecting data
    fmax - maximum frequency in spectrogram
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    vm - 2-element list controlling saturation for [SP1, SP2]
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    sf - smoothing factor for vectors of frequency band power
    offset - shift (s) of laser time points, as control
    @Returns
    pwrband_dicts - list of tuples with EEG power dictionaries (spon_PwrBands, lsr_PwrBands) for each transition
                    * if $laser == True  : spon_PwrBands and lsr_PwrBands dictionaries (keys = freq bands, 
                                           values = arrays of trials/mice x time bins) contains data for spontaneous
                                           and laser-triggered transitions, respectively
                    * if $laser == False : spon_PwrBands contains data for all transitions, lsr_PwrBands is empty
    labels - list of tuples with trial #s/mouse names (spon_labels, lsr_labels), corresponding with rows in data arrays
    t - list of time points, corresponding with columns in data arrays
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if len(vm) == 2:
        if type(vm[0]) in [int, float]:
            vm = [[vm], [vm]]
    else:
        vm = [[],[]]

    states = {1:'R', 2:'W', 3:'N', 4:'tN', 5:'ftN', 6:'MA'}
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    
    # create data dictionaries to collect spontaneous & laser-triggered transitions
    spon_sp = {states[si] + states[sj] : [] for (si,sj) in transitions}
    if laser:
        lsr_sp = {states[si] + states[sj] : [] for (si,sj) in transitions}
    
    for (si,sj) in transitions:
        print('')
        print(f'NOW COLLECTING INFORMATION FOR {states[si]}{states[sj]} TRANSITIONS ...' )
        print('')
        
        # collect data for each transition
        sid = states[si] + states[sj]
        spon_sp_rec_dict = {rec:[] for rec in recordings}
        if laser:
            lsr_sp_rec_dict = {rec:[] for rec in recordings}
        
        for rec in recordings:
            print("Getting spectrogram for", rec, "...")
            
            # load sampling rate
            sr = sleepy.get_snr(ppath, rec)
            nbin = int(np.round(sr)*2.5)
            dt = (1.0 / sr)*nbin
            
            # load and adjust brain state annotation
            M, _ = sleepy.load_stateidx(ppath, rec)
            M = adjust_brainstate(M, dt, ma_thr, ma_state, flatten_is)
            
            # load laser
            if laser:
                lsr_raw = sleepy.load_laser(ppath, rec)
                lsr_s, lsr_e = sleepy.laser_start_end(lsr_raw, sr, offset=offset)
                lsr = np.zeros((len(lsr_raw),))
                # remove pulse info
                for i, j in zip(lsr_s, lsr_e):
                    lsr[i:j] = 1
            
            # load and normalize spectrogram
            P = so.loadmat(os.path.join(ppath, rec,   'sp_' + rec + '.mat'), squeeze_me=True)
            SP = P['SP']
            f = P['freq']
            ifreq = np.where(f <= fmax)[0]
            if pnorm:
                sp_mean = SP.mean(axis=1)
                SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            
            # define start and end points of analysis
            istart = int(np.round((1.0*tstart) / dt))
            if tend == -1: iend = len(M)
            else: iend = int(np.round((1.0*tend) / dt))
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
                    
                    # if si and sj meet duration criteria, collect SP
                    if ipre <= ti < len(M)-ipost and len(s)*dt >= si_threshold[si-1]:
                        if len(sj_idx)*dt >= sj_threshold[sj-1] and istart <= ti < iend:
                            sp_si = SP[:, ti-ipre+1 : ti+1]
                            sp_sj = SP[:, ti+1 : ti+ipost+1]
                            sp_trans = np.concatenate((sp_si, sp_sj), axis=1)
                            # if $laser=1, save as either laser-triggered or spontaneous transition
                            if laser:
                                if lsr[int((ti+1)*sr*2.5)] == 1:
                                    lsr_sp_rec_dict[rec].append(sp_trans)
                                else:
                                    spon_sp_rec_dict[rec].append(sp_trans)
                            # if $laser=0, save as spontaneous transition
                            else:
                                spon_sp_rec_dict[rec].append(sp_trans)
        spon_sp[sid] = spon_sp_rec_dict
        if laser:
            lsr_sp[sid] = lsr_sp_rec_dict
    
    pwrband_dicts = []
    labels = []
    # get frequency band power
    for (si,sj) in transitions:
        sid = states[si]+states[sj]
        # create 3D data matrix for SPs (freq x time bins x subject)
        spon_sp_mx, spon_labels = pwaves.mx3d(spon_sp[sid], mouse_avg)
        # create dictionary for freq band power (key=freq band, value=matrix of subject x time bins)
        spon_PwrBands = {b : np.zeros((spon_sp_mx.shape[2], spon_sp_mx.shape[1])) for b in bands}
        for layer in range(spon_sp_mx.shape[2]):
            trial_sp = spon_sp_mx[:,:,layer]
            # get mean power of each freq band from SP
            for b in bands:
                bfreq = np.intersect1d(np.where(f >= b[0])[0], np.where(f <= b[1])[0])
                band_mean = np.nanmean(trial_sp[bfreq, :], axis=0)
                if sf > 0:
                    band_mean = convolve_data(band_mean, sf)
                spon_PwrBands[b][layer, :] = band_mean
        # average/adjust spectrogram
        spon_sp_plot = adjust_spectrogram(np.nanmean(spon_sp_mx, axis=2), pnorm=0, 
                                          psmooth=psmooth, freq=f, fmax=fmax)
        # collect laser-triggered transitions
        if laser:
            lsr_sp_mx, lsr_labels = pwaves.mx3d(lsr_sp[sid], mouse_avg)
            lsr_PwrBands = {b : np.zeros((lsr_sp_mx.shape[2], lsr_sp_mx.shape[1])) for b in bands}
            for layer in range(lsr_sp_mx.shape[2]):
                trial_sp = lsr_sp_mx[:,:,layer]
                # get mean power of each freq band from SP
                for b in bands:
                    bfreq = np.intersect1d(np.where(f >= b[0])[0], np.where(f <= b[1])[0])
                    band_mean = np.nanmean(trial_sp[bfreq, :], axis=0)
                    if sf > 0:
                        band_mean = convolve_data(band_mean, sf)
                    lsr_PwrBands[b][layer, :] = band_mean
            # average/adjust spectrogram
            lsr_sp_plot = adjust_spectrogram(np.nanmean(lsr_sp_mx, axis=2), False, psmooth, f, fmax)
        else:
            lsr_PwrBands = {}
            lsr_labels = []

        t = np.linspace(-pre, post, spon_sp_plot.shape[1])
        freq = f[ifreq]
        
        ###   GRAPHS   ###
        plt.ion()
        if laser:
            fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
            [ax1,ax3,ax2,ax4] = axs.reshape(-1)
            ax1_title = 'Spontaneous Transitions'
        else:
            fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
            ax1_title = ''
        fig.suptitle(sid + ' TRANSITIONS')
        
        # plot spectrogram for spontaneous transitions
        im = ax1.pcolorfast(t, freq, spon_sp_plot, cmap='jet')
        if len(vm[0]) == 2:
            im.set_clim(vm[0])
        cbar = plt.colorbar(im, ax=ax1, pad=0.0)
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # set axis limits
        ax1.set_xlim((t[0], t[-1]))
        ax1.set_xticklabels([])
        ax1.set_ylabel('Freq. (Hz)')
        ax1.set_title(ax1_title)
        # plot mean frequency band power
        for b,l,c in zip(bands, band_labels, band_colors):
            data = spon_PwrBands[b].mean(axis=0)
            yerr = spon_PwrBands[b].std(axis=0) / np.sqrt(spon_PwrBands[b].shape[0])
            ax2.plot(t, data, color=c, label=l)
            ax2.fill_between(t, data-yerr, data+yerr, color=c, alpha=0.3)
        ax2.set_xlim((t[0], t[-1]))
        ax2.set_xlabel('Time (s)')
        if pnorm > 0:
            ax2.set_ylabel('Rel. Power')
        else:
            ax2.set_ylabel('Avg. band power (uV^2)')
        ax2.legend()
        
        # plot spectrogram for laser-triggered transitions
        if laser:
            im = ax3.pcolorfast(t, freq, lsr_sp_plot, cmap='jet')
            if len(vm[1]) == 2:
                im.set_clim(vm[1])
            cbar = plt.colorbar(im, ax=ax3, pad=0.0)
            if pnorm >0:
                cbar.set_label('Rel. Power')
            else:
                cbar.set_label('Power uV^2s')
            # set axis limits
            ax3.set_xlim((t[0], t[-1]))
            ax3.set_xticklabels([])
            ax3.set_ylabel('Freq. (Hz)')
            ax3.set_title('Laser-Triggered Transitions')
            # plot mean frequency band power
            for b,l,c in zip(bands, band_labels, band_colors):
                data = lsr_PwrBands[b].mean(axis=0)
                yerr = lsr_PwrBands[b].std(axis=0) / np.sqrt(lsr_PwrBands[b].shape[0])
                ax4.plot(t, data, color=c, label=l)
                ax4.fill_between(t, data-yerr, data+yerr, color=c, alpha=0.3)
            ax4.set_xlim((t[0], t[-1]))
            ax4.set_xlabel('Time (s)')
            if pnorm > 0: ax4.set_ylabel('Rel. Power')
            else: ax4.set_ylabel('Avg. band power (uV^2)')
            ax4.legend()
            # set equal y axis limits
            y = (min([ax2.get_ylim()[0], ax4.get_ylim()[0]]), max([ax2.get_ylim()[1], ax4.get_ylim()[1]]))
            ax2.set_ylim(y); ax4.set_ylim(y)
        
        pwrband_dicts.append((spon_PwrBands, lsr_PwrBands))
        labels.append((spon_labels, lsr_labels))
        
    plt.show()
    
    return pwrband_dicts, labels, t


def sleep_spectrum_simple(ppath, recordings, istate=1, pnorm=0, pmode=1, fmax=30, tstart=0, tend=-1,  
                          ma_thr=20, ma_state=3, flatten_is=False, noise_state=0, mu=[10,100], ci='sd', 
                          harmcs=0, pemg2=False, exclusive_mode=False, pplot=True, ylims=[]):
    """
    Get EEG power spectrum using pre-calculated spectogram saved in sp_"name".mat file
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    pmode - method for analyzing laser
            0 - plot all state episodes regardless of laser
            1 - compare states during laser vs. baseline outside laser interval
    fmax - maximum frequency in power spectrum
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    noise_state - brain state to assign manually annotated regions of EEG noise
                  (if 0, do not analyze)
    mu - [min,max] frequencies summed to get EMG amplitude
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                          integer between 0 and 100=confidence interval)
    harmcs - if > 0, interpolate harmonics of base frequency $harmcs
    pemg2 - if True, use EMG2 for EMG amplitude calcuation
    exclusive_mode - if True, isolate portions of brain state episodes with laser as "laser ON"
                     if False, consider brain state episodes with any laser overlap as "laser ON"
    pplot - if True, show plots
    ylims - optional list of y axis limits for each brain state plot
    @Returns
    ps_mx - data dictionary (key=laser state, value=power value matrix of mice x frequencies)
    freq - list of frequencies, corresponding to columns in $ps_mx arrays
    df - dataframe with EEG power spectrums
    df_amp - dataframe with EMG amplitudes
    """
    def _interpolate_harmonics(SP, freq, fmax, harmcs, iplt_level):
        """
        Interpolate harmonics of base frequency $harmcs by averaging across 3-5 
        surrounding frequencies
        """
        df = freq[2]-freq[1]
        for h in np.arange(harmcs, fmax, harmcs):
            i = np.argmin(np.abs(np.round(freq,1) - h))
            if np.abs(freq[i] - h) < df/2 and h != 60: 
                if iplt_level == 2:
                    SP[i,:] = (SP[i-2:i,:] + SP[i+1:i+3,:]).mean(axis=0) * 0.5
                elif iplt_level == 1:
                    SP[i,:] = (SP[i-1,:] + SP[i+1,:]) * 0.5
                else:
                    pass
        return SP
    
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # clean data inputs
    if type(istate) != list:
        istate=[istate]
    if len(ylims) != len(istate):
        ylims = [[]]*len(istate)
    
    mice = []
    # get unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice.append(idf)
    
    # create data dictionaries to store mouse-averaged power spectrums
    ps_mice = {s: {0:{m:[] for m in mice}, 1:{m:[] for m in mice} } for s in istate}
    amp_mice = {s: {0:{m:0 for m in mice}, 1:{m:0 for m in mice} } for s in istate}
    count_mice = {s: {0:{m:0 for m in mice}, 1:{m:0 for m in mice} } for s in istate}
    
    data = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        print('Getting data for ' + rec + ' ...')
        emg_loaded =  False
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is, 
                              noise_state=noise_state)
        
        # define start and end points of analysis
        istart = int(np.round(tstart / dt))
        if tend > -1:
            iend = int(np.round(tend / dt))
        else:
            iend = len(M)
        istart_eeg = istart*nbin
        iend_eeg   = iend*nbin
        M = M[istart:iend]
        
        # load/normalize EEG spectrogram
        tmp = so.loadmat(os.path.join(ppath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = tmp['SP'][:,istart:iend]
        if pnorm:
            sp_mean = np.mean(SP, axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        freq = tmp['freq']
        df = freq[1]-freq[0]
        if fmax > -1:
            ifreq = np.where(freq <= fmax)[0]
            freq = freq[ifreq]
            SP = SP[ifreq,:]
        
        # load EMG spectrogram
        tmp = so.loadmat(os.path.join(ppath, rec, 'msp_%s.mat' % rec), squeeze_me=True)
        if not pemg2:
            MSP = tmp['mSP'][:,istart:iend]
            freq_emg = tmp['freq']
        else:
            MSP = tmp['mSP2'][:,istart:iend]
        imu = np.where((freq_emg>=mu[0]) & (freq_emg<=mu[-1]))[0]
        # remove harmonic frequencies
        if harmcs > 0:
            harm_freq = np.arange(0, freq_emg.max(), harmcs)
            for h in harm_freq:
                imu = np.setdiff1d(imu, imu[np.where(np.round(freq_emg[imu], decimals=1)==h)[0]])
            tmp = 0
            for i in imu:
                tmp += MSP[i,:] * (freq_emg[i]-freq_emg[i-1])
            emg_ampl = np.sqrt(tmp)            
        else:
            emg_ampl = np.sqrt(MSP[imu,:].sum(axis=0)*df)

        # load laser, downsample to SP time
        if pmode == 1:
            lsr = sleepy.load_laser(ppath, rec)
            idxs, idxe = sleepy.laser_start_end(lsr[istart_eeg:iend_eeg])
            # downsample laser
            idxs = [int(i/nbin) for i in idxs]
            idxe = [int(i/nbin) for i in idxe]
            lsr_vec = np.zeros((len(M),))
            for (i,j) in zip(idxs, idxe):
                lsr_vec[i:j+1] = 1
            lsr_vec = lsr_vec[istart:iend]
            laser_idx = np.where(lsr_vec==1)[0]
        
        for state in istate:
            idx = np.where(M==state)[0]
            # get indices of laser ON and laser OFF brain state episodes
            if pmode == 1:
                idx_lsr   = np.intersect1d(idx, laser_idx)
                idx_nolsr = np.setdiff1d(idx, laser_idx)
                # eliminate bins without laser in each episode
                if exclusive_mode == True:
                    rm_idx = []
                    state_seq = sleepy.get_sequences(np.where(M==state)[0])
                    for s in state_seq:
                        d = np.intersect1d(s, idx_lsr)
                        if len(d) > 0:
                            drm = np.setdiff1d(s, d)
                            rm_idx.append(drm)
                            idx_nolsr = np.setdiff1d(idx_nolsr, drm)
                            idx_lsr = np.union1d(idx_lsr, drm)
                # get no. of laser ON and laser OFF episodes
                count_mice[state][0][idf] += len(idx_nolsr)
                count_mice[state][1][idf] += len(idx_lsr)
                # collect summed SPs & EMG amplitudes
                ps_lsr   = SP[:,idx_lsr].sum(axis=1)
                ps_nolsr = SP[:,idx_nolsr].sum(axis=1)
                ps_mice[state][1][idf].append(ps_lsr)
                ps_mice[state][0][idf].append(ps_nolsr)
                amp_mice[state][1][idf] += emg_ampl[idx_lsr].sum()
                amp_mice[state][0][idf] += emg_ampl[idx_nolsr].sum()
            # collect all brain state episodes, regardless of laser
            else:
                count_mice[state][0][idf] += len(idx)
                ps_nolsr = SP[:,idx].sum(axis=1)
                ps_mice[state][0][idf].append(ps_nolsr)
                amp_mice[state][0][idf] += emg_ampl[idx].sum()
    lsr_cond = []
    if pmode == 0:
        lsr_cond = [0]
    else:
        lsr_cond = [0,1]
    
    # create dataframes for EEG power spectrums and EMG amplitudes
    df = pd.DataFrame(columns=['mouse', 'freq', 'pow', 'lsr', 'state'])
    df_amp = pd.DataFrame(columns=['mouse', 'amp', 'lsr', 'state'])
    for state, y in zip(istate, ylims):
        ps_mx  = {0:[], 1:[]}
        amp_mx = {0:[], 1:[]}
        for l in lsr_cond:
            mx  = np.zeros((len(mice), len(freq)))
            amp = np.zeros((len(mice),))
            # get mouse-averaged data
            for (i,idf) in zip(range(len(mice)), mice):
                mx[i,:] = np.array(ps_mice[state][l][idf]).sum(axis=0) / count_mice[state][l][idf]
                amp[i]  = amp_mice[state][l][idf] / count_mice[state][l][idf]
            ps_mx[l]  = mx
            amp_mx[l] = amp
        # transform data arrays to store in dataframes
        data_nolsr = list(np.reshape(ps_mx[0], (len(mice)*len(freq),)))
        amp_freq = list(freq)*len(mice)
        amp_idf = reduce(lambda x,y: x+y, [[b]*len(freq) for b in mice])
        if pmode == 1:
            data_lsr = list(np.reshape(ps_mx[1], (len(mice)*len(freq),)))
            list_lsr = ['yes']*len(freq)*len(mice) + ['no']*len(freq)*len(mice)
            data = [[a,b,c,d] for (a,b,c,d) in zip(amp_idf*2, amp_freq*2, data_lsr+data_nolsr, list_lsr)]
        else:
            list_lsr = ['no']*len(freq)*len(mice)
            data = [[a,b,c,d] for (a,b,c,d) in zip(amp_idf, amp_freq, data_nolsr, list_lsr)]
        sdf = pd.DataFrame(columns=['mouse', 'freq', 'pow', 'lsr'], data=data)
        # store EMG amplitudes
        sdf_amp = pd.DataFrame(columns=['mouse', 'amp', 'lsr'])
        if pmode == 1:
            sdf_amp['mouse'] = mice*2
            sdf_amp['amp'] = list(amp_mx[0]) + list(amp_mx[1])
            sdf_amp['lsr'] = ['no'] * len(mice) + ['yes'] * len(mice)
        else:
            sdf_amp['mouse'] = mice
            sdf_amp['amp'] = list(amp_mx[0]) 
            sdf_amp['lsr'] = ['no'] * len(mice) 
        sdf['state'] = state
        sdf_amp['state'] = state
        df = pd.concat([df,sdf], axis=0, ignore_index=True)
        df_amp = pd.concat([df_amp,sdf_amp], axis=0, ignore_index=True)
            
        # plot power spectrum(s)
        if pplot:
            plt.ion()
            plt.figure()
            sns.set_style('ticks')
            sns.lineplot(data=sdf, x='freq', y='pow', hue='lsr', ci=ci, 
                         palette={'yes':'blue', 'no':'gray'})
            sns.despine()
            # set axis limits and labels
            plt.xlim([freq[0], freq[-1]])
            plt.xlabel('Freq. (Hz)')
            if not pnorm:    
                plt.ylabel('Power ($\mathrm{\mu V^2}$)')
            else:
                plt.ylabel('Norm. Pow.')
            if len(y) == 2:
                plt.ylim(y)
            plt.title(f'Power spectral density during {state}')
            plt.show()
    return ps_mx, freq, df, df_amp


def compare_power_spectrums(ppath, rec_list, cond_list, istate, pnorm=0, pmode=0, fmax=30, 
                            tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_is=4, 
                            noise_state=0, exclusive_mode=False, colors=[], ylims=[]):
    """
    Plot average power spectrums for any brain state; compare between multiple groups of mice
    @Params
    ppath - base folder
    rec_list - list of lists; each sub-list contains recording folders for one mouse group
    cond_list - list of labels for each group
    istate - brain state(s) to analyze
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    pmode - method for analyzing laser
            0 - plot all state episodes regardless of laser
            1 - compare states during laser vs. baseline outside laser interval
    fmax - maximum frequency in power spectrum
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    noise_state - brain state to assign manually annotated regions of EEG noise
                  (if 0, do not analyze)
    exclusive_mode - if True, isolate portions of brain state episodes with laser as "laser ON"
                     if False, consider brain state episodes with any laser overlap as "laser ON"
    colors - optional list of colors for each group
    ylims - optional list of y axis limits for each brain state plot
    @Returns
    None
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'Microarousals'}
    if flatten_is == 4:
        states[4] = 'IS'
    
    # clean data inputs
    if len(cond_list) != len(rec_list):
        cond_list = ['group ' + str(i) for i in np.arange(1,len(rec_list)+1)]
    if len(colors) != len(cond_list):
        colors = colorcode_mice([], return_colorlist=True)[0:len(rec_list)]
    pal = {cond:col for cond,col in zip(cond_list, colors)}
    if len(ylims) != len(istate):
        ylims = [[]]*len(istate)
    # create dataframe of frequency power values, with mouse/group/brain state/laser info
    dfs = []
    for recordings, condition in zip(rec_list, cond_list):
        # calculate brain state power spectrums for each mouse group
        grp_df = sleep_spectrum_simple(ppath, recordings, istate, pmode=pmode, pnorm=pnorm, fmax=fmax,
                                       tstart=tstart, tend=tend, ma_thr=ma_thr, ma_state=ma_state, 
                                       flatten_is=flatten_is, noise_state=noise_state, pplot=False)[2]
        grp_df['cond'] = condition
        dfs.append(grp_df)
    df = pd.concat(dfs, axis=0)
    
    # compare group power spectrums for each brain state
    for s,y in zip(istate, ylims):
        sdf = df.iloc[np.where(df['state']==s)[0], :]
        plt.ion()
        plt.figure()
        sns.set_style('ticks')
        sns.lineplot(data=sdf, x='freq', y='pow', hue='cond', ci='sd', palette=pal)
        sns.despine()
        plt.xlabel('Freq. (Hz)')
        if not pnorm:    
            plt.ylabel('Power ($\mathrm{\mu V^2}$)')
        else:
            plt.ylabel('Norm. Pow.')
        if len(y) == 2:
            plt.ylim(y)
        plt.title(states[s])
        plt.show()
        
    
    
#################            PLOTTING FUNCTIONS            #################

def hypno_colormap():
    """
    Create colormap for Weber lab sleep annotations
    @Params
    None
    @Returns
    my_map - colormap for brain state annotations
    vmin - minimum brain state value
    vmax - maximum brain state value
    """
    # assign each brain state to a color
    state_names = ['Noise', 'REM', 'Wake', 'NREM', 'IS', 'IS-R', 'IS-W', 'MA']
    state_colors = ['black', 'cyan', 'darkviolet', 'darkgray', 'darkblue', 'darkblue', 'red', 'magenta']
    rgb_colors = [matplotlib.colors.to_rgba(sc) for sc in state_colors]
    
    # create colormap
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', rgb_colors, len(rgb_colors))
    vmin = 0
    vmax = 7

    return my_map, vmin, vmax


def colorcode_mice(names, return_colorlist=False):   
    """
    Load .txt file with mouse/brain state names and associated colors
    @Params
    names - list of mouse or brain state names
    return_colorlist - if True, return list of 20 pre-chosen colors
    @Returns
    colors - dictionary (key=mouse/state name, value=color) OR list of 20 colors
    """
    # 20 colors, maximally distinguishable from each other
    colorlist = ['red', 'green', 'blue', 'black', 'orange', 'fuchsia', 'yellow', 'brown', 
                 'pink', 'dodgerblue', 'chocolate', 'turquoise', 'darkviolet', 'lime', 
                 'skyblue', 'lightgray', 'darkgreen', 'yellowgreen', 'maroon', 'gray']
    if return_colorlist:
        colors = colorlist
        return colors
    
    # load txt file of mouse/brain state names and assigned colors
    colorpath = '/home/fearthekraken/Documents/Data/sleepRec_processed/mouse_colors.txt'
    f = open(colorpath, newline=None)
    lines = f.readlines()
    f.close()
    # create dictionary with mouse/brain state names as keys
    if type(names) != list:
        names = [names]
    names = [n.split('_')[0] for n in names]
    colors = {n:'' for n in names}
    
    # match mouse/brain state name to paired color in txt file
    for l in lines:
        mouse_name = re.split('\s+', l)[0]
        assigned_mouse = [n for n in names if n.lower() == mouse_name.lower()]
        if len(assigned_mouse) > 0:
            colors[assigned_mouse[0]] = re.split('\s+', l)[1]
    # assign colors to mice/states not in txt file
    unassigned_mice = [name for name in names if colors[name]=='']
    unassigned_colors =  [color for color in colorlist if color not in list(colors.values())]
    for i, um in enumerate(unassigned_mice):
        colors[um] = unassigned_colors[i]
    
    return colors

def filter_signal(data, sr, f1, f2=None):
    if f2 is None:
        # notch filter
        w0 = f1 / (sr/2)
        filt = sleepy.my_notchfilter(data, sr=sr, band=170, freq=f1)
    else:
        if f1 > -1 and f2 > -1:
            # bandpass filter
            w0 = f1 / (sr/2)
            w1 = f2 / (sr/2)
            filt = sleepy.my_bpfilter(data, w0, w1)
        elif f1 > -1:
            # highpass filter
            w0 = f1 / (sr/2)
            filt = sleepy.my_hpfilter(data, w0)
            print('highpass!')
        elif f2 > -1:
            # lowpass filter
            w0 = f2 / (sr/2)
            filt = sleepy.my_lpfilter(data, w0)
            print('lowpass!')
        else:
            return
    return filt


def plot_example(ppath, rec, PLOT, tstart, tend, ma_thr=20, ma_state=3, flatten_is=False,
                 eeg_nbin=1, eeg_filt=[], emg_nbin=1, emg_filt=[], lfp_nbin=17, dff_nbin=250, highres=False,
                 recalc_highres=True, nsr_seg=2.5, perc_overlap=0.8, pnorm=0, psmooth=0,
                 fmax=30, vm=[], cmap='jet', ylims=[], add_boxes=[], return_data=False):
    """
    Plot any combination of available signals on the same time scale
    @Params
    ppath - base folder
    rec - recording folder
    PLOT - list of signals to be plotted
           'HYPNO'               - brain state annotation
           'SP', 'SP2'           - hippocampal or prefrontal EEG spectrogram
           'EEG', 'EEG2'         - raw hippocampal or prefrontal EEG signal
           'EMG', 'EMG2'         - raw EMG signals
           'EMG_AMP'             - amplitude of EMG signal
           'LFP'                 - filtered LFP signal
                                   * to plot P-wave detection threshold, add '_THRES'
                                   * to label detected P-waves, add '_ANNOT'
           'DFF'                 - DF/F signal
           'LSR'                 - laser stimulation pulse trains
           'LSR_DN'              - downsampled laser stimulation train
           'AUDIO'               - audio stimulation train
           'PULL'                - head pulls for REM sleep deprivation
	   
           e.g. PLOT = ['EEG', 'EEG2', 'LSR', 'LFP_THRES_ANNOT'] will plot both EEG 
           channels, the laser train, and the LFP signal with P-wave detection threshold 
           and labeled P-waves, in order from top to bottom.

    tstart, tend - time (s) into recording to start and stop plotting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_is - brain state for transition sleep
    eeg_nbin, emg_nbin, lfp_nbin, dff_nbin - factors by which to downsample raw EEG, EMG, LFP, and DF/F signals
    eeg_filt, emg_filt - 2-element list specifying [low, high] frequencies by which to filter EEG/EMG signal
                         * [f, -1] defines a high-pass filter above frequency f
                         * [-1, f] defines a low-pass filter below frequency f
    highres - if True, plot high-resolution spectrogram; if False, plot standard SP (2.5 s time resolution)
    recalc_highres - if True, recalculate high-res spectrogram from EEG using $nsr_seg and $perc_overlap params
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    pnorm - if > 0, normalize each spectrogram frequency by its mean power across the recording
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    fmax - maximum frequency in spectrogram
    vm - controls spectrogram saturation
    cmap - colormap to use for spectrogram plot
    ylims = optional list of y axis limits for each plot
    add_boxes - optional list of tuples specifying (start, end) time points (s) to highlight with red box
    @Returns
    if $return_data is True, returns $pplot dictionary of plotted items; otherwise, none
    """
    
    pplot = dict.fromkeys(PLOT, None)
    if len(ylims) != len(PLOT):
        ylims = ['']*len(PLOT)

    # load sampling rate
    sr = sleepy.get_snr(ppath, rec)
    
    # load EEG, get no. of Intan samples in recording
    EEG = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
    nsamples = len(EEG)
    
    # get start and end indices for Intan data (EEG/EMG/LFP/DFF/LSR/AUDIO)
    intan_start = int(round(tstart*sr))
    if tend == -1:
        intan_end = nsamples
        tend = int(round(nsamples/sr))
    else:
        intan_end = int(round(tend*sr))
    
    # adjust Intan idx to properly translate to Fourier idx (SP/EMG_AMP)
    f_adjust = np.linspace(-(nsr_seg*sr/2), (nsr_seg*sr/2), len(EEG))
    # set initial dt and Fourier time bins
    dt = 2.5
    fourier_start = int(round(intan_start/sr/dt))
    fourier_end = int(round(intan_end/sr/dt))
    
    if 'SP' in PLOT:
        # load hippocampal spectrogram
        if not highres:
            SPEEG = so.loadmat(os.path.join(ppath, rec, 'sp_%s.mat' % rec))
            SP = SPEEG['SP']
            freq = SPEEG['freq'][0]
            t = SPEEG['t'][0]
            sp_dt = SPEEG['dt'][0][0]
            sp_nbin = sp_dt*sr
        # load/calculate hippocampal high-res spectrogram
        else:
            SP, freq, t, sp_nbin, sp_dt = highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, 
                                                                    perc_overlap=perc_overlap, 
                                                                    recalc_highres=recalc_highres, mode='EEG')
        # normalize/smooth and collect spectrogram
        SP = adjust_spectrogram(SP, pnorm=pnorm, psmooth=psmooth, freq=freq, fmax=fmax)
        fourier_start = int(round((intan_start+f_adjust[intan_start])/sp_nbin))
        fourier_end = int(round((intan_end+f_adjust[intan_end])/sp_nbin))
        SP_cut = SP[:, fourier_start:fourier_end]
        pplot['SP'] = SP_cut
    
    if 'SP2' in PLOT:
        # load prefrontal spectrogram
        if not highres:
            SPEEG2 = so.loadmat(os.path.join(ppath, rec, 'sp2_%s.mat' % rec))
            SP2 = SPEEG2['SP2']
            freq = SPEEG2['freq'][0]
            t = SPEEG2['t'][0]
            sp_dt = SPEEG2['dt'][0][0]
            sp_nbin = sp_dt*sr
        # load/calculate prefrontal high-res spectrogram
        else:
            SP2, freq, t, sp_nbin, sp_dt = highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                                               recalc_highres=recalc_highres, mode='EEG', peeg2=True)
        # normalize/smooth and collect spectrogram
        SP2 = adjust_spectrogram(SP2, pnorm=pnorm, psmooth=psmooth, freq=freq, fmax=fmax)
        fourier_start = int(round((intan_start+f_adjust[intan_start])/sp_nbin))
        fourier_end = int(round((intan_end+f_adjust[intan_end])/sp_nbin))
        SP2_cut = SP2[:, fourier_start:fourier_end]
        pplot['SP2'] = SP2_cut
    
    
    if 'EMG_AMP' in PLOT:
        SPEMG = so.loadmat(os.path.join(ppath, rec, 'msp_%s.mat' % rec))
        mSP = SPEMG['mSP']
        mfreq = SPEMG['freq'][0]
        mt = SPEMG['t'][0]
        msp_dt = SPEMG['dt'][0][0]
        msp_nbin = msp_dt*sr
        # EMG amplitude = square root of (integral over each frequency)
        imfreq = np.where((mfreq >= 10) & (mfreq <= 100))[0]
        EMGAmpl = np.sqrt(mSP[imfreq,:].sum(axis=0) * (mfreq[1] - mfreq[0]))
        
        fourier_start = int(round((intan_start+f_adjust[intan_start])/msp_nbin))
        fourier_end = int(round((intan_end+f_adjust[intan_end])/msp_nbin))
        EMGAmpl_cut = EMGAmpl[fourier_start:fourier_end]
        pplot['EMG_AMP'] = EMGAmpl_cut
    
    
    if 'SP' not in PLOT and 'SP2' not in PLOT:
        sp_dt = 2.5 if not highres else 0.0
    
    if 'HYPNO' in PLOT:
        # get colormap for brain state annotation
        hypno_cmap, vmin, vmax = hypno_colormap()
        M_dt = sleepy.load_stateidx(ppath, rec)[0]
        M_dt_cut = M_dt[int(round(tstart/2.5)) : int(round(tend/2.5))]
        # adjust and collect brain state annotation
        M_dt_cut = adjust_brainstate(M_dt_cut, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is)
        pplot['HYPNO'] = M_dt_cut
            
    if 'EEG' in PLOT:
        # filter loaded EEG1 data
        if len(eeg_filt)==2:
            EEG = filter_signal(EEG, sr, f1=eeg_filt[0], f2=eeg_filt[1])
        # divide by 1000 to convert Intan data from uV to mV
        EEG_cut = EEG[intan_start:intan_end] / 1000
        EEG_cut_dn = downsample_vec(EEG_cut, eeg_nbin)
        pplot['EEG'] = EEG_cut_dn
    
    if 'EEG2' in PLOT:
        # load, collect, & filter EEG2 data
        EEG2 = so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'), squeeze_me=True)['EEG2']
        if len(eeg_filt)==2:
            EEG2 = filter_signal(EEG2, sr, f1=eeg_filt[0], f2=eeg_filt[1])
        EEG2_cut = EEG2[intan_start:intan_end] / 1000
        EEG2_cut_dn = downsample_vec(EEG2_cut, eeg_nbin)
        pplot['EEG'] = EEG_cut_dn
        
    if 'EMG' in PLOT:
        # load, collect, & filter EMG data
        EMG = so.loadmat(os.path.join(ppath, rec, 'EMG.mat'), squeeze_me=True)['EMG']
        if len(emg_filt)==2:
            EMG = filter_signal(EMG, sr, f1=emg_filt[0], f2=emg_filt[1])
        EMG_cut = EMG[intan_start:intan_end] / 1000
        EMG_cut_dn = downsample_vec(EMG_cut, emg_nbin)
        pplot['EMG'] = EMG_cut_dn
        
    if 'EMG2' in PLOT:
        # load, collect, & filter EMG data
        EMG2 = so.loadmat(os.path.join(ppath, rec, 'EMG2.mat'), squeeze_me=True)['EMG2']
        if len(emg_filt)==2:
            EMG2 = filter_signal(EMG2, sr, f1=emg_filt[0], f2=emg_filt[1])
        EMG2_cut = EMG2[intan_start:intan_end] / 1000
        EMG2_cut_dn = downsample_vec(EMG2_cut, emg_nbin)
        pplot['EMG2'] = EMG2_cut_dn
    
    lfps = [i for i in PLOT if 'LFP' in i]
    for l in lfps:
        # load & collect LFP data
        LFP = so.loadmat(os.path.join(ppath, rec, 'LFP_processed.mat'), squeeze_me=True)['LFP_processed']
        LFP_cut = LFP[intan_start:intan_end] / 1000
        LFP_cut_dn = downsample_vec(LFP_cut, lfp_nbin)
        # collect LFP signal, P-wave detection threshold, and P-wave indices
        ldata = [LFP_cut_dn, [], []]
        if l != 'LFP':
            pwave_info = so.loadmat(os.path.join(ppath, rec, 'p_idx.mat'), squeeze_me=True)
            # P-wave detection threshold
            if 'THRES' in l:
                pthres = np.empty((len(LFP,)))
                if 'thres' in pwave_info.keys():
                    pthres[:] = -pwave_info['thres']
                    ldata[1] = pthres[intan_start:intan_end] / 1000
            # P-wave indices
            if 'ANNOT' in l:
                pidx = np.zeros((len(LFP,)))
                pi = pwave_info['p_idx']
                for i in pi:
                    pidx[i] = LFP[i]
                ldata[2] = pidx[intan_start:intan_end] / 1000
        pplot[l] = ldata
        
    if 'DFF' in PLOT:
        #load & collect DF/F data
        DFF = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dff']*100
        DFF_cut = DFF[intan_start:intan_end]
        DFF_cut_dn = downsample_vec(DFF_cut, dff_nbin)
        pplot['DFF'] = DFF_cut_dn        
    
    if 'LSR' in PLOT:
        # load & collect laser stimulation vector
        LSR = sleepy.load_laser(ppath, rec)
        LSR_cut = LSR[intan_start:intan_end]
        pplot['LSR'] = LSR_cut
    
    if 'LSR_DN' in PLOT:
        # load, downsample, & collect laser stimulation vector
        LSR_DN = sleepy.load_laser(ppath, rec)
        (start_idx, end_idx) = sleepy.laser_start_end(LSR_DN, sr)
        for (i,j) in zip(start_idx, end_idx):
            LSR_DN[i:j+1] = 1
        LSR_DN_cut = LSR_DN[intan_start:intan_end]
        pplot['LSR_DN'] = LSR_DN_cut
    
    if 'AUDIO' in PLOT:
        # load & collect audio stimulation vector
        AUDIO = load_audio(ppath, rec)
        AUDIO_cut = AUDIO[intan_start:intan_end]
        pplot['AUDIO'] = AUDIO_cut
    
    if 'PULL' in PLOT:
        # load & collect head pull vector
        PULL = so.loadmat(os.path.join(ppath, rec, 'pull_'+rec+'.mat'), squeeze_me=True)['pull']
        PULL_cut = PULL[intan_start:intan_end]
        pplot['PULL'] = PULL_cut


    ###   GRAPHS   ###
    plt.ion()
    fig, axs = plt.subplots(nrows=len(PLOT), ncols=1, constrained_layout=True, sharex=True)
    if len(PLOT) == 1:
        axs = [axs]
    
    # create subplot for each item in $PLOT
    for i, data_type in enumerate(PLOT):
        data = pplot[data_type]
        ax = axs[i]
        y = ylims[i]
        # plot spectrogram
        if data_type == 'SP' or data_type == 'SP2':
            x = np.linspace(tstart, tend, data.shape[1])
            im = ax.pcolorfast(x, freq[np.where(freq <= fmax)[0]], data, cmap=cmap)
            if len(vm) == 2:
                im.set_clim(vm)
            ax.set_ylabel('Freq. (Hz)')
            plt.colorbar(im, ax=ax, pad=0.0)
        elif 'LFP' in data_type:
            # plot LFP signal
            ax.plot(np.linspace(tstart, tend, len(data[0])), data[0], color='black')
            # plot P-wave detection threshold
            if len(data[1]) > 0:
                ax.plot(np.linspace(tstart, tend, len(data[1])), data[1], color='green')
            # label detected P-waves
            if len(data[2]) > 0:
                x = np.linspace(tstart, tend, len(data[2]))
                ax.plot(x[np.where(data[2] != 0)[0]], data[2][np.where(data[2]!=0)[0]], 
                        color='red', marker='o', linewidth=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if len(y) == 2:
                ax.set_ylim(y)
            ax.set_ylabel('LFP (mV)')
        else:
            try:
                x = np.linspace(tstart, tend, len(data))
            except:
                pdb.set_trace()
            # plot brain state annotation
            if data_type == 'HYPNO':
                ax.pcolorfast(x, [0, 1], np.array([data]), vmin=vmin, vmax=vmax, cmap=hypno_cmap)
                ax.axes.get_yaxis().set_visible(False)
                
            # plot other data (EEG/EMG/EMG_AMP/DFF/LSR/AUDIO)
            else:
                ax.plot(x, data, color='black')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if len(y) == 2:
                    ax.set_ylim(y)
                if data_type == 'DFF':
                    ax.set_ylabel('DF/F (%)')
                elif data_type in ['LSR','LSR_DN']:
                    ax.set_ylabel('Laser')
                elif data_type == 'AUDIO':
                    ax.set_ylabel('Audio')
                elif data_type == 'PULL':
                    ax.set_ylabel('Head\npulls')
                else:
                    ax.set_ylabel('mV')
    # draw boxes
    if len(add_boxes) > 0:
        draw_boxes(axs, add_boxes)
    # set plot title
    axs[0].set_title(f'{rec}: {tstart}s - {tend}s')
    axs[-1].set_xlabel('Time (s)')
    
    plt.show()
    
    if return_data:
        return pplot


def draw_boxes(axs, coors):
    """
    Draw box(es) across vertically stacked subplots with shared x axes
    @Params
    axs - list of subplots
    coors - list of x coordinate pairs specifying (left edge, right edge) of each box
    @Returns
    None
    """
    line_kw = dict(color='red', linewidth=2, clip_on=False)
    box_t, box_b = axs[0].get_ylim()[1], axs[-1].get_ylim()[0]
    for coor in coors:
        # left/right lines
        box_l = coor[0]
        box_r = coor[1]
        # top/bottom lines
        axs[0].hlines(box_t, box_l, box_r, **line_kw)
        axs[-1].hlines(box_b, box_l, box_r, **line_kw)
        # connect lines
        line_l = matplotlib.patches.ConnectionPatch(xyA=[box_l,box_b], xyB=[box_l,box_t], 
                                                    coordsA='data', coordsB='data', 
                                                    axesA=axs[-1], axesB=axs[0], **line_kw)
        line_r = matplotlib.patches.ConnectionPatch(xyA=[box_r,box_b], xyB=[box_r,box_t], 
                                                    coordsA='data', coordsB='data', 
                                                    axesA=axs[-1], axesB=axs[0], **line_kw)
        axs[-1].add_artist(line_l)
        axs[-1].add_artist(line_r)


def get_unique_labels(ax):
    """
    Add legend of all uniquely labeled plot elements
    @Params
    ax - plot axis
    @Returns
    None
    """
    h,l = ax.get_legend_handles_labels()
    l_idx = list( dict.fromkeys([l.index(x) for x in l]) )
    legend = ax.legend(handles=[h[i] for i in l_idx], labels=[l[i] for i in l_idx], framealpha=0.3)
    ax.add_artist(legend)


def legend_mice(ax, mouse_names, symbol='', loc=''):
    """
    Add legend of all markers labeled with unique mouse names
    @Params
    ax - plot axis
    mouse_names - list of mouse names to include in legend
    symbol - if multiple marker types (e.g. '*' and 'o') are labeled with the same
             mouse name, the marker specified by $symbol is included in legend
    @Returns
    None
    """
    # find plot handles labeled by mouse names
    h,l = ax.get_legend_handles_labels()
    mouse_idx = [idx for idx, mouse in enumerate(l) if mouse in mouse_names]
    
    unique_mice = []
    # find preferred marker symbol in axes
    if symbol!='':
        symbol_idx = [idx for idx, handle in enumerate(h) if idx in mouse_idx and handle.get_marker() == symbol]
    else:
        symbol_idx = np.arange(0,len(h))
    for mname in list( dict.fromkeys(mouse_names) ):
        ms_symbol_idx = [si for si in symbol_idx if l[si] == mname]
        # use preferred symbol if it appears in the graph
        if len(ms_symbol_idx) > 0:
            unique_mice.append(ms_symbol_idx[0])
        else:
            unique_mice.append([mi for mi in mouse_idx if l[mi] == mname][0])
    # add legend of mouse names & markers 
    legend = ax.legend(handles=[h[i] for i in unique_mice], 
                       labels=[l[i] for i in unique_mice], framealpha=0.3)
    ax.add_artist(legend)


def legend_lines(ax, skip=[], loc=0):
    """
    Add legend of all uniquely labeled lines in plot
    @Params
    ax - plot axis
    skip - optional list of labels to exclude from legend
    loc - location of legend (0='best')
    @Returns
    None
    """
    # find handles & labels of lines in plot
    h,l = ax.get_legend_handles_labels()
    line_idx = [idx for idx,line in enumerate(h) if line in ax.lines]
    skip_idx = [idx for idx,lab in enumerate(l) if lab in skip]
    line_idx = [li for li in line_idx if li not in skip_idx]
    legend = ax.legend(handles=[h[i] for i in line_idx], labels=[l[i] for i in line_idx], framealpha=0.3, loc=loc)
    ax.add_artist(legend)
    

def legend_bars(ax, loc=0):
    """
    Add legend of all uniquely labeled bars in plot
    @Params
    ax - plot axis
    loc - location of legend (0='best')
    @Returns
    None
    """
    # find handles & labels of bars in plot
    h,l = ax.get_legend_handles_labels()
    bar_idx = [idx for idx,bar in enumerate(h) if bar in ax.containers]
    if len(bar_idx) > 0:
        legend = ax.legend(handles=[h[i] for i in bar_idx], labels=[l[i] for i in bar_idx], framealpha=0.3, loc=loc)
        ax.add_artist(legend)
    else:
        print('***No labeled bar containers found in these axes.')


def label_bars(ax, text=[], y_pos=[], above=0, dec=0, box=False):
    """
    Add text labels to bars in plot
    @Params
    ax - plot axis
    text - optional list of text labels for each bar
           * if empty list, label bars with y axis values
    y_pos - position of text on y axis
    above -  height above bars to place text, as fraction of bar height
    dec - no. of decimals in bar value labels
    box - if True, draw box around text labels
    @Returns
    None
    """
    for i, bar in enumerate(ax.patches):
        # label bar with y axis value
        height = bar.get_height()
        if len(text) == 0:
            if dec==0:
                txt = int(height)
            else:
                txt = round(height, dec)
        # label bar with input text
        else:
            txt = text[i]
        # set position of text label on y axis
        x = bar.get_x() + bar.get_width()/2
        if above > 0:
            y = bar.get_y() + height + above
        else:
            if len(y_pos)==0:
                y = bar.get_y() + height + (height/4)
            elif len(y_pos) == 1:
                y = bar.get_y() + y_pos[0]
            elif len(y_pos) == len(ax.patches):
                y = bar.get_y() + y_pos[i]
        # draw text
        if box:
            ax.text(x, y, txt, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
        else:
            ax.text(x, y, txt, ha='center', va='bottom')
    # pad ymax so text doesn't run off graph
    ypad = np.abs(ax.get_ylim()[0] - ax.get_ylim()[1])*0.2
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]+ypad])


def create_auto_title(brainstates=[], group_labels=[], add_lines=[]):
    """
    Generate descriptive plot title
    @Params
    brainstates - list of brain states in plot
    group_labels - list of groups in plot
    add_lines - list of extra text lines to add to title
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'Microarousals'}
    # clean data inputs
    if type(brainstates) != list:
        brainstates = [brainstates]
    if type(group_labels) != list:
        group_labels = [group_labels]
    if type(add_lines) != list:
        add_lines = [add_lines]
    brainstate_title = ''
    group_title = ''
    
    # create title for brainstate(s) in graph
    if len(brainstates) > 0:
        brainstate_names = []
        for b in brainstates:
            try:
                brainstate_names.append(states[int(b)])
            except:
                brainstate_names.append(b)
        # get rid of duplicate brainstate names and join into title
        brainstate_names = list(dict.fromkeys(brainstate_names).keys())
        brainstate_title = ' vs '.join(brainstate_names)
    # create title for group(s) in graph
    if len(group_labels) > 0:
        group_names = []
        for g in group_labels:
            try: group_names.append('group ' + str(int(g)))
            except: group_names.append(g)
        # get rid of duplicate group names
        group_names = list(dict.fromkeys(group_names).keys())
        group_title = ' vs '.join(group_names)
    
    # arrange titles in coherent order
    if 'vs' in brainstate_title:
        if group_title == ''        : title = brainstate_title
        else                        : title = brainstate_title + ' (' + group_title + ')'
    else:
        if brainstate_title == ''   : title = group_title
        elif group_title == ''      : title = brainstate_title
        elif 'vs' in group_title    : title = group_title + ' (' + brainstate_title + ')'
        else                        : title = brainstate_title + ' (' + group_title + ')'
    # add custom lines to title
    if len(add_lines) > 0:
        for line in add_lines:
            title = line + '\n' + title
        if title.split('\n')[-1] == '':
            title = title[0:-1]
    return title


def print_anova(res_anova, mc1, mc2=None, mc1_msg='', mc2_msg='', alpha=0.05, print_ns=False):
    """
    Print results of ANOVA and post-hoc tests
    """
    for pcol in ['p-unc', 'p-corr', 'p-GG-corr']:
        if pcol in res_anova.columns:
            res_anova[pcol] = pp(res_anova[pcol])
        if pcol in mc1.columns:
            mc1[pcol] = pp(mc1[pcol])
        if mc2 is not None and pcol in mc2.columns:
            mc2[pcol] = pp(mc2[pcol])
    txt1 = 'ANOVA SUMMARY'
    print('\n' + '='*len(txt1) + '\n' + txt1 + '\n' + '='*len(txt1) + '\n')
    print(res_anova.to_string())
    
    sig = any([float(x) < alpha for x in res_anova['p-unc']])
    if sig==True or print_ns==True:
        txt2 = 'POST-HOC TESTS'
        print('\n\n' + '='*len(txt2) + '\n' + txt2 + '\n' + '='*len(txt2) + '\n')
        if mc1_msg:
            print('###   ' + str(mc1_msg) + '   ###\n')
        print(mc1.to_string())
        if mc2 is not None:
            print('\n\n')
            if mc2_msg:
                print('###   ' + str(mc2_msg) + '   ###\n')
            print(mc2.to_string())