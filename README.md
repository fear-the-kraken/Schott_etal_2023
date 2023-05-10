## Schott et al., 2022
Code to generate analyses and figures in *Schott et al., 2022*

***
### Overview

* System Requirements and Installation Information


* Data Organization
   * ```data```


* Data Analysis Modules
   * ```pwaves.py```, ```AS.py```, ```sleepy.py```, and ```pyphi.py```


* Live Data Annotation
   * ```sleep_annotation_qt.py```


* Figure Generating Scripts
   * ```MAIN_FIGURES.py``` and ```SUPPLEMENTAL_FIGURES.py```


* Example Analyses
   * ```demo_analysis.py```

***

### System Requirements and Installation Information
* All code is written in Python 3, and has been tested on Python versions 3.5, 3.7, and 3.9.


* The required packages for Python scripts/modules are listed at the beginning of each file. Packages can be installed using the ```conda``` package manager in the Anaconda distribution of Python (version 4.10.1). Go to <https://www.anaconda.com/> to install Anaconda on Windows, MacOS, or Linux. Installation will take approximately 15 minutes.

* To run our code, we recommend using the Spyder IDE (version 5.1.5), available through the Anaconda distribution. To import our user-defined modules, the working directory must be set to the folder containing those modules. This can be done using the following commands:
```
import os
os.chdir([path to module folder])
```

Alternatively, the module location(s) may be manually added to the file path.
```
import sys
sys.path.insert(0, [path to module folder])
```

* For functions that load data from recording folders (see **Data Organization**) the argument for ```ppath``` must be the path to the location of those recording folders; each user must set this for his/her system.

<br />

### Data Organization

#### The recording folder
For each experimental recording, all raw and processed data is stored in a unique folder, named in the following format: ```[mouse name]_[recording date]_n[recording order]```. For example, two recordings from mouse "Alice" on March 17th, 2022 would be named **Alice_031722_n1** and **Alice_031722_n2**. Raw data is collected using a TDT or Intan amplifier, saved as *.mat* files in the recording folder, and processed during post-hoc analysis.

* **Essential Files**
   * ```EEG.mat``` and ```EEG2.mat``` - parietal and frontal raw EEG signals
   * ```sp_[recording name].mat``` - EEG spectrogram with 2.5 s time resolution
   * ```remidx_[recording name].txt``` - brain state classification for each 2.5 s time bin
   * ```info.txt``` - general recording information (e.g. time of day, amplifier sampling rate, etc)


* **Experiment-Specific Files**
   * ```EMG.mat``` and ```msp_[recording name].mat``` - raw EMG signal and EMG spectrogram
   * ```LFP_raw.mat``` and ```LFP_raw2.mat``` - raw LFP signals
   * ```LFP_processed.mat``` - filtered and subtracted LFP signal
   * ```p_idx.mat``` - indices of detected P-waves
   * ```sp_highres_[recording name].mat``` - EEG spectrogram with higher time resolution than 2.5 s
   * ```laser_[recording name].mat``` - laser stimulation vector for optogenetic recordings
   * ```rem_trig_[recording name].mat``` - automatic REM detection for closed-loop optogenetic recordings
   * ```DFF.mat``` - $\Delta$F/F calcium signal for fiber photometry recordings

#### ```data```
The ```data``` folder can be downloaded [here](https://upenn.box.com/v/schott-etal-data), and contains three recordings used to demonstrate the expected output of the code in ```demo_analysis.py``` (see **Example Analyses** for more information). Mouse 1 was used to record P-waves, Mouse 2 is from a fiber photometry experiment, and Mouse 3 was used to analyze laser-triggered P-waves.
```
data
├── mouse1_072420n1
│   ├── EEG.mat
│   ├── EEG2.mat
│   ├── EMG.mat
│   ├── LFP_raw.mat
│   ├── LFP_raw2.mat
│   ├── LFP_processed.mat
│   ├── p_idx.mat
│   ├── sp_mouse1_072420n1.mat
│   ├── remidx_mouse1_072420n1.txt
│   ├── info.txt
├── mouse2_032819n1
│   ├── DFF.mat
│   ├── EEG.mat
│   ├── EEG2.mat
│   ├── LFP_raw.mat
│   ├── LFP_raw2.mat
│   ├── LFP_processed.mat
│   ├── p_idx.mat
│   ├── sp_mouse2_032819n1.mat
│   ├── sp_highres_mouse2_032819n1.mat
│   ├── remidx_mouse2_032819n1.txt
│   ├── info.txt
├── mouse3_110619n1
│   ├── laser_mouse3_110619n1.mat
│   ├── EEG.mat
│   ├── EEG2.mat
│   ├── LFP_raw.mat
│   ├── LFP_raw2.mat
│   ├── LFP_processed.mat
│   ├── p_idx.mat
│   ├── sp_mouse3_110619n1.mat
│   ├── sp_highres_mouse3_110619n1.mat
│   ├── remidx_mouse3_110619n1.txt
│   ├── rem_trig_mouse3_110619n1.txt
│   ├── info.txt
```
<br />

### Data Analysis Modules
Custom-written Python code for loading, processing, analyzing, and visualizing data. The modules ```pwaves.py``` and ```AS.py``` contain all functions used to generate graphs and statistics in *Schott et al., 2022*. ```sleepy.py``` and ```pyphi.py``` are general lab modules for data analysis; support functions in these files are used for basic loading and processing of data signals. Details about a given function can be found in its documentation.

#### ```pwaves.py```
Functions for analyzing and visualizing P-wave data
   * *Supporting Functions*: organize and transform raw data into structures used by other Python functions
   
      * e.g. ```mx2d(rec_dict, mouse_avg)``` accepts the dictionary ```rec_dict```containing a list of 1D data vectors for each recording, and returns a 2D data matrix with rows as single trials or as averages within recordings or mice (```mouse_avg='trial'```, ```'rec'```, or ```'mouse'```, respectively).



   * *Data Collection Functions*: isolate and collect particular data from raw recording files
   
      * e.g. ```get_surround(ppath, recordings, istate, win, signal_type)``` loads a series of experimental ```recordings``` and collects raw data in the time window ```win``` relative to P-waves occurring in each brain state in ```istate```. The parameter ```signal_type``` specifies the type of data to load (e.g. ```'LFP'```, ```'EEG'```, or ```'SP'``` to collect subtracted LFP signals, raw hippocampal EEG, or EEG spectrogram, respectively).


   * *P-wave Detection Functions*: detect P-waves in pontine LFP signal, classify into sub-categories, get basic waveform features 
   
      * e.g. ```detect_pwaves(ppath, recordings, channel, thres)``` loads raw LFP signals from experimental `recordings`, subtracts/processes signal according to ```channel``` parameter, and detects waveforms crossing a threshold of (signal mean - ```thres``` \* signal std). Additional parameters specify threshold values for detecting motor artifacts.


   * *Analysis and Plotting Functions*: use functions for data organization/collection to generate P-wave plots and statistics in manuscript figures
   
      * e.g. ```stateseq(ppath, recordings, sequence, nstates, statethres, mode='pwaves')``` plots the mean time-normalized EEG spectrogram and P-wave frequency (```mode='pwaves'```) or calcium activity (```mode='dff'```) during transitions across the consecutive brain states in ```sequence```. Parameters ```nstates``` and ```statethres``` specify the number of downsampled time bins and the minimum episode duration for each brain state, and additional arguments control plotting features and timecourse statistics.


   * *Statistical Functions*: organize input data into format accepted by Python modules ```scipy.stats``` and ```statsmodels.stats```, and perform statistical comparisons
   
      * e.g. ```stats_timecourse(mx, pre, post, sr, base_int``` accepts the data matrix ```mx``` (subject x time bins), and performs repeated t-tests with Bonferroni correction, comparing baseline time bin to each subsequent bin. ```pre``` and ```post``` define the time range of the data matrix, either relative to an event (e.g. -5 and +5 s) or in absolute time (e.g. 100 and 200 s). Each input matrix column spans ```sr``` s, and each statistical comparison bin averages ```base_int``` s of data.
      
#### ```AS.py```
General-purpose functions for loading/handling raw data and plotting sleep behavior

   * *Supporting Functions*: organize and transform raw data into structures used by other Python functions
   
      * e.g. ```highres_spectrogram(ppath, rec, nsr_seg=2, perc_overlap=0.95, recalc_highres=False)``` loads or calculates (```recalc_highres=False``` or ```True```, respectively) the EEG spectrogram for experimental recording ```rec```. Spectrogram is calculated using ```scipy.signal.spectrogram``` with sliding Hanning windows of ```nsr_seg``` s overlapping by ```perc_overlap```.


   * *Data Analysis Functions*: use functions for data organization/collection to generate sleep plots and statistics in manuscript figures
   
      * e.g. ```laser_brainstate(ppath, recordings, pre, post)``` loads list of ```recordings``` from open-loop optogenetic experiments, and calculates the percent time spent in each brain state (REM, NREM, wake, IS) surrounding the laser. Plots show a time window spanning ```pre``` s before and ```post``` s following laser onset, and statistical comparison is performed using paired t-test of mean brain state percentage during 120 s laser interval vs 120 s pre-laser interval.


   * *Plotting Functions*: format parameters for matplotlib graphs and plot example data from single recordings
   
      * e.g. ```plot_example(ppath, recordings, PLOT, tstart, tend)``` plots data from ```tstart``` to ```tend``` s into experimental recording ```rec```. Parameter ```PLOT``` specifies the type of data signals to include in graph (e.g. ```['LSR','EEG','SP','LFP']``` would plot the laser train, raw EEG trace, EEG spectrogram, and subtracted LFP signal on the same time scale).


#### Other
Support functions for loading/processing data signals (e.g. ```get_snr``` to load recording sampling rate from a .txt file)
   * ```sleepy.py``` - common lab module for analyzing and plotting sleep behavior data
      * Functions that exist in both ```AS.py``` and ```sleepy.py``` (e.g. ```laser_brainstate```) were adapted from original code in ```sleepy.py```. Versions in ```AS.py``` have been modified to produce the analyses in *Schott et al., 2022*.<br />


   * ```pyphi.py``` - common lab module for analyzing and plotting fiber photometry data

<br />

### Live Data Annotation
Custom user interface for manually scoring behavioral states.

#### ```sleep_annotation_qt.py```
* Creates interactive window displaying data from one recording. Available signals to plot:
   * Raw parietal and frontal EEG traces
   * Spectrogram calculated from parietal EEG, with time resolution of 2.5 s
   * Raw EMG traces and calculated EMG amplitude
   * Raw LFP traces and subtracted LFP signal
   * Markers identifying P-waves and P-wave detection threshold
   * P-wave frequency in each 2.5 s bin
   * Raw DF/F calcium signal and downsampled DF/F in 2.5 s bins
   * Laser stimulation vector
   * Automatic REM detection from closed-loop optogenetic recordings
   * Color-coded hypnogram of assigned brain state for each 2.5 s bin (editable by user)


* Brain states are initially classified by an automatic algorithm using raw EEG and EMG data. Experimenters manually annotate the recording by visually inspecting these signals, along with the EEG spectrogram and EMG amplitude.

<br />

### Figure Generation Scripts
Code for generating graphs and statistics for each panel in the main and supplemental figures of *Schott et al., 2022*. Files are organized in cells, with one cell per figure panel and all necessary Python packages imported in the first cell. Figure components that were not generated by Python (e.g. schematics and histology images) are not included.

#### ```MAIN_FIGURES.py```
Generates plots and statistics for Figures 1-5.

#### ```SUPPLEMENTAL_FIGURES.py```
Generates plots and statistics for Supplementary Figures 1-6.

<br />

### Example Analyses
Code to run three example analyses and demonstrate expected output.

#### ```demo_analysis.py```
Python script for generating example plots, organized into cells which contain the code for one analysis each.
* **Example 1: P-wave detection**
   * Filter and subtract raw pontine LFPs from one recording, and detect P-waves in processed LFP signal
   * Output: updated ```LFP_processed.mat``` and ```p_idx.mat``` files in the recording folder, and an example plot showing detected P-waves during a REM sleep episode
   * Data source: */data/mouse1_072420n1*


* **Example 2: Spectral field estimation**
   * Estimate the "spectral field" optimally mapping the EEG spectrogram onto dmM CRH neuron calcium activity for one recording
   * Output - a plot showing the estimated spectral field, and R2 values for model performance with each candidate value of the regularization term $\lambda$
   * Data source: */data/mouse2_032819n1*


* **Example 3: Spectral profile analysis**
   * Calculate average EEG spectrograms surrounding spontaneous P-waves, laser-triggered P-waves, failed laser pulses, and random control points in one recording, and plot the normalized power spectral density of each event
   * Output - a plot showing the average spectrograms for spontaneous and laser-triggered P-waves, and a plot comparing their spectral profiles to failed laser pulses and random control points
   * Data source: */data/mouse3_110619n1*

#### How to Run

* Step 0: Install the Anaconda distribution of Python (version 3.5 or higher) on your system
* Step 1: Click [here](https://upenn.box.com/v/schott-etal-data) to download the raw data for the example code. Download the ```data``` folder, and make sure to save it inside **this folder**, to ensure that all scripts and modules are in the same location.
* Step 2: Launch the Spyder IDE, and open ```demo_analysis.py```
* Step 3: In the IPython console, use the following commands to set the working directory to **this folder**. For example:

```
    import os
    os.chdir('/home/fearthekraken/Downloads/schott_etal_code')
```

* Step 4: Click on the first code cell in the script, and press "Run current cell" to import all required Python packages 
   * For any ```ModuleNotFoundError```, use ```conda``` to install the missing packages

```
        import h5py
        ModuleNotFoundError: No module named 'h5py'
        conda install h5py
```

* Step 5: Run the code cell for each example analysis, and view the output(s)
   * Expected run time for all 3 analyses is approximately 30-60 seconds.

**NOTE**: To ensure that the demo code works property, the following items must be stored in the same directory: ```pwaves.py```, ```AS.py```, ```sleepy.py```, ```pyphi.py```, ```demo_analysis.py```, and the ```data``` folder. This directory must be set as the current working directory for the Python interpreter.
