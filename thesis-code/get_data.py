
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, find_events
from mne.time_frequency import psd_welch, psd_multitaper

import scipy.io
import numpy as np

tmin, tmax = -1., 4.
event_id = dict(left=2, right=3)


runs_hf = [6, 10, 14]  # motor imagery: hands vs feet
runs_lr = [4, 8, 12] # motor imagery: left vs right

# let's explore some frequency bands
iter_freqs = [
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25)
    #('Gamma', 30, 45)
]

# Array to store all trials for image generation
all_trials = np.empty([109*45, 64*3*7+1], dtype=float)
subj_nums = np.empty([109*45, 1], dtype=int)
subject_point = 0

for subject in range(1, 109):

    raw_fnames = eegbci.load_data(subject, runs_lr, path='/home/harry/eeg')

    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                 raw_fnames]

    #print(raw_fnames.shape, raw_files.shape)

    """Note: concatenate_raws
    
    raws[0] is modified in-place to achieve the concatenation. Boundaries of the raw 
    files are annotated bad. If you wish to use the data as continuous recording, 
    you can remove the boundary annotations after concatenation"""
    raw = concatenate_raws(raw_files)
    #events = mne.io.find_edf_events(raw)

    #print("GOT RAWS")
    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))
    #raw_input()

    # Apply band-pass filter
    #raw.filter(1., 45., fir_design='firwin', skip_by_annotation='edge')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Check if subject has adequate number of samples per trial
    checkShape = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True).get_data().shape


    print('check')

    #if epochs.get_data()
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier

    if 1 and (checkShape[2] == 801):
        subj_trials = np.empty([checkShape[0], 64*3*7+1], dtype=float)
        point = 0
        # Get 7 time windows over 801 samples, min width of 255 for fft
        for win in range(0, 7):
            start = win * 91
            end = win * 91 + 255
            if end == 801:
                start -= 1
                end -= 1

            # Get 3 bands (theta, alpha, beta)
            for band, fmin, fmax in iter_freqs:

                # Get 64 channels, individually
                for chan in range(0, 64):
                    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=[chan],
                                    baseline=None, preload=True)

                    # DOC: https://martinos.org/mne/dev/python_reference.html?highlight=time_frequency#module-mne.time_frequency
                    psds, freqs = psd_welch(epochs, picks=[0], fmin=fmin, fmax=fmax,
                                            tmin=epochs._raw_times[start], tmax=epochs._raw_times[end])
                    means = psds.mean(2)[:, 0] * ((fmax+fmin)/2)
                    #subj_trials[:, (64*point) + chan] = np.sqrt((psds.sum(2)[:, 0])*160)
                    subj_trials[:, (64 * point) + chan] = 10 + np.log10(means)
                    print("OK")

                point += 1
        subj_trials[:, (64 * point)] = epochs.events[:, -1] - 1
        all_trials[subject_point:subject_point + checkShape[0], :] = subj_trials[:, :]
        subj_nums[subject_point:subject_point + checkShape[0]] = subject
        subject_point += checkShape[0]
        print("OK")

scipy.io.savemat('/home/harry/eeg/data/timeseries/LR_FeatureMat_timeWin.mat',
                 mdict={'features1': all_trials[0:subject_point, :]})
scipy.io.savemat('/home/harry/eeg/data/timeseries/LR_trials_subNums.mat',
                 mdict={'subjectNum1': subj_nums[0:subject_point]})

    # epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
    #                 baseline=None, preload=True)
    # labels = epochs.events[:, -1] - 2
    # ###############################################################################
    # # Classification with linear discrimant analysis
    #
    # # Define a monte-carlo cross-validation generator (reduce variance):
    # epochs_data = epochs.get_data()
    # #epochs_data_train = epochs_train.get_data()
    # #cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # #cv_split = cv.split(epochs_data_train)
    #
    # scipy.io.savemat('/home/harry/eeg/data/timeseries/'+format(subject, '03d')+'_lr_data.mat', mdict={'arr': epochs_data})
    # scipy.io.savemat('/home/harry/eeg/data/timeseries/'+format(subject, '03d')+'_lr_label.mat', mdict={'label': labels})
    # #print("OK")
    # print('subject: ', subject)
