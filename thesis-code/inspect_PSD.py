import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, EpochsArray, pick_types, find_events

import mne
from mne.baseline import rescale
from mne.stats import _bootstrap_ci

import scipy.io

# set epoching parameters
tmin, tmax = -1., 4.
event_id = dict(left=2, right=3)
baseline = None

# define runs
runs_hf = [6, 10, 14]  # motor imagery: hands vs feet
runs_lr = [4, 8, 12]  # motor imagery: left vs right

# let's explore some frequency bands
iter_freqs = [
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25),
    ('Gamma', 30, 45)
]

for subject in range(1, 109):

    raw_fnames = eegbci.load_data(subject, runs_lr, path='/home/harry/eeg')
    # Let's get power spectral density for all trials
    for raw_fname in raw_fnames:

        # get the header to extract events
        raw = read_raw_edf(raw_fname, preload=True)
        events = find_events(raw, shortest_event=0, stim_channel='STI 014')

        frequency_map = list()

        for band, fmin, fmax in iter_freqs:

            # (re)load the data to save memory
            raw = read_raw_edf(raw_fname, preload=True)

            # Pick channels
            raw.pick_types(raw.info, eeg=True, stim=False, eog=False,
                           exclude='bads')

            # bandpass filter and compute Hilbert
            raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                       l_trans_bandwidth=1,  # make sure filter params are the same
                       h_trans_bandwidth=1,  # in each band and skip "auto" option.
                       fir_design='firwin')
            raw.apply_hilbert(n_jobs=1, envelope=False)

            epochs = Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                                preload=True)

            # remove evoked response and get analytic signal (envelope)
            epochs.subtract_evoked()  # for this we need to construct new epochs.
            epochs = EpochsArray(
                data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin)
            # now average and move on
            frequency_map.append(((band, fmin, fmax), epochs.average()))
            print('OK')

        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        for ((freq_name, fmin, fmax), average), color, ax in zip(
                frequency_map, colors, axes.ravel()[::-1]):
            times = average.times * 1e3
            gfp = np.sum(average.data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
            ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
            ax.axhline(0, linestyle='--', color='grey', linewidth=2)
            ci_low, ci_up = _bootstrap_ci(average.data, random_state=0,
                                          stat_fun=lambda x: np.sum(x ** 2, axis=0))
            ci_low = rescale(ci_low, average.times, baseline=(None, 0))
            ci_up = rescale(ci_up, average.times, baseline=(None, 0))
            ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
            ax.grid(True)
            ax.set_ylabel('GFP')
            ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                        xy=(0.95, 0.8),
                        horizontalalignment='right',
                        xycoords='axes fraction')
            ax.set_xlim(-1000, 3000)
            print(freq_name)
        axes.ravel()[-1].set_xlabel('Time [ms]')
        plt.show()



    print('OK')
    exit(0)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    ###############################################################################
    # Classification with linear discrimant analysis

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    # epochs_data_train = epochs_train.get_data()
    # cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # cv_split = cv.split(epochs_data_train)

    # scipy.io.savemat('/home/harry/eeg/data/timeseries/'+format(subject, '03d')+'_lr_data.mat', mdict={'arr': epochs_data})
    # scipy.io.savemat('/home/harry/eeg/data/timeseries/'+format(subject, '03d')+'_lr_label.mat', mdict={'label': labels})
    # print("OK")
    print('subject: ', subject)