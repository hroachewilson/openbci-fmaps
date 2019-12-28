from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, find_events
from mne.time_frequency import psd_welch, psd_multitaper
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from utils import augment_EEG, cart2sph, pol2cart
import cv2
import sys
import mne
import scipy.io
import numpy as np
import math as m


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(int(n_colors)):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])
    if augment:
        if pca:
            for c in range(int(n_colors)):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(int(n_colors)):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
                     ]
    temp_interp = []
    for c in range(int(n_colors)):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(int(n_colors)):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating

    # POTENTIALLY DANGEROUS CHANGE FROM XRANGE
    for i in range(nSamples):
        for c in range(int(n_colors)):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, nSamples), end='\r')
    # Normalizing
    for c in range(int(n_colors)):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def mean_zero(matrix):
    normed = matrix.copy()
    normed = (normed - normed.mean(axis=0)) / normed.std(axis=0)
    return normed


############################################################

tmin, tmax = 0., 30.
event_id = dict(left=2, right=3)

runs_hf = [6, 10, 14]  # motor imagery: hands vs feet
runs_lr = [4, 8, 12]  # motor imagery: left vs right

# let's explore some frequency bands
iter_freqs = [
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25)
    # ('Gamma', 30, 45)
]

# Load electrode locations
print('Loading data...')
locs = scipy.io.loadmat('Sample data/Neuroscan_locs_orig.mat')
locs_3d = locs['A']
locs_2d = []

# Convert to 2D
for e in locs_3d:
    locs_2d.append(azim_proj(e))

# Array to store all trials for image generation
all_trials = np.empty([109 * 45, 64 * 3 * 30 + 1], dtype=float)
subj_nums = np.empty([109 * 45, 1], dtype=int)
subject_point = 0

mne.set_log_level(verbose=False, return_old_level=False)
for subject in range(1, 109):

    raw_fnames = eegbci.load_data(subject, runs_hf, path='/home/harry/eeg')

    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto', verbose=False) for f in
                 raw_fnames]

    """Note: concatenate_raws

    raws[0] is modified in-place to achieve the concatenation. Boundaries of the raw 
    files are annotated bad. If you wish to use the data as continuous recording, 
    you can remove the boundary annotations after concatenation"""
    raw = concatenate_raws(raw_files)
    # events = mne.io.find_edf_events(raw)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(1., 45., fir_design='firwin', skip_by_annotation='edge')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014', verbose=False)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Check if subject has adequate number of samples per trial
    checkShape = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True, verbose=False).get_data().shape

    if checkShape[-1] >= 4800:

        subj_trials = np.empty([checkShape[0], 64 * 3 * 30 + 1], dtype=float)
        point = 0
        # Get 7 time windows over 801 samples, min width of 255 for fft
        for win in range(0, 30):
            start = win * 142
            end = win * 142 + 512
            # if end == 801:
            #     start -= 1
            #     end -= 1

            # Get 3 bands (theta, alpha, beta)
            for band, fmin, fmax in iter_freqs:

                # Get 64 channels, individually
                for chan in range(0, 64):
                    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=[chan],
                                    baseline=None, preload=True, verbose=False)

                   # DOC: https://martinos.org/mne/dev/python_reference.html?highlight=time_frequency#module-mne.time_frequency
                    psds, freqs = psd_welch(epochs, picks=[0], fmin=fmin, fmax=fmax, n_fft=512,
                                            tmin=epochs._raw_times[start], tmax=epochs._raw_times[end])
                    norms = np.true_divide(psds[:, 0, :], freqs)
                    subj_trials[:, (64 * point) + chan] = np.mean(norms, axis=1)
                    # means = psds.mean(2)[:, 0] * ((fmax + fmin) / 2)
                    # subj_trials[:, (64 * point) + chan] = 10 + np.log10(means)

                point += 1
            images = gen_images(np.array(locs_2d),
                                mean_zero(subj_trials[:, (64 * (point - 3)):(64 * point)]),
                                80,
                                # augment=True,
                                pca=True,
                                edgeless=True,
                                normalize=False)

            for frame in range(0, checkShape[0]):
                img = cv2.merge([images[frame][0], images[frame][1], images[frame][2]])
                norm_img = img + img.min()
                norm_img = img * 255.0 / min(img.max(), 1.0)
                cv2.imwrite('/home/harry/eeg/data/trial_hf/{0:04d}_{1:02d}_lab{2:01d}_80.jpg'.format(
                    frame + subject_point, win, epochs.events[frame, - 1] + 1), norm_img)
            # img = cv2.merge([images[0][0], images[0][1], images[0][2]])
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

        subj_trials[:, :] = mean_zero(subj_trials[:, :])
        subj_trials[:, (64 * point)] = epochs.events[:, - 1] + 1
        all_trials[subject_point:subject_point + checkShape[0], :] = subj_trials[:, :]
        subj_nums[subject_point:subject_point + checkShape[0]] = subject
        subject_point += checkShape[0]
        print("got {0:04d} sequences for {1:03d} subjects".format(subject_point, subject), file=sys.stdout)

scipy.io.savemat('/home/harry/eeg/data/timeseries/HF_trial_FeatureMat_timeWin.mat',
                 mdict={'features1': all_trials[0:subject_point, :]})
scipy.io.savemat('/home/harry/eeg/data/timeseries/HF_trial_trials_subNums.mat',
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
