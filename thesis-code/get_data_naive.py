from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, find_events
import shutil

import scipy.io
import numpy as np


def move_img(imgNum, label):

    if label:
        try:
            shutil.copy('/home/harry/eeg/eegmmidb/img/S019/R04/S019R04_{0:05d}.png'.format(imgNum),
                        '/home/harry/eeg/data/naive/class{0:d}/S019R04_{1:05d}_{2:0}.png'.format(label, imgNum, label))
        except:
            print("broke on ", imgNum, label)
    else:
        print("E Didn't get label")



tmin, tmax = -1., 4.
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

# Array to store all trials for image generation
all_trials = np.empty([109 * 45, 64 * 3 * 7 + 1], dtype=float)
subj_nums = np.empty([109 * 45, 1], dtype=int)
subject_point = 0

for subject in range(20, 21):

    raw_fnames = eegbci.load_data(subject, [4], path='/home/harry/eeg')

    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                 raw_fnames]

    # print(raw_fnames.shape, raw_files.shape)

    """Note: concatenate_raws

    raws[0] is modified in-place to achieve the concatenation. Boundaries of the raw 
    files are annotated bad. If you wish to use the data as continuous recording, 
    you can remove the boundary annotations after concatenation"""
    raw = concatenate_raws(raw_files)
    # events = mne.io.find_edf_events(raw)

    # print("GOT RAWS")
    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))
    # raw_input()

    # Apply band-pass filter
    # raw.filter(1., 45., fir_design='firwin', skip_by_annotation='edge')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    label = None

    for img in range(656, raw.n_times):
        for row in range(0, events.shape[0] - 1):

            if img >= events[row, 0] and img < events[row + 1, 0]:    # Label in this row
                label = events[row, 2]

        # Handle final row
        if img >= events[-1, 0]:
            label = events[-1, 2]

        move_img(img, label)

