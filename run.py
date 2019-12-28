# general imports
import numpy as np
from collections import deque
import time

# openbci imports
from pyOpenBCI import OpenBCICyton

# scikit-cuda imports
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft

sequence = np.zeros((30000, 16))
fps_counter = deque(maxlen=50)
previous_time = time.time()

def print_raw(sample):
    #print(sample.channels_data)
    global sequence
    global previous_time

    sequence = np.roll(sequence, 1, 0)
    sequence[0, ...] = sample.channels_data

    fps_counter.append(time.time() - previous_time)
    previous_time = time.time()
    print(  f'FPS: {1/(sum(fps_counter)/len(fps_counter)):.2f}')

board = OpenBCICyton(daisy=True)

board.start_stream(print_raw)
