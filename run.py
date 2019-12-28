from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import cv2

last_print = time.time()
fps_counter = deque(maxlen=150)

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

inlet = StreamInlet(streams[0])

channel_data = {}

low_band = 15
mid_band = 30
high_band = 45
band_width = 15

FFT_MAX_HZ = 60

HM_SECONDS = 10  # this is approximate. Not 100%. do not depend on this.
TOTAL_ITERS = HM_SECONDS*25  # ~25 iters/sec

while True:

    image = np.zeros((4,4,3), np.uint8)
    for i in range(16):
        sample, timestamp = inlet.pull_sample()
        sample = np.array(sample[:high_band])
        sample_range = sample.max() - sample.min()
        sample[:] = sample[:] / sample_range

        image[int(i / 4), i % 4, 0] = int(255.0 * sum(sample[0:low_band]) / band_width)
        image[int(i / 4), i % 4, 1] = int(255.0 * sum(sample[low_band:mid_band]) /  band_width)
        image[int(i / 4), i % 4, 2] = int(255.0 * sum(sample[mid_band:high_band]) / band_width)

    cv2.imshow('', cv2.resize(image, (800,600), interpolation=cv2.INTER_CUBIC))
    #cv2.imshow('', image)
    cv2.waitKey(1)
