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
    sample, timestamp = inlet.pull_sample()
    sample = np.array(sample[:high_band])
    print(sample)
    sample[:] = 255.0 * (sample[:] / max(sample.max() - sample.min(), 0.001))
    channel_data = []
    image = np.zeros((4,4,3), np.uint8)
    print(sample)

    for i in range(4):
        for j in range(4):
            #image[i, j, 0] = 0
            #print(i,j)
            #image[i, j, 0] = 255
            #image[i, j, 1] = 0
            #image[i, j, 2] = 0
            image[i, j, 0] = int(sum(sample[0:low_band]) / band_width)
            image[i, j, 1] = int(sum(sample[low_band:mid_band]) /  band_width)
            image[i, j, 2] = int(sum(sample[mid_band:high_band]) / band_width)
            print(sample[0:low_band])
            print(sample[low_band:mid_band])
            print(sample[mid_band:high_band])
            print(i,j, " has ", int(sum(sample[0:low_band]) / band_width),
                                int(sum(sample[low_band:mid_band]) /  band_width),
                                int(sum(sample[mid_band:high_band]) / band_width))
            #print(i, j)

    cv2.imshow('', cv2.resize(image, (800,600), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
