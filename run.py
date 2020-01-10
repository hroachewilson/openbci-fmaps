import sys
import lib.transforms as transforms
from lib.utils import azim_proj

from lib.BCI import BCI

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import cv2

if __name__ == '__main__':

    band_freqs = {  'Delta' : (0, 4),       # index:    0
                    'Theta' : (4, 7),       # index:    1
                    'Alpha' : (7, 15),      # index:    2
                    'Beta'  : (15, 31),     # index:    3
                    'Gamma' : (31, 45)  }   # index:    4

    bci = BCI(band_freqs)
    while True:

        # Run LSL in Band Power Mode, 16 channels
        bci.get_band_power(2, 1, 4)
        cv2.imshow('', cv2.resize(bci.get_img(50), (800,800), interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(1)

        # Run LSL in FFT Mode, 125 channels
        #bci.plot_lsl_fft()
        
    print('done')
    locs_3d = np.array(transforms.neuroscan_positions_64)
    locs_2d = []
    print(locs_3d)

    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    breaknum = 0
    #for line in locs_3d:
    #    ax.scatter(line[0], line[1], line[2], cmap='Greens')
    for line in transforms.openbci_positions_16:
        ax.scatter(line[0], line[1], 0, cmap='Greens')
    plt.show()
    exit(0)
