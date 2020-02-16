# general imports
import numpy as np
import time, sys
from threading import Thread
from pynput.keyboard import Key, Listener
import cv2

# helper imports
import lib.transforms as transforms
from lib.utils import gen_images

# openbci imports
from pyOpenBCI import OpenBCICyton
from pylsl import StreamInlet, resolve_stream


class BCI:
    def __init__(self, bands, blue=2, green=1, red=4):
        self.band_freqs = bands
        self.streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(self.streams[0])
        self.feats = []
        self.blue_chan = blue
        self.green_chan = green
        self.red_chan = red

        # Get electrode locations
        self.locs = np.array(transforms.openbci_positions_16)

        # Thread objects
        self.keyboard = Listener(on_press=self._process_key)
        #self.bci_stream = Thread(target=self._process_stream)

        # Generate band data
        self.theta_low = bands['Theta'][0]
        self.theta_high = bands['Theta'][1]
        self.theta_width = self.theta_high - self.theta_low
        self.alpha_low = bands['Alpha'][0]
        self.alpha_high = bands['Alpha'][1]
        self.alpha_width = self.alpha_high - self.alpha_low
        self.beta_low = bands['Beta'][0]
        self.beta_high = bands['Beta'][1]
        self.beta_width = self.beta_high - self.beta_low
        self.gamma_low = bands['Gamma'][0]
        self.gamma_high = bands['Gamma'][1]
        self.gamma_width = self.gamma_high - self.gamma_low
        self.delta_low = bands['Delta'][0]
        self.delta_high = bands['Delta'][1]
        self.delta_width = self.delta_high - self.delta_low

    def _process_key(self, key):
        if key == Key.esc:
            self.board.stop_stream()
            self.board.disconnect()
            return False
        else:
            print(  f'status:\nFPS: {self.fps:.2f}\n')

    def plot_fft_naive(self, side_length):
        image = np.zeros((4,4,3), np.uint8)
        for i in range(16):
            sample, timestamp = self.inlet.pull_sample()
            sample = np.array(sample[self.delta_low:self.gamma_high])
            sample[:] = sample[:] / sample.max()
            image[int(i / 4), i % 4, 2] = int(255.0 * sum(sample[self.theta_low:self.theta_high]) / self.theta_width)
            image[int(i / 4), i % 4, 1] = int(255.0 * sum(sample[self.alpha_low:self.alpha_high]) /  self.alpha_width)
            image[int(i / 4), i % 4, 0] = int(255.0 * sum(sample[self.gamma_low:self.gamma_high]) / self.gamma_width)
        return cv2.resize(image, (side_length, side_length), interpolation=cv2.INTER_LANCZOS4)

    def get_band_power(self):
        temp = []
        # Get data on 5 bands
        for i in range(5):
            sample, timestamp = self.inlet.pull_sample()
            if(i == self.blue_chan or i == self.green_chan or i == self.red_chan):
                temp += sample

        self.feats = np.atleast_2d(np.array(temp))

    def plot_fft_bashivan(self, side_length):
        self.get_band_power()
        img_array = gen_images(self.locs, self.feats, side_length, edgeless=True)
        return (((np.dstack((img_array[1, :, :], img_array[2, :, :], img_array[0, :, :])) + 1.0) / 2.0)* 255.999).astype(np.uint8)

    def run_threads(self):

        self.keyboard.start()
        #self.bci_stream.start()
