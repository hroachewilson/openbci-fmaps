# general imports
import numpy as np
from collections import deque
import time, sys
from threading import Thread
from pynput.keyboard import Key, Listener

# openbci imports
from pyOpenBCI import OpenBCICyton

# scikit-cuda imports
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft


class BCI:
    def __init__(self, cython_board):
        self.sequence = np.zeros((512, 16))
        self.fps_counter = deque(maxlen=50)
        self.previous_time = time.time()
        self.fps = 0
        self.board = cython_board

        # Thread objects
        self.keyboard = Listener(on_press=self._process_key)
        self.bci_stream = Thread(target=self.board.start_stream, args=(self._read_bci,))

    def _process_key(self, key):
        if key == Key.esc:
            self.board.stop_stream()
            self.board.disconnect()
            return False
        else:
            print(  f'status:\nFPS: {self.fps:.2f}\n')

    def _read_bci(self, sample):

        # put data in sequence object
        self.sequence = np.roll(self.sequence, 1, 0)
        self.sequence[0, ...] = sample.channels_data

        # calculate fps
        self.fps_counter.append(time.time() - self.previous_time)
        self.previous_time = time.time()
        self.fps = 1/(sum(self.fps_counter)/len(self.fps_counter))

        # calculate cu_fft


    def calculate_fft(self):

    def run_threads(self):

        self.keyboard.start()
        self.bci_stream.start()


if __name__ == '__main__':
    bci = BCI(OpenBCICyton(daisy=True))
    bci.run_threads()
    print('done')
