# general imports
import numpy as np
from collections import deque
import time
import threading
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
        self.bci_board = cython_board

    def read_keyboard(self):
        while True:
            with Listener(on_press=self._process_key) as listener:
                listener.join()

    def _process_key(self, key):
        if key == Key.esc:
            self.bci_board.stop_stream()
            print('stopped_stream')
            return False

    def read_bci(self, sample):
        # main body
        self.sequence = np.roll(self.sequence, 1, 0)
        self.sequence[0, ...] = sample.channels_data

        self.fps_counter.append(time.time() - self.previous_time)
        self.previous_time = time.time()
        print(  f'FPS: {1/(sum(self.fps_counter)/len(self.fps_counter)):.2f}')

    def run_threads(self):
        threading.Thread(target=self.read_keyboard).start()
        threading.Thread(target=self.bci_board.start_stream(self.read_bci)).start()

        
if __name__ == '__main__':
    bci = BCI(OpenBCICyton(daisy=True))
    bci.run_threads()
    #bci.read_stream()
