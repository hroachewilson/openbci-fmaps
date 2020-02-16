#Deep Dream imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image


import sys
import lib.transforms as transforms
from lib.utils import azim_proj
from lib.BCI import BCI
from lib.DeepDream import DeepDream, calc_loss

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import cv2

# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)


def run_deep_dream_simple(img, steps=100, step_size=0.01):

    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps

            loss, img = dream(img, run_steps, tf.constant(step_size))

            #display.clear_output(wait=True)
            #show(deprocess(img))
            #print ("Step {}, loss {}".format(step, loss))


    result = deprocess(img)
    #display.clear_output(wait=True)
    #show(result)

    return np.array(result)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    band_freqs = {  'Delta' : (0, 4),       # index:    0
                    'Theta' : (4, 7),       # index:    1
                    'Alpha' : (7, 15),      # index:    2
                    'Beta'  : (15, 31),     # index:    3
                    'Gamma' : (31, 45)  }   # index:    4

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # Maximize the activations of these layers
    names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    dream = DeepDream(dream_model)


    bci = BCI(band_freqs, blue=1, green=2, red=4)
    while True:

        # Run LSL in Band Power Mode, 16 channels
        #cv2.imshow('', cv2.resize(bci.plot_fft_bashivan(50), (600,600), interpolation=cv2.INTER_CUBIC))
        #cv2.waitKey(1)

        # Run LSL in FFT Mode, 125 channels
        #cv2.imshow('', bci.plot_fft_naive(600))
        #cv2.waitKey(1)
        img = bci.plot_fft_bashivan(50)
        img = run_deep_dream_simple(img=img, steps=100, step_size=0.01)
        cv2.imshow('', cv2.resize(img, (1000,1000), interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(1)


    print('done')
    exit(0)
