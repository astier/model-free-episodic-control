#! /usr/bin/env python2

__author__ = 'frankhe'

import launcher
import sys


class Defaults(object):
    STEPS_PER_EPOCH = 300  # 10000
    EPOCHS = 2  # 5000
    STEPS_PER_TEST = 0

    BASE_ROM_PATH = "./roms/"
    ROM = 'qbert.bin'
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    K_NEAREST_NEIGHBOR = 11
    EC_DISCOUNT = 1.0
    BUFFER_SIZE = 1000000
    DIMENSION_OF_STATE = 64
    PROJECTION_TYPE = 'random'  # or VAE
    EPSILON_START = 1.0
    EPSILON_MIN = .005
    EPSILON_DECAY = 10000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False


if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
