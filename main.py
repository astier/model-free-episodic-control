#! /usr/bin/env python2

import cPickle
import logging
import os

import numpy as np
from atari_py.ale_python_interface import ALEInterface

from mfec.agent import EpisodicControl
from mfec.experiment import Experiment
from mfec.qec import QEC

# TODO parameters as json-config
ROMS = "./roms/"
ROM = 'qbert.bin'
STEPS_PER_EPOCH = 500  # 10000
EPOCHS = 3  # 5000
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0
KNN = 11
DISCOUNT = 1.0
BUFFER_SIZE = 100000  # 1000000
EPSILON = 1.0
EPSILON_MIN = .005
EPSILON_DECAY = 10000
RESIZE_WIDTH = 84
RESIZE_HEIGHT = 84
STATE_DIM = RESIZE_HEIGHT * RESIZE_WIDTH
PROJECTION_DIM = 64
DISPLAY_SCREEN = False
QEC_TABLE = ''
SEED = 1


def main():  # TODO merge with experiment.py
    logging.basicConfig(level=logging.INFO)
    ale = setup_ale()
    actions = len(ale.getMinimalActionSet())
    rng = np.random.RandomState(seed=SEED)
    qec = load_qec_table(actions, rng)
    agent = EpisodicControl(qec, DISCOUNT, actions, EPSILON, EPSILON_MIN,
                            EPSILON_DECAY, ROM, rng)
    Experiment(ale, agent, RESIZE_WIDTH, RESIZE_HEIGHT, EPOCHS,
               STEPS_PER_EPOCH, FRAME_SKIP).run()


def setup_ale():  # TODO ale vs gym
    ale = ALEInterface()
    ale.setInt('random_seed', SEED)
    ale.setBool('display_screen', DISPLAY_SCREEN)
    ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
    ale.setBool('color_averaging', True)
    ale.loadROM(os.path.join(ROMS, ROM))
    return ale


def load_qec_table(actions, rng):  # TODO cpickle vs json etc.
    if QEC_TABLE:
        return cPickle.load(open(QEC_TABLE, 'r'))
    # TODO Gauss and others?
    projection = rng.randn(PROJECTION_DIM, STATE_DIM).astype(np.float32)
    return QEC(KNN, BUFFER_SIZE, actions, projection)


if __name__ == "__main__":
    main()
