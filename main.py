#! /usr/bin/ale python2

import cPickle
import logging
import os

import numpy as np
from atari_py.ale_python_interface import ALEInterface

from mfec.qec import QEC
from mfec.agent import EpisodicControl
from mfec.experiment import Experiment

# TODO parameters as json-config
ROMS = "./roms/"
ROM = 'qbert.bin'
STEPS_PER_EPOCH = 5000  # 10000
EPOCHS = 30  # 5000
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0
KNN = 11
DISCOUNT = 1.0
BUFFER_SIZE = 1000000  # 1000000
EPSILON = 1.0
EPSILON_MIN = .005
EPSILON_DECAY = 10000
RESIZE_METHOD = 'scale'  # TODO whats that?
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
STATE_DIM = RESIZED_HEIGHT * RESIZED_WIDTH
PROJECTION_DIM = 64
DEATH_ENDS_EPISODE = True
MAX_NULLOPS = 30
DISPLAY_SCREEN = False
QEC_TABLE = ''
SEED = 1


def main():
    logging.basicConfig(level=logging.INFO)
    ale = setup_ale()
    actions = len(ale.getMinimalActionSet())
    rng = np.random.RandomState(seed=SEED)
    qec = load_qec_table(actions, rng)
    agent = EpisodicControl(qec, DISCOUNT, actions, EPSILON, EPSILON_MIN,
                            EPSILON_DECAY, ROM, rng)
    Experiment(ale, agent, RESIZED_WIDTH, RESIZED_HEIGHT, RESIZE_METHOD,
               EPOCHS, STEPS_PER_EPOCH, FRAME_SKIP, DEATH_ENDS_EPISODE,
               MAX_NULLOPS, rng).run()


def setup_ale():  # TODO ale vs gym
    ale = ALEInterface()
    ale.setInt('random_seed', SEED)
    ale.setBool('display_screen', DISPLAY_SCREEN)
    ale.loadROM(os.path.join(ROMS, ROM))
    ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
    return ale


def load_qec_table(actions, rng):  # TODO cpickle vs json etc.
    if QEC_TABLE:
        return cPickle.load(open(QEC_TABLE, 'r'))
    # TODO Gauss and others?
    projection = rng.randn(PROJECTION_DIM, STATE_DIM).astype(np.float32)
    return QEC(KNN, BUFFER_SIZE, actions, projection)


if __name__ == "__main__":
    main()
