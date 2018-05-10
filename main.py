#! /usr/bin/env python2


import cPickle
import logging
import os

import atari_py.ale_python_interface
import numpy as np

import ec_agent
import ec_functions
import experiment

ROMS = "./roms/"
ROM = 'qbert.bin'
STEPS_PER_EPOCH = 1000  # 10000
EPOCHS = 3  # 5000
STEPS_PER_TEST = 0
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0
KNN = 11
DISCOUNT = 1.0
BUFFER_SIZE = 100000  # 1000000
STATE_DIMENSION = 64
PROJECTION_TYPE = 'random'  # or VAE
EPSILON_START = 1.0
EPSILON_MIN = .005
EPSILON_DECAY = 10000
RESIZE_METHOD = 'scale'
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
DEATH_ENDS_EPISODE = True
MAX_START_NULLOPS = 30
DISPLAY_SCREEN = False
QEC_TABLE = ''
SEED = 1


def main():
    """Execute a complete training run."""
    logging.basicConfig(level=logging.INFO)
    rng = np.random.RandomState(seed=SEED)
    ale = setup_ale(rng)
    agent = setup_agent(ale, rng)
    experiment.ALEExperiment(ale, agent, RESIZED_WIDTH, RESIZED_HEIGHT,
                             RESIZE_METHOD, EPOCHS, STEPS_PER_EPOCH,
                             STEPS_PER_TEST, FRAME_SKIP, DEATH_ENDS_EPISODE,
                             MAX_START_NULLOPS, rng).run()


def setup_ale(rng):
    ale = atari_py.ale_python_interface.ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))
    ale.setBool('display_screen', DISPLAY_SCREEN)
    ale.loadROM(os.path.join(ROMS, ROM))
    ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
    return ale


def setup_agent(ale, rng):
    num_actions = len(ale.getMinimalActionSet())
    qec_table = load_qec_table(num_actions, rng)
    return ec_agent.EpisodicControl(qec_table, DISCOUNT, num_actions,
                                    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
                                    ROM, rng)


def load_qec_table(num_actions, rng):
    if QEC_TABLE:
        return cPickle.load(open(QEC_TABLE, 'r'))
    return ec_functions.QECTable(KNN, STATE_DIMENSION, PROJECTION_TYPE,
                                 RESIZED_WIDTH * RESIZED_HEIGHT,
                                 BUFFER_SIZE, num_actions, rng)


if __name__ == "__main__":
    main()
