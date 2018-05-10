#! /usr/bin/env python2

import cPickle
import logging
import os

import numpy as np
from atari_py.ale_python_interface import ALEInterface

from mfec.qec import QEC
from mfec.agent import EpisodicControl
from mfec.experiment import Experiment

ROMS = "./roms/"
ROM = 'qbert.bin'
STEPS_PER_EPOCH = 500  # 10000
EPOCHS = 3  # 5000
STEPS_PER_TEST = 0
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0
KNN = 11
DISCOUNT = 1.0
BUFFER_SIZE = 100000  # 1000000
EPSILON = 1.0
EPSILON_MIN = .005
EPSILON_DECAY = 10000
RESIZE_METHOD = 'scale'
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
STATE_DIMENSION = RESIZED_HEIGHT * RESIZED_WIDTH
PROJECTION_DIMENSION = 64
DEATH_ENDS_EPISODE = True
MAX_START_NULLOPS = 30
DISPLAY_SCREEN = False
QEC_TABLE = ''
SEED = 1


def main():
    logging.basicConfig(level=logging.INFO)
    rng = np.random.RandomState(seed=SEED)  # TODO rng necessary?
    env = setup_environment(rng)
    agent = setup_agent(len(env.getMinimalActionSet()), rng)
    Experiment(env, agent, RESIZED_WIDTH, RESIZED_HEIGHT, RESIZE_METHOD,
               EPOCHS, STEPS_PER_EPOCH, STEPS_PER_TEST, FRAME_SKIP,
               DEATH_ENDS_EPISODE, MAX_START_NULLOPS, rng).run()


def setup_environment(rng):  # TODO ale vs gym
    env = ALEInterface()
    env.setInt('random_seed', rng.randint(1000))
    env.setBool('display_screen', DISPLAY_SCREEN)
    env.loadROM(os.path.join(ROMS, ROM))
    env.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
    return env


def setup_agent(num_actions, rng):
    return EpisodicControl(load_qec_table(num_actions, rng), DISCOUNT,
                           num_actions, EPSILON, EPSILON_MIN,
                           EPSILON_DECAY, ROM, rng)


# TODO cpickle vs json etc.
def load_qec_table(num_actions, rng):
    if QEC_TABLE:
        return cPickle.load(open(QEC_TABLE, 'r'))
    return QEC(KNN, PROJECTION_DIMENSION, STATE_DIMENSION, BUFFER_SIZE,
               num_actions, rng)


if __name__ == "__main__":
    main()
