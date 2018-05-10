#! /usr/bin/env python2

import cPickle
import logging
import os

import atari_py.ale_python_interface
import numpy as np

import agent
import experiment
import qec

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

    # TODO rng necessary?
    rng = np.random.RandomState(seed=SEED)
    environment = setup_environment(rng)
    agent = setup_agent(len(environment.getMinimalActionSet()), rng)
    experiment.ALEExperiment(environment, agent, RESIZED_WIDTH, RESIZED_HEIGHT,
                             RESIZE_METHOD, EPOCHS, STEPS_PER_EPOCH,
                             STEPS_PER_TEST, FRAME_SKIP, DEATH_ENDS_EPISODE,
                             MAX_START_NULLOPS, rng).run()


# TODO ale vs gym
def setup_environment(rng):
    environment = atari_py.ale_python_interface.ALEInterface()
    environment.setInt('random_seed', rng.randint(1000))
    environment.setBool('display_screen', DISPLAY_SCREEN)
    environment.loadROM(os.path.join(ROMS, ROM))
    environment.setFloat('repeat_action_probability',
                         REPEAT_ACTION_PROBABILITY)
    return environment


def setup_agent(num_actions, rng):
    qec_table = load_qec_table(num_actions, rng)
    return agent.EpisodicControl(qec_table, DISCOUNT, num_actions,
                                 EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
                                 ROM, rng)


# TODO cpickle vs json etc.
def load_qec_table(num_actions, rng):
    if QEC_TABLE:
        return cPickle.load(open(QEC_TABLE, 'r'))
    return qec.QEC(KNN, STATE_DIMENSION, PROJECTION_TYPE,
                   RESIZED_WIDTH * RESIZED_HEIGHT,
                   BUFFER_SIZE, num_actions, rng)


if __name__ == "__main__":
    main()
