#! /usr/bin/env python2

import cPickle
import logging
import os

import numpy as np
import scipy.misc
from atari_py.ale_python_interface import ALEInterface

from mfec.agent import EpisodicControl
from mfec.qec import QEC

# TODO parameters as json-config
ROMS = "./roms/"
ROM = 'qbert.bin'
STEPS_PER_EPOCH = 5000  # 10000
EPOCHS = 3  # 5000
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0.
KNN = 11
DISCOUNT = 1.0
BUFFER_SIZE = 1000000  # 1000000
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

ale = None
agent = None


def init():
    logging.basicConfig(level=logging.INFO)
    setup_ale()
    setup_agent(len(ale.getMinimalActionSet()))
    train()


def setup_ale():  # TODO ale vs gym
    global ale
    ale = ALEInterface()
    # TODO check more variables
    ale.setInt('random_seed', SEED)
    ale.setBool('display_screen', DISPLAY_SCREEN)
    ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
    ale.setBool('color_averaging', True)
    ale.loadROM(os.path.join(ROMS, ROM))


def setup_agent(actions):  # TODO cpickle vs json etc.
    global agent
    rng = np.random.RandomState(seed=SEED)
    if QEC_TABLE:
        qec = cPickle.load(open(QEC_TABLE, 'r'))
    else:
        # TODO Gauss and others?
        projection_matrix = rng.randn(PROJECTION_DIM, STATE_DIM).astype(
            np.float32)
        qec = QEC(KNN, BUFFER_SIZE, actions, projection_matrix)
    agent = EpisodicControl(qec, DISCOUNT, actions, EPSILON, EPSILON_MIN,
                            EPSILON_DECAY, ROM, rng)


def train():
    for epoch in range(1, EPOCHS + 1):
        # TODO steps vs frames
        steps_left = STEPS_PER_EPOCH
        while steps_left > 0:
            logging.info("Epoch: {}\tSteps: {}".format(epoch, steps_left))
            steps_left -= run_episode(steps_left)
        agent.end_epoch(epoch)


def run_episode(max_steps):
    total_reward = 0
    steps = 0
    agent.start_episode()

    while not ale.game_over() and steps < max_steps:
        observation = ale.getScreenGrayscale()[:, :, 0]
        state = projection(observation)
        action = agent.act(state)
        reward = sum([ale.act(action) for _ in range(FRAME_SKIP)])
        agent.add_experience(state, action, reward)
        total_reward += reward
        steps += 1

    ale.reset_game()
    agent.end_episode(total_reward)
    return steps


# TODO projection should happen here!
def projection(observation):
    size = (RESIZE_WIDTH, RESIZE_HEIGHT)
    # TODO other method than scipy?
    return scipy.misc.imresize(observation, size=size)


if __name__ == "__main__":
    init()
