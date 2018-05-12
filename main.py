#! /usr/bin/env python2

import cPickle  # TODO cpickle vs json etc.
import logging
import os

import numpy as np
import scipy.misc  # TODO other method than scipy?
from atari_py.ale_python_interface import ALEInterface  # TODO ale vs gym

from mfec.agent import EpisodicControl
from mfec.qec import QEC

# TODO load parameters as json-config
ROMS = "./roms/"
ROM = 'qbert.bin'
QEC_TABLE = ''

TRAINING_FRAMES = 10000
EPOCHS = 100
FRAME_SKIP = 4
DISCOUNT = 1.0
KNN = 11
BUFFER_SIZE = 1000000  # 1000000

EPSILON = 1.0
EPSILON_MIN = .005
EPSILON_DECAY = 10000

SCALE_WIDTH = 84
SCALE_HEIGHT = 84
STATE_DIMENSION = 64

DISPLAY_SCREEN = False
SEED = 1

ale = None
agent = None


def init():
    logging.basicConfig(level=logging.INFO)
    setup_ale()
    setup_agent(len(ale.getMinimalActionSet()))
    train()


# TODO check more variables
def setup_ale():
    global ale
    ale = ALEInterface()
    ale.setInt('random_seed', SEED)
    ale.setBool('display_screen', DISPLAY_SCREEN)  # TODO test this
    ale.setFloat('repeat_action_probability', 0.)  # DON'T TURN IT ON!
    ale.setBool('color_averaging', True)  # TODO compare to max
    ale.loadROM(os.path.join(ROMS, ROM))


def setup_agent(actions):
    global agent
    rng = np.random.RandomState(seed=SEED)  # TODO necessary?
    if QEC_TABLE:
        qec = cPickle.load(open(QEC_TABLE, 'r'))
    else:
        qec = QEC(KNN, BUFFER_SIZE, actions, STATE_DIMENSION)
    # TODO is this projection correct?
    projection = rng.randn(STATE_DIMENSION,
                           SCALE_HEIGHT * SCALE_WIDTH).astype(np.float32)
    agent = EpisodicControl(qec, DISCOUNT, actions, EPSILON, EPSILON_MIN,
                            EPSILON_DECAY, ROM, projection, rng)


def train():
    for epoch in range(1, EPOCHS + 1):
        frames_left = TRAINING_FRAMES
        while frames_left > 0:
            logging.info("Epoch: {}\tFrames: {}".format(epoch, frames_left))
            frames_left -= run_episode(frames_left)
        agent.update_results(epoch)


def run_episode(max_frames):
    episode_reward = 0
    frames = 0

    # TODO terminal if dead?
    while not ale.game_over() and frames < max_frames:
        # TODO observation should be the last 4 frames
        observation = scale(ale.getScreenGrayscale()[:, :, 0])
        action = agent.act(observation)
        reward = sum([ale.act(action) for _ in range(FRAME_SKIP)])
        agent.receive_reward(reward)
        episode_reward += reward
        frames += FRAME_SKIP

    ale.reset_game()  # TODO what is it doing?
    agent.train(episode_reward)
    agent.reset()
    return frames


# TODO refactor to agent?
def scale(observation):
    size = (SCALE_WIDTH, SCALE_HEIGHT)
    return scipy.misc.imresize(observation, size=size)


if __name__ == "__main__":
    init()
