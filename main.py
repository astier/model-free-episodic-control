#! /usr/bin/env python2

import logging
import os
import time

import numpy as np
import scipy.misc  # TODO other method than scipy?
from atari_py.ale_python_interface import ALEInterface  # TODO ale vs gym

from mfec.agent import MFECAgent
from mfec.qec import QEC
from mfec.utils import Utils

# TODO load parameters as json-config
ROM_FILE_NAME = 'qbert.bin'
SAVE_QEC_TABLE = False
QEC_TABLE_PATH = None

EPOCHS = 5
FRAMES_PER_EPOCH = 1000
FRAMES_PER_ACTION = 4
DISCOUNT = 1.0
KNN = 11
ACTION_BUFFER_SIZE = 100000  # 1000000

EPSILON = 1.0
EPSILON_MIN = .005
EPSILON_DECAY = 10000

SCALE_WIDTH = 84
SCALE_HEIGHT = 84
STATE_DIMENSION = 64

DISPLAY_SCREEN = False
SEED = 42

ale = None
agent = None
utils = None


# TODO define better dir-name
def main():
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO)
    setup_utils()
    setup_ale()
    setup_agent()
    run()


def setup_utils():
    global utils
    execution_time = time.strftime("_%m-%d-%H-%M-%S", time.gmtime())
    result_dir = ROM_FILE_NAME.split('.')[0] + execution_time
    results_dir = os.path.join('results', result_dir)
    os.makedirs(results_dir)
    utils = Utils(results_dir)


# TODO check more variables
def setup_ale():
    global ale
    ale = ALEInterface()
    ale.setInt('random_seed', SEED)
    ale.setBool('display_screen', DISPLAY_SCREEN)  # TODO test this
    ale.setFloat('repeat_action_probability', 0.)  # DON'T TURN IT ON!
    ale.setBool('color_averaging', True)  # TODO compare to max
    ale.loadROM(os.path.join('roms', ROM_FILE_NAME))


def setup_agent():
    global agent
    actions = len(ale.getMinimalActionSet())
    if QEC_TABLE_PATH:
        qec = utils.load_agent(QEC_TABLE_PATH)
    else:
        qec = QEC(KNN, ACTION_BUFFER_SIZE, actions, STATE_DIMENSION)

    # TODO is this projection correct?
    projection = np.random.randn(STATE_DIMENSION,
                                 SCALE_HEIGHT * SCALE_WIDTH).astype(np.float32)
    agent = MFECAgent(qec, DISCOUNT, actions, EPSILON, EPSILON_MIN,
                      EPSILON_DECAY, projection)


def run():
    for epoch in range(1, EPOCHS + 1):
        frames_left = FRAMES_PER_EPOCH
        while frames_left > 0:
            logging.info("Epoch: {}\tFrames: {}".format(epoch, frames_left))
            frames_left -= run_episode(frames_left)
        utils.save_results(epoch)
        if SAVE_QEC_TABLE:
            utils.save_agent(epoch, agent)


def run_episode(max_frames):
    episode_reward = 0
    frames = 0

    # TODO terminal if dead?
    while not ale.game_over() and frames < max_frames:
        # TODO observation should be the last 4 frames?
        observation = scale(ale.getScreenGrayscale()[:, :, 0])
        action = agent.act(observation)
        reward = sum([ale.act(action) for _ in range(FRAMES_PER_ACTION)])
        agent.receive_reward(reward)
        episode_reward += reward
        frames += FRAMES_PER_ACTION

    ale.reset_game()  # TODO what is it doing?
    agent.train()
    utils.update_results(episode_reward)
    return frames


def scale(observation):
    size = (SCALE_WIDTH, SCALE_HEIGHT)
    return scipy.misc.imresize(observation, size=size)


if __name__ == "__main__":
    main()
