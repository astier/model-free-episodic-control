#!/usr/bin/env python2

import os
import sys

import numpy as np
from ale_python_interface import ALEInterface  # TODO gym
from scipy.misc.pilutil import imresize

from mfec.agent import MFECAgent
from mfec.qec import QEC
from mfec.utils import Utils

# TRAINING-PARAMETERS
# TODO store parameters in json-file
ROM_FILE_NAME = 'qbert.bin'
AGENT_PATH = 'example_agent_rambo.pkl'
SAVE_AGENT = True

DISPLAY_SCREEN = False
PLAY_SOUND = False  # Note: Sound doesn't work on OSX anyway

EPOCHS = 2
FRAMES_PER_EPOCH = 10000
SEED = None

# HYPERPARAMETERS
ACTION_BUFFER_SIZE = 1000000
FRAMES_PER_ACTION = 4
K = 11
DISCOUNT = 1
EPSILON = .005

SCALE_WIDTH = 84
SCALE_HEIGHT = 84
STATE_DIMENSION = 64

ale = None
agent = None
utils = None


def main():
    global utils, ale, agent
    np.random.seed(SEED)
    utils = Utils(ROM_FILE_NAME, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    ale = create_ale()
    agent = create_agent()
    run()


def create_ale():
    env = ALEInterface()
    env.setInt('random_seed', np.random.randint(0, 1000000))
    env.setFloat('repeat_action_probability', 0)  # DON'T TURN IT ON!
    env.setBool('color_averaging', True)  # TODO paper?

    if DISPLAY_SCREEN:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            env.setBool('sound', False)  # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            env.setBool('sound', PLAY_SOUND)
    env.setBool('display_screen', DISPLAY_SCREEN)

    env.loadROM(os.path.join('roms', ROM_FILE_NAME))
    return env


def create_agent():
    actions = range(len(ale.getMinimalActionSet()))

    if AGENT_PATH:
        qec = utils.load_agent(AGENT_PATH)

    else:
        projection = np.random.randn(STATE_DIMENSION,
                                     SCALE_HEIGHT * SCALE_WIDTH).astype(
            np.float32)
        qec = QEC(actions, ACTION_BUFFER_SIZE, K, projection)

    return MFECAgent(qec, DISCOUNT, actions, EPSILON)


def run():
    for epoch in range(1, EPOCHS + 1):
        frames_left = FRAMES_PER_EPOCH

        while frames_left > 0:
            episode_frames, episode_reward = run_episode()
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)

        utils.end_epoch()
        if SAVE_AGENT:
            utils.save_agent(agent)


def run_episode():
    episode_frames = 0
    episode_reward = 0

    while not ale.game_over():
        observation = get_observation()
        action = agent.act(observation)
        # TODO stop if dead
        reward = sum([ale.act(action) for _ in range(FRAMES_PER_ACTION)])

        agent.receive_reward(reward)
        episode_reward += reward
        episode_frames += FRAMES_PER_ACTION

    agent.train()
    ale.setInt('random_seed', np.random.randint(0, 1000000))
    ale.reset_game()
    return episode_frames, episode_reward


def get_observation():
    observation = ale.getScreenGrayscale()[:, :, 0]
    return imresize(observation, size=(SCALE_WIDTH, SCALE_HEIGHT))


if __name__ == "__main__":
    main()
