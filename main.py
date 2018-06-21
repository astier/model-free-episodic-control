#!/usr/bin/env python2

import random
import time

import gym
import numpy as np
from scipy.misc.pilutil import imresize

from mfec.agent import MFECAgent
from mfec.utils import Utils

# TODO store parameters in json-file
ENVIRONMENT = 'MsPacman-v0'  # Check https://gym.openai.com/envs/#atari
AGENT_PATH = ''
SAVE_AGENT = True
RENDER = False
RENDER_SLEEP = .04

SEED = 42
EPOCHS = 50
FRAMES_PER_EPOCH = 40000

ACTION_BUFFER_SIZE = 1000000
FRAMESKIP = 4  # TODO test gyms setting (2, 5)
REPEAT_ACTION_PROBABILITY = .0
K = 11
DISCOUNT = 1
EPSILON = .005

SCALE_HEIGHT = 84
SCALE_WIDTH = 84
STATE_DIMENSION = 64

env = None
agent = None
utils = None


def run_algorithm():
    for _ in range(EPOCHS):
        frames_left = FRAMES_PER_EPOCH
        while frames_left > 0:
            episode_frames, episode_reward = run_episode()
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        utils.end_epoch()
    env.close()


def run_episode():  # TODO paper 30 initial states?
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()
    done = False

    while not done:

        if RENDER:
            env.render()
            time.sleep(RENDER_SLEEP)

        action = agent.act(preprocess(observation))
        observation, reward, done, _ = env.step(action)
        agent.receive_reward(reward)

        episode_reward += reward
        episode_frames += FRAMESKIP

    agent.train()
    return episode_frames, episode_reward


def preprocess(observation):
    grey = np.mean(observation, axis=2)
    return imresize(grey, size=(SCALE_HEIGHT, SCALE_WIDTH))


if __name__ == "__main__":
    random.seed(SEED)

    env = gym.make(ENVIRONMENT)
    env.env.frameskip = FRAMESKIP
    env.env.ale.setFloat('repeat_action_probability',
                         REPEAT_ACTION_PROBABILITY)

    agent = MFECAgent(AGENT_PATH, ACTION_BUFFER_SIZE, K, DISCOUNT, EPSILON,
                      SCALE_HEIGHT, SCALE_WIDTH, STATE_DIMENSION,
                      range(env.action_space.n), SEED)
    utils = Utils(ENVIRONMENT, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH,
                  SAVE_AGENT, agent)

    run_algorithm()
