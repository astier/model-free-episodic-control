#!/usr/bin/env python3

import os
import random
import time

import gym
import numpy as np
from scipy.misc.pilutil import imresize

from mfec.agent import MFECAgent
from mfec.utils import Utils

# TODO store parameters in json-file
ENVIRONMENT = 'Qbert-v0'  # More games at: https://gym.openai.com/envs/#atari
AGENT_PATH = ''
RENDER = False
RENDER_SPEED = .04

EPOCHS = 2
FRAMES_PER_EPOCH = 10000
SEED = 42

ACTION_BUFFER_SIZE = 1000000
K = 11
DISCOUNT = 1
EPSILON = .005
FRAMESKIP = 4
REPEAT_ACTION_PROB = .0

SCALE_HEIGHT = 84
SCALE_WIDTH = 84
STATE_DIMENSION = 64

env = None
agent = None
utils = None
agent_dir = None


def run_algorithm():
    frames_left = 0
    for _ in range(EPOCHS):

        frames_left += FRAMES_PER_EPOCH
        while frames_left > 0:
            episode_frames, episode_reward = run_episode()
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)

        utils.end_epoch()
        agent.save(agent_dir)

    utils.close()
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
            time.sleep(RENDER_SPEED)

        action = agent.choose_action(preprocess(observation))
        observation, reward, done, _ = env.step(action)
        agent.receive_reward(reward)

        episode_reward += reward
        episode_frames += FRAMESKIP

    agent.train()
    return episode_frames, episode_reward


def preprocess(observation):
    grey = np.mean(observation, axis=2)
    return imresize(grey, size=(SCALE_HEIGHT, SCALE_WIDTH))


if __name__ == '__main__':
    random.seed(SEED)
    env = gym.make(ENVIRONMENT)
    env.env.frameskip = FRAMESKIP
    env.env.ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROB)

    if AGENT_PATH:
        agent = MFECAgent.load(AGENT_PATH)
    else:
        agent = MFECAgent(ACTION_BUFFER_SIZE, K, DISCOUNT, EPSILON,
                          SCALE_HEIGHT, SCALE_WIDTH, STATE_DIMENSION,
                          range(env.action_space.n), SEED)

    execution_time = time.strftime('_%m-%d-%H-%M-%S', time.gmtime())
    agent_dir = os.path.join('agents', ENVIRONMENT + execution_time)
    os.makedirs(os.path.join(agent_dir))

    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    run_algorithm()
