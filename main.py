#!/usr/bin/env python2

import gym

from mfec.agent import MFECAgent
from mfec.utils import Utils

# TODO store parameters in json-file
ENVIRONMENT = 'Qbert-v0'  # Check https://gym.openai.com/envs/#atari
AGENT_PATH = ''
SAVE_AGENT = True
DISPLAY = False
EPOCHS = 25
FRAMES_PER_EPOCH = 40000

ACTION_BUFFER_SIZE = 1000000
FRAMES_PER_ACTION = 4
K = 11
DISCOUNT = 1
EPSILON = .005

SCALE_WIDTH = 84
SCALE_HEIGHT = 84
STATE_DIMENSION = 64
COLOR_AVERAGING = True  # TODO file issue & test

env = None
agent = None
utils = None


def init():
    global env, agent, utils
    env = init_env()
    agent = MFECAgent(AGENT_PATH, ACTION_BUFFER_SIZE, K, DISCOUNT, EPSILON,
                      SCALE_WIDTH, SCALE_HEIGHT, STATE_DIMENSION,
                      range(env.action_space.n))
    utils = Utils(ENVIRONMENT, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH,
                  SAVE_AGENT, agent)


def init_env():
    e = gym.make(ENVIRONMENT)
    e.env.frameskip = FRAMES_PER_ACTION
    e.env.ale.setBool('color_averaging', COLOR_AVERAGING)
    return e


# TODO 30 initial states
def run_algorithm():
    for epoch in range(1, EPOCHS + 1):
        frames_left = FRAMES_PER_EPOCH
        while frames_left > 0:
            episode_frames, episode_reward = run_episode()
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        utils.end_epoch()
    env.close()


def run_episode():
    episode_frames = 0
    episode_reward = 0

    observation = env.reset()
    done = False
    while not done:

        if DISPLAY:  # TODO slow down
            env.render()

        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)

        agent.receive_reward(reward)
        episode_reward += reward
        episode_frames += FRAMES_PER_ACTION

    agent.train()
    return episode_frames, episode_reward


if __name__ == "__main__":
    init()
    run_algorithm()
