#!/usr/bin/env python2

import cPickle
import time

import numpy as np
from scipy.misc.pilutil import imresize

from qec import QEC


# TODO use some common agent-interface
class MFECAgent(object):

    def __init__(self, qec_path, buffer_size, k, discount, epsilon, width,
                 height, state_dimension, actions, seed):
        self.discount = discount
        self.epsilon = epsilon

        self.scale_size = (width, height)
        self.actions = actions
        self.memory = []

        self.current_state = None
        self.current_action = None
        self.current_time = None

        self.rng = np.random.RandomState(seed)

        self.qec = self._init_qec(qec_path, buffer_size, k, width, height,
                                  state_dimension)

    def _init_qec(self, qec_path, size, k, width, height, dimension):
        if qec_path:
            return cPickle.load(open(qec_path, 'r'))
        projection = self.rng.randn(dimension, width * height).astype(
            np.float32)
        return QEC(self.actions, size, k, projection)

    def act(self, observation):
        """Choose an action for the given observation."""
        self.current_state = self._process(observation)
        self.current_time = time.clock()
        if self.rng.rand() > self.epsilon:
            self.current_action = self._exploit()
        else:
            self.current_action = self.rng.choice(self.actions)
        return self.current_action

    def _process(self, observation):
        gray_scale = np.mean(observation, axis=2)
        rescaled = imresize(gray_scale, size=self.scale_size)
        return self.qec.project(rescaled)

    def _exploit(self):
        """Determine the action with the highest Q-value. If multiple
        actions with the the highest value exist then choose from this set
        of actions randomly."""
        action_values = [
            self.qec.estimate(self.current_state, action, self.current_time)
            for action in self.actions]
        best_value = np.max(action_values)
        best_actions = np.argwhere(action_values == best_value).flatten()
        return self.rng.choice(best_actions)

    def receive_reward(self, reward):
        """Store (state, action, reward) tuple in memory."""
        self.memory.append(
            {'state': self.current_state, 'action': self.current_action,
             'reward': reward, 'time_step': self.current_time})

    # TODO batch-update
    def train(self):
        """Update Q-Values via backward-replay."""
        value = .0
        for _ in range(len(self.memory)):
            experience = self.memory.pop()
            value = value * self.discount + experience['reward']
            self.qec.update(experience['state'], experience['action'], value,
                            experience['time_step'])
