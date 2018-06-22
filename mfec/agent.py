#!/usr/bin/env python3

import pickle
import time

import numpy as np

from mfec.qec import QEC


# TODO use some common agent-interface
class MFECAgent(object):

    def __init__(self, qec_path, buffer_size, k, discount, epsilon, height,
                 width, state_dimension, actions, seed):
        self.rs = np.random.RandomState(seed)

        self.discount = discount
        self.epsilon = epsilon
        self.actions = actions
        self.scale_size = (height, width)

        self.memory = []
        self.qec = self._init_qec(qec_path, buffer_size, k)
        self.projection = self.rs.randn(state_dimension,
                                        height * width).astype(np.float32)

        self.current_state = None
        self.current_action = None
        self.current_time = None

    def _init_qec(self, qec_path, buffer_size, k):
        if qec_path:
            with open(qec_path, 'rb') as qec_file:
                qec = pickle.load(qec_file)
                return qec
        return QEC(self.actions, buffer_size, k)

    def act(self, observation):
        self.current_state = np.dot(self.projection, observation.flatten())
        self.current_time = time.clock()
        if self.rs.random_sample() < self.epsilon:
            self.current_action = self.rs.choice(self.actions)
        else:
            self.current_action = self._exploit()
        return self.current_action

    def _exploit(self):
        values = [
            self.qec.estimate(self.current_state, action, self.current_time)
            for action in self.actions]
        return self.rs.choice(np.argwhere(values == np.max(values)).flatten())

    def receive_reward(self, reward):
        self.memory.append(
            {'state': self.current_state, 'action': self.current_action,
             'reward': reward, 'time_step': self.current_time})

    def train(self):
        value = .0
        for _ in range(len(self.memory)):
            experience = self.memory.pop()
            value = value * self.discount + experience['reward']
            self.qec.update(experience['state'], experience['action'], value,
                            experience['time_step'])
