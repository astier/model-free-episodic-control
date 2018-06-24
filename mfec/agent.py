#!/usr/bin/env python3

import os.path
import pickle

import numpy as np

from mfec.qec import QEC


# TODO use some common agent-interface
class MFECAgent:

    def __init__(self, buffer_size, k, discount, epsilon, height,
                 width, state_dimension, actions, seed):
        self.rs = np.random.RandomState(seed)
        self.memory = []
        self.actions = actions
        self.qec = QEC(self.actions, buffer_size, k)
        self.projection = self.rs.randn(state_dimension,  # TODO float16?
                                        height * width).astype(np.float32)
        self.discount = discount
        self.epsilon = epsilon

        self.state = np.empty(state_dimension, self.projection.dtype)
        self.action = int
        self.time = 0  # TODO ordering instead?

    def choose_action(self, observation):
        self.time += 1
        self.state = np.dot(self.projection, observation.flatten())

        if self.rs.random_sample() < self.epsilon:  # explore
            self.action = self.rs.choice(self.actions)

        else:  # exploit
            values = [self.qec.estimate(self.state, action) for action in
                      self.actions]
            best_actions = np.argwhere(values == np.max(values)).flatten()
            self.action = self.rs.choice(best_actions)

        return self.action

    def receive_reward(self, reward):
        self.memory.append(
            {'state': self.state, 'action': self.action, 'reward': reward,
             'time': self.time})

    def train(self):
        value = .0
        for _ in range(len(self.memory)):
            experience = self.memory.pop()
            value = value * self.discount + experience['reward']
            self.qec.update(experience['state'], experience['action'], value,
                            experience['time'])

    def save(self, results_dir):
        with open(os.path.join(results_dir, 'agent.pkl'), 'wb') as file:
            pickle.dump(self, file, 2)

    @staticmethod
    def load(agent_path):
        with open(agent_path, 'rb') as qec_file:
            return pickle.load(qec_file)
