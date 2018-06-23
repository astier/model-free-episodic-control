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

        self.discount = discount
        self.epsilon = epsilon
        self.actions = actions
        self.scale_size = (height, width)

        self.memory = []
        self.qec = QEC(self.actions, buffer_size, k)
        self.projection = self.rs.randn(state_dimension,
                                        height * width).astype(np.float32)

        self.current_state = None
        self.current_action = None
        self.current_step = 0

    def choose_action(self, observation):
        self.current_step += 1
        self.current_state = np.dot(self.projection, observation.flatten())
        if self.rs.random_sample() < self.epsilon:
            self.current_action = self.rs.choice(self.actions)
        else:
            self.current_action = self._exploit()
        return self.current_action

    def _exploit(self):
        values = [
            self.qec.estimate(self.current_state, action, self.current_step)
            for action in self.actions]
        return self.rs.choice(np.argwhere(values == np.max(values)).flatten())

    def receive_reward(self, reward):
        self.memory.append(
            {'state': self.current_state, 'action': self.current_action,
             'reward': reward, 'step': self.current_step})

    def train(self):
        value = .0
        for _ in range(len(self.memory)):
            experience = self.memory.pop()
            value = value * self.discount + experience['reward']
            self.qec.update(experience['state'], experience['action'], value,
                            experience['step'])

    def save(self, results_dir):
        with open(os.path.join(results_dir, 'agent.pkl'), 'wb') as file:
            pickle.dump(self, file, 2)

    @staticmethod
    def load(agent_path):
        with open(agent_path, 'rb') as qec_file:
            return pickle.load(qec_file)
