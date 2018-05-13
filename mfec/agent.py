#!/usr/bin/env python2

import numpy as np


class MFECAgent(object):

    def __init__(self, qec, discount, actions, epsilon, epsilon_min,
                 epsilon_decay, projection, ):
        self.qec = qec
        self.discount = discount
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_rate = self._compute_epsilon_rate(epsilon_decay)
        self.memory = []
        self.current_state = None
        self.current_action = None
        self.projection = projection

    def _compute_epsilon_rate(self, epsilon_decay):
        if epsilon_decay != 0:
            return (self.epsilon - self.epsilon_min) / epsilon_decay
        return 0

    # TODO initialize first knn buffer?
    def act(self, observation):
        """Choose an action for the given observation."""
        self.current_state = np.dot(self.projection, observation.flatten())
        # TODO generator?
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
        if np.random.rand() > self.epsilon:
            self.current_action = self._exploit()
        else:
            self.current_action = np.random.choice(self.actions)
        return self.current_action

    def _exploit(self):
        """Determine the action with the highest q-value."""
        best_value = float('-inf')
        best_action = 0
        for action in self.actions:
            value = self.qec.estimate(self.current_state, action)
            # TODO random choice for multiple actions with same value
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def receive_reward(self, reward):
        """Store (state, action, reward) tuple in memory."""
        self.memory.append(
            {'state': self.current_state, 'action': self.current_action,
             'reward': reward})

    def train(self):
        """Update Q-Values"""
        q_value = 0.
        for i in range(len(self.memory) - 1, -1, -1):
            experience = self.memory[i]
            q_value = q_value * self.discount + experience['reward']
            self.qec.update(experience['state'], experience['action'], q_value)
        self.memory = []
