#!/usr/bin/env python2

import time

import numpy as np
from sklearn.neighbors.kd_tree import KDTree  # TODO pyflann??


class QEC(object):

    def __init__(self, knn, buffer_size, actions, state_dimension):
        self.knn = knn
        self.buffers = tuple([ActionBuffer(buffer_size, state_dimension)
                              for _ in actions])

    def estimate(self, state, action):
        a_buffer = self.buffers[action]

        neighbor = a_buffer.neighbor(state)
        if neighbor:
            a_buffer.lru[neighbor] = time.clock()  # TODO not in mfec paper
            return a_buffer.q_values[neighbor]

        if a_buffer.curr_capacity < self.knn:  # TODO delegate to agent
            return float('inf')  # TODO tree.query([state], curr_capacity)???

        neighbors = a_buffer.tree.query([state], self.knn)[1][0]
        value = .0
        for state_index in neighbors:
            value += a_buffer.q_values[state_index]
            a_buffer.lru[state_index] = time.clock()  # TODO not mfec paper
        return value / self.knn

    # TODO batch update
    def update(self, state, action, new_value):
        a_buffer = self.buffers[action]
        neighbor = a_buffer.neighbor(state)
        if neighbor:
            old_value = a_buffer.q_values[neighbor]
            a_buffer.insert(state, max(old_value, new_value), neighbor)
        else:
            a_buffer.add(state, new_value)


class ActionBuffer(object):

    def __init__(self, capacity, state_dimension):
        self.tree = None
        self.curr_capacity = 0
        self.capacity = capacity
        # TODO set of dictionaries
        self.states = np.zeros((capacity, state_dimension))
        self.q_values = np.zeros(capacity)
        self.lru = np.zeros(capacity)

    def neighbor(self, state):
        if self.curr_capacity > 0:
            neighbor = self.tree.query([state])[1][0]
            # TODO check np.allclose
            if np.allclose(self.states[neighbor], state):
                return neighbor
        return None

    def add(self, state, value):
        if self.curr_capacity < self.capacity:
            self.insert(state, value, self.curr_capacity)
            self.curr_capacity += 1
        else:
            self.insert(state, value, np.argmin(self.lru))

        # TODO smarter tree-update
        # memory ~ n_samples / leaf_size, default_leaf_size=40
        self.tree = KDTree(self.states[:self.curr_capacity])

    # TODO mean between states?
    def insert(self, state, value, index):
        self.states[index] = state
        self.q_values[index] = value
