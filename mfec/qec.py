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
        value = a_buffer.stored_value(state)[1]
        if value:
            return value
        if a_buffer.curr_capacity < self.knn:  # TODO delegate to agent
            return float('inf')
        return a_buffer.estimated_value(state, self.knn)

    # TODO batch update
    def update(self, state, action, new_value):
        a_buffer = self.buffers[action]
        state_index, old_value = a_buffer.stored_value(state)
        if state_index:
            a_buffer.insert(state, max(old_value, new_value), state_index)
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

    def stored_value(self, state):
        if self.curr_capacity > 0:
            state_index = self.tree.query([state])[1][0][0]
            # TODO check np.allclose
            if np.allclose(self.states[state_index], state):
                self.lru[state_index] = time.clock()
                return state_index, self.q_values[state_index]
        return None, None  # TODO None, None is ugly

    def estimated_value(self, state, knn):
        closest_neighbors_indices = self.tree.query([state], k=knn)[1][0]
        value = .0
        for state_index in closest_neighbors_indices:
            value += self.q_values[state_index]
            self.lru[state_index] = time.clock()
        return value / knn

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
        self.lru[index] = time.clock()  # TODO delete here?
