#!/usr/bin/env python2

import numpy as np
from sklearn.neighbors.kd_tree import KDTree


class QEC(object):

    def __init__(self, knn, buffer_size, actions, state_dimension):
        self.knn = knn
        self.buffers = tuple([ActionBuffer(buffer_size, state_dimension)
                              for _ in actions])

    def estimate(self, state, action):
        a_buffer = self.buffers[action]
        q_value = a_buffer.get_state_value(state)
        if q_value is not None:
            return q_value
        if a_buffer.curr_capacity < self.knn:  # TODO delegate to agent
            return float('inf')
        return a_buffer.knn_value(state, self.knn)

    def update(self, state, action, value_new):
        value_old = self.buffers[action].get_state_value(state)
        self.buffers[action].add(state, max(value_old, value_new))


class ActionBuffer(object):

    def __init__(self, capacity, state_dimension):
        self.capacity = capacity
        self.curr_capacity = 0

        # TODO set of dictionaries
        self.states = np.zeros((capacity, state_dimension))
        self.q_values = np.zeros(capacity)
        self.lru = np.zeros(capacity)

        self.tree = None  # TODO init here!
        self.tm = .0

    def get_state_value(self, state):
        if self.curr_capacity > 0:
            # tree = KDTree(self.states[:self.curr_capacity])
            _, closest_neighbors_indices = self.tree.query([state], k=1)
            closest_neighbor_index = closest_neighbors_indices[0][0]
            if np.allclose(self.states[closest_neighbor_index], state):
                self.lru[closest_neighbor_index] = self.tm
                self.tm += .01
                return self.q_values[closest_neighbor_index]
        return None

    def knn_value(self, key, knn):
        # tree = KDTree(self.states[:self.curr_capacity])
        _, ind = self.tree.query([key], k=knn)
        value = .0
        for index in ind[0]:
            value += self.q_values[index]
            self.lru[index] = self.tm
            self.tm += .01
        return value / knn

    def add(self, key, value):
        if self.curr_capacity >= self.capacity:
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.q_values[old_index] = value
            self.lru[old_index] = self.tm
        else:
            self.states[self.curr_capacity] = key
            self.q_values[self.curr_capacity] = value
            self.lru[self.curr_capacity] = self.tm
            self.curr_capacity += 1
        self.tm += 0.01
        self.tree = KDTree(self.states[:self.curr_capacity])
