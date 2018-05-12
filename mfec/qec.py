#! /usr/bin/env python2

import numpy as np
from sklearn.neighbors.kd_tree import KDTree


class QEC(object):

    def __init__(self, knn, buffer_size, actions, state_dimension):
        self.knn = knn
        self.buffers = [ActionBuffer(buffer_size, state_dimension)
                        for _ in range(actions)]

    def estimate(self, state, action):
        a_buffer = self.buffers[action]
        q_value = a_buffer.peek(state, None, modify=False)
        if q_value is not None:
            return q_value
        # If the number of elements in the action-buffer is smaller than k
        # then return an 'infinitely' high reward to make sure that this action
        # gets explored, otherwise an estimation by the knn-algorithm
        # can not be performed and throws an exception,
        # because it needs at least k elements as neighbors.
        # TODO Consider implementing as -inf or filling up action-buffers
        # TODO randomly before anything else
        elif a_buffer.curr_capacity < self.knn:
            return float('inf')
        return a_buffer.knn_value(state, self.knn)

    def update(self, state, action, q_value):
        q_value = self.buffers[action].peek(state, q_value, modify=True)
        if q_value is None:
            self.buffers[action].add(state, q_value)


class ActionBuffer(object):

    def __init__(self, capacity, state_dimension):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dimension))  # TODO vstack
        self.q_values = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = .0
        self.tree = None

    def peek(self, key, value, modify):
        if self.curr_capacity == 0:
            return None

        # tree = KDTree(self.states[:self.curr_capacity])
        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]

        if np.allclose(self.states[ind], key):
            self.lru[ind] = self.tm
            self.tm += 0.01
            if modify:
                self.q_values[ind] = max(self.q_values[ind], value)
            return self.q_values[ind]

        return None

    def knn_value(self, key, knn):
        if self.curr_capacity == 0:
            return .0

        # tree = KDTree(self.states[:self.curr_capacity])
        dist, ind = self.tree.query([key], k=knn)

        value = 0.0
        for index in ind[0]:
            value += self.q_values[index]
            self.lru[index] = self.tm
            self.tm += 0.01

        return value / knn

    def add(self, key, value):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
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
