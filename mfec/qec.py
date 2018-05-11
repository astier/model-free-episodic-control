#! /usr/bin/env python2

import numpy as np
from sklearn.neighbors.kd_tree import KDTree


class QEC(object):

    def __init__(self, knn, buffer_size, num_actions, projection):
        self.knn = knn
        self.projection = projection
        self.buffers = [ActionBuffer(buffer_size, projection.shape[0])
                        for _ in range(1, num_actions + 1)]

    def estimate(self, s, a):
        state = np.dot(self.projection, s.flatten())
        a_buffer = self.buffers[a]
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

    def update(self, s, a, r):
        """s is 84*84*3;  a is 0 to num_actions; r is reward"""
        state = np.dot(self.projection, s.flatten())
        q_value = self.buffers[a].peek(state, r, modify=True)
        if q_value is None:
            self.buffers[a].add(state, r)


class ActionBuffer(object):

    def __init__(self, capacity, projection_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, projection_dim))
        self.q_values = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
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
            return 0.0

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
