#! /usr/bin/env python2

import numpy as np

from action_buffer import ActionBuffer


class QEC(object):

    def __init__(self, knn, projection_dim, state_dim, buffer_size,
                 num_actions, rng):
        self.knn = knn
        self.rng = rng
        self.buffers = [ActionBuffer(buffer_size, projection_dim)
                        for _ in range(num_actions)]
        # TODO Gauss and others?
        self.projection = rng.randn(projection_dim, state_dim).astype(
            np.float32)

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
        # TODO: Consider implementing as -inf or filling up action-buffers
        # randomly before anything else
        elif a_buffer.curr_capacity < self.knn:
            return float('inf')

        return a_buffer.knn_value(state, self.knn)

    def update(self, s, a, r):
        """s is 84*84*3;  a is 0 to num_actions; r is reward"""
        state = np.dot(self.projection, s.flatten())
        q_value = self.buffers[a].peek(state, r, modify=True)
        if q_value is None:
            self.buffers[a].add(state, r)
