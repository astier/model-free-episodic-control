#!/usr/bin/env python2

import numpy as np
from sklearn.neighbors.kd_tree import KDTree  # TODO pyflann??


class QEC(object):

    def __init__(self, k, buffer_size, actions):
        self.k = k
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])

    def estimate(self, state, action, time_step):
        a_buffer = self.buffers[action]
        state_index = a_buffer.find_state(state)

        if state_index:
            a_buffer.time_steps[state_index] = time_step  # TODO not in paper
            return a_buffer.values[state_index]

        if len(a_buffer) < self.k:  # TODO delegate to agent
            return float('inf')  # TODO tree.query([state], curr_capacity)???

        value = .0
        neighbors = a_buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += a_buffer.values[neighbor]
            a_buffer.time_steps[neighbor] = time_step  # TODO not in paper
        return value / self.k

    # TODO batch update
    def update(self, state, action, new_value, new_time):
        a_buffer = self.buffers[action]
        state_index = a_buffer.find_state(state)

        if state_index:
            old_value = a_buffer.values[state_index]
            old_time = a_buffer.time_steps[state_index]
            a_buffer.replace(state, max(old_value, new_value),
                             max(old_time, new_time), state_index)
        else:
            a_buffer.add(state, new_value, new_time)


class ActionBuffer(object):

    def __init__(self, capacity):
        self.tree = None  # TODO init here!
        self.capacity = capacity

        # TODO set of dictionaries
        self.states = []
        self.values = []
        self.time_steps = []

    # TODO simply np.allclose over all states?
    def find_state(self, state):
        if len(self.states) > 0:
            neighbor = self.tree.query([state])[1][0][0]
            # TODO check np.allclose
            if np.allclose(self.states[neighbor], state):
                return neighbor
        return None

    def find_neighbors(self, state, k):
        return self.tree.query([state], k)[1][0]

    def add(self, state, value, time_step):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.time_steps.append(time_step)
        else:
            self.replace(state, value, time_step, np.argmin(self.time_steps))

        # TODO smarter tree-update
        # memory ~ n_samples / leaf_size, default_leaf_size=40
        self.tree = KDTree(self.states)

    # TODO mean between states?
    def replace(self, state, value, time_step, index):
        self.states[index] = state
        self.values[index] = value
        self.time_steps[index] = time_step

    # TODO use it!
    def __len__(self):
        return len(self.states)
