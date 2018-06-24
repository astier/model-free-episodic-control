#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors.kd_tree import KDTree  # TODO pyflann?? paper?


class QEC:

    def __init__(self, actions, buffer_size, k):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])
        self.k = k

    def estimate(self, state, action, step):
        a_buffer = self.buffers[action]
        state_index = a_buffer.find_state(state)

        if state_index:
            a_buffer.steps[state_index] = step  # TODO not in paper
            return a_buffer.values[state_index]
        if len(a_buffer) <= self.k:  # TODO init-phase
            return float('inf')

        value = .0
        neighbors = a_buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += a_buffer.values[neighbor]
            a_buffer.steps[neighbor] = step  # TODO not in paper

        return value / self.k

    # TODO batch update
    def update(self, state, action, new_value, new_step):
        a_buffer = self.buffers[action]
        state_index = a_buffer.find_state(state)
        if state_index:
            old_value = a_buffer.values[state_index]
            old_time = a_buffer.steps[state_index]
            a_buffer.replace(state, max(old_value, new_value),
                             max(old_time, new_step), state_index)
        else:
            a_buffer.add(state, new_value, new_step)


class ActionBuffer:

    def __init__(self, capacity):
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.steps = []

    def find_state(self, state):
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            if np.allclose(self.states[neighbor_idx], state):
                return neighbor_idx
        return None

    def find_neighbors(self, state, k):
        if self._tree:
            return self._tree.query([state], k)[1][0]
        return []

    def add(self, state, value, step):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.steps.append(step)
        else:
            self.replace(state, value, step, np.argmin(self.steps))

        # TODO smarter tree-update
        # memory ~ n_samples / leaf_size, default_leaf_size=40
        self._tree = KDTree(self.states)

    def replace(self, state, value, step, index):
        self.states[index] = state
        self.values[index] = value
        self.steps[index] = step

    def __len__(self):
        return len(self.states)
