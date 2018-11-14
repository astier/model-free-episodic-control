#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors.kd_tree import KDTree


class QEC:

    def __init__(self, actions, buffer_size, k):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])
        self.k = k

    def estimate(self, state, action):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index:
            return buffer.values[state_index]
        if len(buffer) <= self.k:
            return float('inf')

        value = .0
        neighbors = buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += buffer.values[neighbor]
        return value / self.k

    def update(self, state, action, value, time):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time)


class ActionBuffer:

    def __init__(self, capacity):
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.times = []

    def find_state(self, state):
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            if np.allclose(self.states[neighbor_idx], state):
                return neighbor_idx
        return None

    def find_neighbors(self, state, k):
        return self._tree.query([state], k)[1][0] if self._tree else []

    def add(self, state, value, time):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(self.states)

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self):
        return len(self.states)
