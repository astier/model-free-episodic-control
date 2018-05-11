#! /usr/bin/env python2

import logging
import numpy as np
import scipy.misc


class Experiment(object):

    def __init__(self, ale, agent, resize_width, resize_height, epochs,
                 steps_per_epoch, frame_skip, death_ends_episode):
        self.ale = ale
        self.agent = agent
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        self.actions = ale.getMinimalActionSet()
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.width, self.height = ale.getScreenDims()
        self.frame_buffer = np.empty((2, self.height, self.width),
                                     dtype=np.uint8)
        self.buffer_count = 0  # TODO get rid of this somehow!
        self.death = False  # Last episode ended because agent died

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self.death = False
            steps_left = self.steps_per_epoch
            while steps_left > 0:
                logging.info("Epoch: {}\tSteps: {}".format(epoch, steps_left))
                steps_left -= self.run_episode(steps_left)
            self.agent.finish_epoch(epoch)  # TODO refactor to experiment

    def run_episode(self, max_steps):
        if not self.death or self.ale.game_over():
            self.ale.reset_game()

        # TODO fill during initialization
        for _ in range(self.frame_buffer.shape[0]):  # fill the frame-buffer
            self.act(0)

        start_lives = self.ale.lives()
        game_over = False
        reward = None
        action = self.agent.start_episode(self.get_observation())
        steps = 0

        while not game_over and steps < max_steps:
            reward = sum(self.act(self.actions[action]) for _ in
                         range(self.frame_skip))
            self.death = (self.death_ends_episode and
                          self.ale.lives() < start_lives)
            game_over = self.ale.game_over() or self.death
            action = self.agent.step(reward, self.get_observation())
            steps += 1

        self.agent.end_episode(reward, game_over)
        return steps

    def act(self, action):
        """Perform an action for a single frame and store the frame."""
        reward = self.ale.act(action)
        index = self.buffer_count % self.frame_buffer.shape[0]
        self.ale.getScreenGrayscale(self.frame_buffer[index, ...])
        self.buffer_count += 1
        return reward

    def get_observation(self):
        """ Resize and merge the previous two screen images."""
        assert self.buffer_count >= 2
        index = self.buffer_count % self.frame_buffer.shape[0] - 1
        image = np.maximum(self.frame_buffer[index, ...],
                           self.frame_buffer[index - 1, ...])
        rescale_size = (self.resize_width, self.resize_height)
        return scipy.misc.imresize(image, size=rescale_size)
