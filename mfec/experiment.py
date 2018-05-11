#! /usr/bin/env python2

import logging
import numpy as np
import scipy.misc

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8


class Experiment(object):

    def __init__(self, ale, agent, resize_width, resize_height,
                 resize_method, epochs, steps_per_epoch, frame_skip,
                 death_ends_episode, rng):
        self.ale = ale
        self.agent = agent
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        self.actions = ale.getMinimalActionSet()
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.resize_method = resize_method
        self.width, self.height = ale.getScreenDims()
        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty(
            (self.buffer_length, self.height, self.width), dtype=np.uint8)
        self.death = False  # Last episode ended because agent died
        self.rng = rng

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self.run_epoch(epoch, self.steps_per_epoch)
            self.agent.finish_epoch(epoch)

    def run_epoch(self, epoch, steps):
        self.death = False  # Make sure each epoch starts with a reset.
        while steps > 0:
            logging.info("Epoch: {}\tSteps Left: {}".format(epoch, steps))
            steps -= self.run_episode(steps)

    def run_episode(self, max_steps):
        self.init_episode()
        start_lives = self.ale.lives()
        action = self.agent.start_episode(self.get_observation())
        steps = 0
        game_over = False
        reward = None

        while not game_over and steps < max_steps:
            reward = self.step(self.actions[action])
            self.death = (self.death_ends_episode and
                          self.ale.lives() < start_lives)
            game_over = self.ale.game_over() or self.death
            action = self.agent.step(reward, self.get_observation())
            steps += 1

        self.agent.end_episode(reward, game_over)
        return steps

    def init_episode(self):
        """Reset the game and perform null actions to fill the frame-buffer."""
        if not self.death or self.ale.game_over():
            self.ale.reset_game()
        self.act(0)
        self.act(0)

    def act(self, action):
        """Perform an action for a single frame and store the frame."""
        reward = self.ale.act(action)
        index = self.buffer_count % self.buffer_length
        self.ale.getScreenGrayscale(self.screen_buffer[index, ...])
        self.buffer_count += 1
        return reward

    def get_observation(self):
        """ Resize and merge the previous two screen images."""
        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.resize_image(max_image)

    def step(self, action):
        """ Repeat an action for a defined number of frames."""
        reward = 0
        for _ in range(self.frame_skip):
            reward += self.act(action)
        return reward

    def resize_image(self, image):
        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(
                round(float(self.height) * self.resize_width / self.width))
            resized = self.resize(image, (self.resize_width, resize_height))
            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resize_height
            cropped = resized[
                      crop_y_cutoff: crop_y_cutoff + self.resize_height, :]
            return cropped

        elif self.resize_method == 'scale':
            return self.resize(image, (self.resize_width, self.resize_height))
        else:
            raise ValueError('Unrecognized image resize method.')

    @staticmethod
    def resize(image, size):
        return scipy.misc.imresize(image, size=size)
