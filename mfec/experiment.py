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

        start_lives = self.ale.lives()
        game_over = False
        reward = 0
        action = self.agent.start_episode(self.get_state())
        steps = 0

        # TODO stick with game_over and terminal
        while not game_over and steps < max_steps:
            reward = sum(self.ale.act(self.actions[action]) for _ in
                         range(self.frame_skip))
            self.death = (self.death_ends_episode and
                          self.ale.lives() < start_lives)
            game_over = self.ale.game_over() or self.death
            action = self.agent.step(reward, self.get_state())
            steps += 1

        self.agent.end_episode(reward, game_over)
        return steps

    # TODO projection should happen here!
    def get_state(self):
        rescale_size = (self.resize_width, self.resize_height)
        frame = self.ale.getScreenGrayscale()[:, :, 0]
        return scipy.misc.imresize(frame, size=rescale_size)
