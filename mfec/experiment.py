#! /usr/bin/env python2

import logging
import scipy.misc


# TODO refactor to main
class Experiment(object):

    def __init__(self, ale, agent, resize_width, resize_height, epochs,
                 steps_per_epoch, frame_skip):
        self.ale = ale
        self.agent = agent
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.frame_skip = frame_skip
        self.actions = ale.getMinimalActionSet()
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.width, self.height = ale.getScreenDims()

    def run(self):
        for epoch in range(1, self.epochs + 1):
            # TODO steps vs episodes
            steps_left = self.steps_per_epoch
            while steps_left > 0:
                logging.info("Epoch: {}\tSteps: {}".format(epoch, steps_left))
                steps_left -= self.run_episode(steps_left)
            self.agent.finish_epoch(epoch)  # TODO refactor to experiment

    def run_episode(self, max_steps):
        reward = 0
        action = self.agent.start_episode(self.projection())
        steps = 0

        while not self.ale.game_over() and steps < max_steps:
            reward = sum(self.ale.act(self.actions[action]) for _ in
                         range(self.frame_skip))
            action = self.agent.step(reward, self.projection())
            steps += 1

        self.ale.reset_game()
        self.agent.end_episode(reward)
        return steps

    # TODO projection should happen here!
    def projection(self):
        screen = self.ale.getScreenGrayscale()[:, :, 0]
        size = (self.resize_width, self.resize_height)
        return scipy.misc.imresize(screen, size=size)
