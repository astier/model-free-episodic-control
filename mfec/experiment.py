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
        self.resize_width = resize_width
        self.resize_height = resize_height

    def run(self):
        for epoch in range(1, self.epochs + 1):
            # TODO steps vs episodes
            steps_left = self.steps_per_epoch
            while steps_left > 0:
                logging.info("Epoch: {}\tSteps: {}".format(epoch, steps_left))
                steps_left -= self.run_episode(steps_left)
            self.agent.end_epoch(epoch)  # TODO refactor to experiment

    def run_episode(self, max_steps):
        total_reward = 0
        steps = 0
        self.agent.start_episode()

        while not self.ale.game_over() and steps < max_steps:
            observation = self.ale.getScreenGrayscale()[:, :, 0]
            state = self.projection(observation)
            action = self.agent.act(state)
            reward = sum(
                [self.ale.act(action) for _ in range(self.frame_skip)])
            self.agent.add_experience(state, action, reward)
            total_reward += reward
            steps += 1

        self.ale.reset_game()
        self.agent.end_episode(total_reward)
        return steps

    # TODO projection should happen here!
    def projection(self, observation):
        size = (self.resize_width, self.resize_height)
        return scipy.misc.imresize(observation, size=size)
