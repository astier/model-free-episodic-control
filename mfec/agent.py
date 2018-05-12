#! /usr/bin/env python2

import cPickle
import logging
import os
import time

# TODO Refactor result-creation
import numpy as np


class EpisodicControl(object):

    def __init__(self, qec, discount, actions, epsilon, epsilon_min,
                 epsilon_decay, rom, projection, rng):
        self.qec = qec
        self.discount = discount
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_rate = self.compute_epsilon_rate(epsilon_decay)
        self.rng = rng
        self.memory = []
        self.last_state = None
        self.last_action = None
        self.projection = projection

        self.result_dir = self.create_result_dir(rom)
        self.result_file = self.create_result_file()
        self.epoch_reward = 0
        self.epoch_episodes = 0

    def compute_epsilon_rate(self, epsilon_decay):
        if epsilon_decay != 0:
            return (self.epsilon - self.epsilon_min) / epsilon_decay
        return 0

    # TODO define better dir-name
    @staticmethod
    def create_result_dir(rom):
        rom_name = rom.split('.')[0]
        execution_time = time.strftime("_%m-%d-%H-%M-%S", time.gmtime())
        result_dir = 'results/' + rom_name + execution_time
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    def create_result_file(self):
        logging.info("OPENING " + self.result_dir + '/results.csv')
        result_file = open(self.result_dir + '/results.csv', 'w')
        result_file.write('epoch, episodes, reward_sum, reward_avg\n')
        result_file.flush()
        return result_file

    def reset(self):
        self.memory = []

    def act(self, observation):  # TODO initialize first knn buffer?
        self.last_state = np.dot(self.projection, observation.flatten())
        # TODO generator
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
        if self.rng.rand() > self.epsilon:
            self.last_action = self.exploit()
        else:
            self.last_action = self.rng.randint(0, self.actions)
        return self.last_action

    def exploit(self):
        best_value = float('-inf')
        best_action = 0
        for action in range(self.actions):
            value = self.qec.estimate(self.last_state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def receive_reward(self, reward):
        self.memory.append(
            {'state': self.last_state, 'action': self.last_action,
             'reward': reward})

    def train(self, reward):
        # Update Q-Values
        q_return = 0.
        for i in range(len(self.memory) - 1, -1, -1):
            experience = self.memory[i]
            q_return = q_return * self.discount + experience['reward']
            self.qec.update(experience['state'], experience['action'],
                            q_return)

        # Print stats
        self.epoch_reward += reward
        self.epoch_episodes += 1
        logging.info('episode {} reward: {:.2f}\n'.format(self.epoch_episodes,
                                                          reward))

    def save_me(self, epoch):
        qec_prefix = self.result_dir + '/qec_'

        # Save qec-table
        qec = open(qec_prefix + str(epoch) + '.pkl', 'w')
        cPickle.dump(self.qec, qec, 2)
        qec.close()

        # Remove old qec-table to save storage space
        qec_old = qec_prefix + str(epoch - 1) + '.pkl'
        if os.path.isfile(qec_old):
            os.remove(qec_old)

        self.update_results(epoch)

    def update_results(self, epoch):
        result = "{},{},{},{}\n".format(epoch, self.epoch_episodes,
                                        self.epoch_reward, self.epoch_reward /
                                        self.epoch_episodes)
        self.result_file.write(result)
        self.result_file.flush()
        self.epoch_episodes = 0
        self.epoch_reward = 0
