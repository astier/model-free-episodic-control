#! /usr/bin/env python2

import time
import os
import logging
import numpy as np
import cPickle


# TODO Refactor result-creation
class EpisodicControl(object):

    def __init__(self, qec, discount, num_actions, epsilon,
                 epsilon_min, epsilon_decay, rom, rng):
        self.qec = qec
        self.discount = discount
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.rng = rng
        self.traces = []
        self.epsilon_rate = self.compute_epsilon_rate(epsilon_decay)
        self.result_dir = self.create_result_dir(rom)
        self.result_file = self.create_result_file()
        self.step_counter = 0
        self.episode_reward = 0
        self.total_reward = 0.  # TODO float vs int
        self.total_episodes = 0
        self.start_time = None
        self.state = None
        self.action = None
        self.steps_sec_ema = 0.

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

    def start_episode(self, state):
        """This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           state - height x width numpy array

        Returns:
           An integer action
        """
        self.step_counter = 0
        self.episode_reward = 0
        self.traces = []
        self.start_time = time.time()
        self.action = self.rng.randint(0, self.num_actions)
        self.state = state
        return self.action

    def step(self, reward, state):
        self.step_counter += 1
        self.episode_reward += reward
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
        action = self.choose_action(self.qec, self.epsilon, state,
                                    np.clip(reward, -1, 1))
        self.action = action
        self.state = state
        return action

    def choose_action(self, qec, epsilon, state, reward):
        self.add_trace(self.state, self.action, reward)

        # Epsilon greedy approach chooses random action for exploration
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)

        # argmax(Q(s,a)) for exploitation
        best_value = float('-inf')
        best_action = None
        for action in range(self.num_actions):
            value = qec.estimate(state, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def end_episode(self, reward):
        self.episode_reward += reward
        self.total_reward += self.episode_reward
        self.total_episodes += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        # Store the latest sample.
        self.add_trace(self.state, self.action, np.clip(reward, -1, 1))
        # Do update
        q_return = 0.
        for i in range(len(self.traces) - 1, -1, -1):
            trace = self.traces[i]
            q_return = q_return * self.discount + trace['reward']
            self.qec.update(trace['state'], trace['action'],
                            q_return)

        # calculate time
        rho = 0.98
        self.steps_sec_ema *= rho
        self.steps_sec_ema += (1. - rho) * (self.step_counter / total_time)
        logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
            self.step_counter / total_time, self.steps_sec_ema))
        logging.info('episode {} reward: {:.2f}'.format(self.total_episodes,
                                                        self.episode_reward))

    def add_trace(self, observation, action, reward):
        self.traces.append(
            {'state': observation, 'action': action, 'reward': reward})

    def finish_epoch(self, epoch):
        qec_prefix = self.result_dir + '/qec_'

        # Save qec-table
        qec = open(qec_prefix + str(epoch) + '.pkl', 'w')
        cPickle.dump(self.qec, qec, 2)
        qec.close()

        # Remove old qec-table to save storage space
        qec_old = qec_prefix + str(epoch - 1) + '.pkl'
        if os.path.isfile(qec_old):
            os.remove(qec_old)

        self.update_result_file(epoch, self.total_episodes, self.total_reward)
        self.total_episodes = 0
        self.total_reward = 0

    def update_result_file(self, epoch, total_episodes, total_reward):
        result = "{},{},{},{}\n".format(epoch, total_episodes, total_reward,
                                        total_reward / total_episodes)
        self.result_file.write(result)
        self.result_file.flush()
