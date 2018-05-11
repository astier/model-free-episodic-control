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
        self.observation = None
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

    def start_episode(self, observation):
        """This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """
        self.step_counter = 0
        self.episode_reward = 0
        self.traces = []
        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)
        self.action = return_action
        self.observation = observation
        return return_action

    def _choose_action(self, qec_table, epsilon, observation,
                       reward):
        self.add_trace(self.observation, self.action, reward, False)

        # Epsilon greedy approach chooses random action for exploration
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)

        # argmax(Q(s,a)) for exploitation
        best_value = float('-inf')
        best_action = None
        for action in range(self.num_actions):
            value = qec_table.estimate(observation, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def step(self, reward, observation):
        """This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.
        """
        self.step_counter += 1
        self.episode_reward += reward
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
        action = self._choose_action(self.qec,
                                     self.epsilon, observation,
                                     np.clip(reward, -1, 1))
        self.action = action
        self.observation = observation
        return action

    def end_episode(self, reward, terminal=True):
        """This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        """
        self.episode_reward += reward
        self.total_reward += self.episode_reward
        self.total_episodes += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        # Store the latest sample.
        self.add_trace(self.observation, self.action,
                       np.clip(reward, -1, 1), terminal)
        # Do update
        q_return = 0.
        for i in range(len(self.traces) - 1, -1, -1):
            trace = self.traces[i]
            q_return = q_return * self.discount + trace['reward']
            self.qec.update(trace['observation'], trace['action'],
                            q_return)

        # calculate time
        rho = 0.98
        self.steps_sec_ema *= rho
        self.steps_sec_ema += (1. - rho) * (self.step_counter / total_time)
        logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
            self.step_counter / total_time, self.steps_sec_ema))
        logging.info('episode {} reward: {:.2f}'.format(self.total_episodes,
                                                        self.episode_reward))

    def add_trace(self, observation, action, reward, terminal=True):
        self.traces.append(
            {'observation': observation, 'action': action, 'reward': reward,
             'terminal': terminal})

    def finish_epoch(self, epoch):
        qec_file_prefix = self.result_dir + '/qec_table_file_'

        # Save qec-table
        qec_file = open(qec_file_prefix + str(epoch) + '.pkl', 'w')
        cPickle.dump(self.qec, qec_file, 2)
        qec_file.close()

        # Remove old qec-table to save storage space
        qec_file_old = qec_file_prefix + str(epoch - 1) + '.pkl'
        if os.path.isfile(qec_file_old):
            os.remove(qec_file_old)

        self.update_result_file(epoch, self.total_episodes, self.total_reward)
        self.total_episodes = 0
        self.total_reward = 0

    def update_result_file(self, epoch, total_episodes, total_reward):
        result = "{},{},{},{}\n".format(epoch, total_episodes, total_reward,
                                        total_reward / total_episodes)
        self.result_file.write(result)
        self.result_file.flush()
