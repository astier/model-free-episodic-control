#!/usr/bin/env python2

import cPickle  # TODO json?
import logging
import os


# TODO improve output and stats
class Utils(object):

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.results_file = self._create_results_file()
        self.epoch_reward = 0
        self.epoch_episodes = 0

    def _create_results_file(self):
        results_file_name = os.path.join(self.results_dir, 'results.csv')
        results_file = open(results_file_name, 'w')
        results_file.write('epoch, episodes, reward_sum, reward_avg\n')
        return results_file

    def update_results(self, episode_reward):
        """Should be always and only executed at the end of an episode."""
        self.epoch_reward += episode_reward
        self.epoch_episodes += 1
        logging.info('Episode {} Reward: {:.2f} \n'.format(
            self.epoch_episodes, episode_reward))

    # TODO make output prettier
    def save_results(self, epoch):
        """Save the results for the given epoch in the results-file"""
        results = [epoch, self.epoch_episodes, self.epoch_reward,
                   self.epoch_reward / self.epoch_episodes]
        message = 'Epoch: {}\tEpisodes: {}\tTotal-Reward: {}\tAvg-Reward: {}\n'
        logging.info(message.format(*results))

        self.results_file.write('{},{},{},{}\n'.format(*results))
        self.results_file.flush()
        self.epoch_episodes = 0
        self.epoch_reward = 0

    def save_agent(self, epoch, agent):
        """Save the agents QEC-table in a file."""
        qec_prefix = os.path.join(self.results_dir, 'qec_')

        # Save qec-table
        qec = open(qec_prefix + str(epoch) + '.pkl', 'w')
        cPickle.dump(agent.qec, qec, 2)
        qec.close()

        # Remove old qec-table to save storage space
        qec_old = qec_prefix + str(epoch - 1) + '.pkl'
        if os.path.isfile(qec_old):
            os.remove(qec_old)

    @staticmethod
    def load_agent(qec_path):
        return cPickle.load(open(qec_path, 'r'))
