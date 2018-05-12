#! /usr/bin/env python2

import cPickle  # TODO cpickle vs json
import logging
import os


class Utils(object):

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.stats_file = self._create_stats_file()
        self.epoch_reward = 0
        self.epoch_episodes = 0

    def _create_stats_file(self):
        logging.info("OPENING " + self.results_dir + '/results.csv')
        result_file = open(self.results_dir + '/results.csv', 'w')
        result_file.write('epoch, episodes, reward_sum, reward_avg\n')
        result_file.flush()
        return result_file

    def update_results(self, episode_reward):
        self.epoch_reward += episode_reward
        self.epoch_episodes += 1
        logging.info('episode {} episode_reward: {:.2f}\n'.format(
            self.epoch_episodes, episode_reward))

    def save_results(self, epoch):
        stats = "{},{},{},{}\n".format(epoch, self.epoch_episodes,
                                       self.epoch_reward, self.epoch_reward,
                                       self.epoch_episodes)
        self.stats_file.write(stats)
        self.stats_file.flush()
        self.epoch_episodes = 0
        self.epoch_reward = 0

    def save_agent(self, epoch, agent):
        qec_prefix = self.results_dir + '/qec_'

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
