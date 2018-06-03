#!/usr/bin/env python2

import cPickle  # TODO json?
import os
import time


# TODO improve output and stats? library?
# TODO store detailed-output
# TODO round reward-avg
class Utils(object):

    def __init__(self, rom_file_name, frames_per_epoch, max_frames,
                 store_agent, agent):
        self.results_dir = self._create_results_dir(rom_file_name)
        self.results_file = self._create_results_file()

        self.agent = agent
        self.store_agent = store_agent

        self.frames_per_epoch = frames_per_epoch
        self.max_frames = max_frames
        self.total_frames = 0

        self.epoch = 1
        self.epoch_episodes = 0
        self.epoch_frames = 0
        self.epoch_reward_sum = 0
        self.epoch_reward_max = 0

    @staticmethod
    def _create_results_dir(rom_file_name):
        execution_time = time.strftime("_%m-%d-%H-%M-%S", time.gmtime())
        result_dir = rom_file_name.split('.')[0] + execution_time
        results_dir = os.path.join('results', result_dir)
        os.makedirs(results_dir)
        return results_dir

    def _create_results_file(self):
        results_file_name = os.path.join(self.results_dir, 'results.csv')
        results_file = open(results_file_name, 'w')
        results_file.write("epoch, episodes, frames, reward_sum, "
                           "reward_avg, reward_max\n")
        return results_file

    def end_episode(self, episode_frames, episode_reward):
        """Should be always and only executed at the end of an episode."""
        self.epoch_episodes += 1
        self.epoch_frames += episode_frames
        self.epoch_reward_sum += episode_reward
        if episode_reward > self.epoch_reward_max:
            self.epoch_reward_max = episode_reward
        self.total_frames += episode_frames

        results = [self.epoch, self.epoch_episodes, episode_reward,
                   self.epoch_frames, self.frames_per_epoch]
        message = 'Epoch: {}\tEpisode: {}\tReward: {}\tEpoch-Frames: {}/{}'
        print(message.format(*results))

    def end_epoch(self):
        """Save the results for the given epoch in the results-file"""
        results = [self.epoch, self.epoch_episodes, self.epoch_frames,
                   self.epoch_reward_sum,
                   self.epoch_reward_sum / self.epoch_episodes,
                   self.epoch_reward_max]
        self.results_file.write('{},{},{},{},{},{}\n'.format(*results))
        self.results_file.flush()

        message = '\nEpoch: {}\tEpisodes: {}\tFrames: {}\tReward-Sum: {}\t' \
                  'Reward-Avg: {}\tReward-Max: {}\tTotal-Frames: {}/{}\n'
        results = results + [self.total_frames, self.max_frames]
        print(message.format(*results))

        if self.store_agent:
            self.save_agent()

        self.epoch += 1
        self.epoch_episodes = 0
        self.epoch_frames = 0
        self.epoch_reward_sum = 0
        self.epoch_reward_max = 0

    # TODO keep the best and the newest agent
    def save_agent(self):
        """Save the agents QEC-table in a file."""
        qec_prefix = os.path.join(self.results_dir, 'qec_')

        # Save qec-table
        qec = open(qec_prefix + str(self.epoch) + '.pkl', 'w')
        cPickle.dump(self.agent.qec, qec, 2)
        qec.close()

        # Remove old qec-table to save storage space
        qec_old = qec_prefix + str(self.epoch - 1) + '.pkl'
        if os.path.isfile(qec_old):
            os.remove(qec_old)
