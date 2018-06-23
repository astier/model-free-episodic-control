#!/usr/bin/env python3

import os.path


# TODO library?
class Utils:

    def __init__(self, results_dir, frames_per_epoch, max_frames,
                 store_agent, agent):
        self.results_dir = results_dir
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

        message = 'Epoch: {}\tEpisode: {}\tReward: {}\tEpoch-Frames: {}/{}'
        results = [self.epoch, self.epoch_episodes, episode_reward,
                   self.epoch_frames, self.frames_per_epoch]
        print(message.format(*results))

    def end_epoch(self):
        """Save the results for the given epoch in the results-file"""
        results = [self.epoch, self.epoch_episodes, self.epoch_frames,
                   self.epoch_reward_sum,
                   round(self.epoch_reward_sum / self.epoch_episodes),
                   self.epoch_reward_max]

        self.results_file.write('{},{},{},{},{},{}\n'.format(*results))
        self.results_file.flush()

        message = '\nEpoch: {}\tEpisodes: {}\tFrames: {}\tReward-Sum: {}\t' \
                  'Reward-Avg: {}\tReward-Max: {}\tTotal-Frames: {}/{}\n'
        results = results + [self.total_frames, self.max_frames]
        print(message.format(*results))

        self.epoch += 1
        self.epoch_episodes = 0
        self.epoch_frames = 0
        self.epoch_reward_sum = 0
        self.epoch_reward_max = 0

    def close(self):
        self.results_file.close()
