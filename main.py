#! /usr/bin/env python2

"""Processes the command line arguments and starts the training."""

import argparse
import cPickle
import logging
import os
import sys

import atari_py.ale_python_interface
import numpy as np

import ec_functions
import ec_agent
import ale_experiment

BASE_ROM_PATH = "./roms/"
ROM = 'qbert.bin'

# Training parameters
STEPS_PER_EPOCH = 300  # 10000
EPOCHS = 2  # 5000
STEPS_PER_TEST = 0

# Hyperparameters
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0
K_NEAREST_NEIGHBOR = 11
EC_DISCOUNT = 1.0
BUFFER_SIZE = 1000000
DIMENSION_OF_STATE = 64
PROJECTION_TYPE = 'random'  # or VAE
EPSILON_START = 1.0
EPSILON_MIN = .005
EPSILON_DECAY = 10000
RESIZE_METHOD = 'scale'
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
DEATH_ENDS_EPISODE = 'true'
MAX_START_NULLOPS = 30
DETERMINISTIC = True


def main(args):
    """Execute a complete training run."""
    logging.basicConfig(level=logging.INFO)

    parameters = process_args(args, __doc__)
    rng = setup_rng(parameters.deterministic, seed=123)
    ale = setup_ale(parameters.display_screen, parameters.rom,
                    parameters.repeat_action_probability, rng)
    agent = setup_agent(ale, parameters, rng)
    experiment = ale_experiment.ALEExperiment(ale, agent,
                                              RESIZED_WIDTH,
                                              RESIZED_HEIGHT,
                                              parameters.resize_method,
                                              parameters.epochs,
                                              parameters.steps_per_epoch,
                                              parameters.steps_per_test,
                                              parameters.frame_skip,
                                              parameters.death_ends_episode,
                                              parameters.max_start_nullops,
                                              rng)
    experiment.run()


def setup_rng(deterministic, seed):
    if deterministic:
        return np.random.RandomState(seed)
    return np.random.RandomState()


def setup_ale(display_screen, rom, repeat_action_probability, rng):
    ale = atari_py.ale_python_interface.ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))
    ale.setBool('display_screen', display_screen)
    ale.loadROM(os.path.join(BASE_ROM_PATH, rom))
    ale.setFloat('repeat_action_probability', repeat_action_probability)
    # Manage display-screen on OSX
    if display_screen and sys.platform == 'darwin':
        import pygame
        pygame.init()
        ale.setBool('sound', False)  # Sound doesn't work on OSX
    return ale


def setup_agent(ale, parameters, rng):
    num_actions = len(ale.getMinimalActionSet())
    qec_table = load_qec_table(num_actions, parameters, rng)
    agent = ec_agent.EpisodicControl(qec_table,
                                     parameters.ec_discount,
                                     num_actions,
                                     parameters.epsilon_start,
                                     parameters.epsilon_min,
                                     parameters.epsilon_decay,
                                     parameters.experiment_prefix,
                                     rng)
    return agent


def load_qec_table(num_actions, parameters, rng):
    if parameters.qec_table is None:
        return ec_functions.QECTable(parameters.knn,
                                     parameters.state_dimension,
                                     parameters.projection_type,
                                     RESIZED_WIDTH *
                                     RESIZED_HEIGHT,
                                     parameters.buffer_size,
                                     num_actions,
                                     rng)
    return cPickle.load(open(parameters.qec_table, 'r'))


def process_args(args, description):
    """Processes the command line arguments.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rom', dest="rom", default=ROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=EPOCHS,
                        help='Number of training epochs (default: %('
                             'default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %('
                             'default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('--display-screen', dest="display_screen",
                        action='store_true', default=False,
                        help='Show the game screen.')
    parser.add_argument('--experiment-prefix', dest="experiment_prefix",
                        default=None,
                        help='Experiment name prefix '
                             '(default is the name of the game)')
    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                             '(default: %(default)s)')
    parser.add_argument('--repeat-action-probability',
                        dest="repeat_action_probability",
                        default=REPEAT_ACTION_PROBABILITY, type=float,
                        help=('Probability that action choice will be ' +
                              'ignored (default: %(default)s)'))
    parser.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--resize-method', dest="resize_method",
                        type=str, default=RESIZE_METHOD,
                        help='crop|scale (default: %(default)s)')
    parser.add_argument('--death-ends-episode', dest="death_ends_episode",
                        type=str, default=DEATH_ENDS_EPISODE,
                        help='true|false (default: %(default)s)')
    parser.add_argument('--max-start-nullops', dest="max_start_nullops",
                        type=int, default=MAX_START_NULLOPS,
                        help=('Maximum number of null-ops at the start ' +
                              'of games. (default: %(default)s)'))
    parser.add_argument('--deterministic', dest="deterministic",
                        type=bool, default=DETERMINISTIC,
                        help=('Whether to use deterministic parameters ' +
                              'for learning. (default: %(default)s)'))
    parser.add_argument('--knn', dest='knn',
                        type=int, default=K_NEAREST_NEIGHBOR,
                        help='k nearest neighbor')
    parser.add_argument('--ec-discount', dest='ec_discount',
                        type=float, default=EC_DISCOUNT,
                        help='episodic control discount')
    parser.add_argument('--buffer-size', dest='buffer_size',
                        type=int, default=BUFFER_SIZE,
                        help='buffer size for each action in episodic control')
    parser.add_argument('--state-dimension', dest='state_dimension',
                        type=int, default=DIMENSION_OF_STATE,
                        help='the dimension of the stored state')
    parser.add_argument('--projection-type', dest='projection_type',
                        type=str, default=PROJECTION_TYPE,
                        help='the type of the ec projection')
    parser.add_argument('--qec-table', dest='qec_table',
                        type=str, default=None,
                        help='Qec table file for episodic control')

    parameters = parser.parse_args(args)
    if parameters.experiment_prefix is None:
        parameters.experiment_prefix = "results/" + os.path.splitext(
            os.path.basename(parameters.rom))[0]

    if parameters.death_ends_episode == 'true':
        parameters.death_ends_episode = True
    elif parameters.death_ends_episode == 'false':
        parameters.death_ends_episode = False
    else:
        raise ValueError("--death-ends-episode must be true or false")

    return parameters


if __name__ == "__main__":
    main(sys.argv[1:])
