#! /usr/bin/env python2

"""This script handles reading command line arguments and starting the
training process.  It shouldn't be executed directly; it is used by the
running script.
"""

import argparse
import cPickle
import logging
import os

import atari_py.ale_python_interface
import numpy as np

import EC_agent
import EC_functions
import ale_experiment


def process_args(args, defaults, description):
    """Implements the command line interface.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rom', dest="rom", default=defaults.ROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %('
                             'default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %('
                             'default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('--display-screen', dest="display_screen",
                        action='store_true', default=False,
                        help='Show the game screen.')
    parser.add_argument('--experiment-prefix', dest="experiment_prefix",
                        default=None,
                        help='Experiment name prefix '
                             '(default is the name of the game)')
    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                             '(default: %(default)s)')
    parser.add_argument('--repeat-action-probability',
                        dest="repeat_action_probability",
                        default=defaults.REPEAT_ACTION_PROBABILITY, type=float,
                        help=('Probability that action choice will be ' +
                              'ignored (default: %(default)s)'))
    parser.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=defaults.EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=defaults.EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--resize-method', dest="resize_method",
                        type=str, default=defaults.RESIZE_METHOD,
                        help='crop|scale (default: %(default)s)')
    parser.add_argument('--death-ends-episode', dest="death_ends_episode",
                        type=str, default=defaults.DEATH_ENDS_EPISODE,
                        help='true|false (default: %(default)s)')
    parser.add_argument('--max-start-nullops', dest="max_start_nullops",
                        type=int, default=defaults.MAX_START_NULLOPS,
                        help=('Maximum number of null-ops at the start ' +
                              'of games. (default: %(default)s)'))
    parser.add_argument('--deterministic', dest="deterministic",
                        type=bool, default=defaults.DETERMINISTIC,
                        help=('Whether to use deterministic parameters ' +
                              'for learning. (default: %(default)s)'))
    parser.add_argument('--knn', dest='knn',
                        type=int, default=defaults.K_NEAREST_NEIGHBOR,
                        help='k nearest neighbor')
    parser.add_argument('--ec-discount', dest='ec_discount',
                        type=float, default=defaults.EC_DISCOUNT,
                        help='episodic control discount')
    parser.add_argument('--buffer-size', dest='buffer_size',
                        type=int, default=defaults.BUFFER_SIZE,
                        help='buffer size for each action in episodic control')
    parser.add_argument('--state-dimension', dest='state_dimension',
                        type=int, default=defaults.DIMENSION_OF_STATE,
                        help='the dimension of the stored state')
    parser.add_argument('--projection-type', dest='projection_type',
                        type=str, default=defaults.PROJECTION_TYPE,
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


def launch(args, defaults, description):
    """Execute a complete training run."""

    logging.basicConfig(level=logging.INFO)
    parameters = process_args(args, defaults, description)

    if parameters.rom.endswith('.bin'):
        rom = parameters.rom
    else:
        rom = "%s.bin" % parameters.rom
    full_rom_path = os.path.join(defaults.BASE_ROM_PATH, rom)

    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()

    ale = atari_py.ale_python_interface.ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))

    if parameters.display_screen:
        import sys
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False)  # Sound doesn't work on OSX

    ale.setBool('display_screen', parameters.display_screen)
    ale.setFloat('repeat_action_probability',
                 parameters.repeat_action_probability)
    ale.loadROM(full_rom_path)

    num_actions = len(ale.getMinimalActionSet())

    if parameters.qec_table is None:
        qec_table = EC_functions.QECTable(parameters.knn,
                                          parameters.state_dimension,
                                          parameters.projection_type,
                                          defaults.RESIZED_WIDTH *
                                          defaults.RESIZED_HEIGHT,
                                          parameters.buffer_size,
                                          num_actions,
                                          rng)
    else:
        handle = open(parameters.qec_table, 'r')
        qec_table = cPickle.load(handle)

    agent = EC_agent.EpisodicControl(qec_table,
                                     parameters.ec_discount,
                                     num_actions,
                                     parameters.epsilon_start,
                                     parameters.epsilon_min,
                                     parameters.epsilon_decay,
                                     parameters.experiment_prefix,
                                     rng)

    experiment = ale_experiment.ALEExperiment(ale, agent,
                                              defaults.RESIZED_WIDTH,
                                              defaults.RESIZED_HEIGHT,
                                              parameters.resize_method,
                                              parameters.epochs,
                                              parameters.steps_per_epoch,
                                              parameters.steps_per_test,
                                              parameters.frame_skip,
                                              parameters.death_ends_episode,
                                              parameters.max_start_nullops,
                                              rng)
    experiment.run()


if __name__ == '__main__':
    pass
