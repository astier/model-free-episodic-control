# Introduction
Implementation of the
**[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)**
algorithm. This is a fork of the repository from
[sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control),
whereas his work is a fork of the original work from
[ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control).


# Dependencies
Here's what you need to run the program:
- Python 2
- Numpy
- SciPy
- Matplotlib
- OpenAI Gym
- A reasonable CPU
- Rom of your desired game (some example roms are already in the rom directory)


# Running
Within the terminal execute:

`python run_episodic_control.py`

To change hyperparameters change them directly in the *run_episodic_control.py*
file or pass them as options in the command line. To see all command line
options run:

`python run_episodic_control.py -h`.

A result-directory will be created where the agents Q<sup>EC</sup>-tables for
each epoch and their results are stored.

**WARNING:** The Q<sup>EC</sup>-tables become very big very quick.
Every Q<sup>EC</sup>-table can be several gigabytes big.

