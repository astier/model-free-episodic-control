# Introduction
Implementation of the
**[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)**
algorithm. This is a fork from
[sudeepraja/Model-Free-Episodic-Control](https://github.com/sudeepraja/Model-Free-Episodic-Control)
which is a modification of
[ShibiHe/Model-Free-Episodic-Control](https://github.com/ShibiHe/Model-Free-Episodic-Control).

# Dependencies
* Python 2.7
* Numpy
* SciPy
* Matplotlib
* OpenAI Gym
* A reasonable CPU
* Game Rom (put them in the rom directory)

# Running
Within the terminal execute:

`python run_episodic_control.py`

To get more running details use:

`python run_episodic_control.py -h`.

A results directory will be created where the agents Q<sup>EC</sup>-tables for
each epoch and their results are stored.

**WARNING:** The Q<sup>EC</sup>-tables become very big very quick.
Every Q<sup>EC</sup>-table can be several gigabytes big.
