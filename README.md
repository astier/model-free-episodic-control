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
* [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
or [OpenAI gym](https://github.com/openai/gym)
* A reasonable CPU
* Game roms which must be stored in the roms directory.

# Running
Within the terminal execute:

`python run_episodic_control.py`

To get more running details use:

`python run_episodic_control.py -h`.

A results directory will be created where the agents Q<sup>EC</sup>-tables for
each epoch and their results are stored.

**WARNING:** The Q<sup>EC</sup>-tables
become very big very quick (several gigabytes).
