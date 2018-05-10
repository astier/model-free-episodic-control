# Model-Free Episodic Control

## Description
Implementation of the
**[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)**
algorithm. This is a fork of the repository from
**[sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control)**,
whereas his work is a fork of the original work from
**[ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control)**.

The contributions of this project so far are:
- Heavy Refactoring
- Leaner & cleaner code
- Bugfix

## Requirements
- Python 2
- Numpy
- SciPy
- Matplotlib
- OpenAI Gym
- Game-rom which has to be placed in the rom-directory

## Run
Execute `python main.py`

You can change the hyperparameters directly in the *main.py* file.
A result-directory will be created where the agents Q<sup>EC</sup>-tables for
each epoch and their results are stored.

## Notes
- The agent is trained on the CPU and not GPU
- VAE is not implemented. Only random projection.
- The Q<sup>EC</sup> tables can become very big very quick
(several gigabytes) depending on the action-buffer-size.
