# Model-Free Episodic Control

## Description
Implementation of the
**[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)**
algorithm. This is a fork of the repository from
**[sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control)**,
whereas his work is a fork of the original work from
**[ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control)**.

The projects goal is a high quality implementation of the algorithm.
This includes:
- Bugfixes
- Refactoring and cleaning of the codebase and project to make it more
maintainable, readable and extensible.
- Optimizations
- Active maintenance
- Modularity for easy altering of the algorithms behavior
- Maybe implementations of related algorithms like the
[Neural Episodic Control](https://arxiv.org/pdf/1703.01988.pdf)

## Requirements
- Python 2
- Numpy
- SciPy
- scikit-learn
- [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment).
It maybe be easier to install via [OpenAI Gym](https://github.com/openai/gym):

      `pip install gym`
      `pip install -e '.[atari]'`
      
- Game-rom which has to be placed in the rom-directory

## Run
Execute `python main.py`

You can change the hyperparameters directly in the *main.py* file.

A result-directory will be created where the agents results with some
statistics are stored. If desired the agents Q<sup>EC</sup> table can also be
stored by setting:
> SAVE_QEC_TABLE = True

**Warning:** The Q<sup>EC</sup>-tables can become very big very quick
(several gigabytes) depending on the action-buffer-size.

## Notes
- Currently the algorithm might does not work exactly like in the paper.
- The agent is trained on the CPU and not GPU.
- VAE is not implemented. Only random projection.
