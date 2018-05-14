# Model-Free Episodic Control

## Description
Implementation of the
**[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)**
algorithm. This is a fork of the repository from
**[sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control)**,
whereas his work is a fork of the original work from
**[ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control)**.

Current contributions:
- Bugfixes
- Optimizations
- Make it actually work like in the paper (not 100% accomplished)
- Heavy refactoring and cleaning of the codebase to make it:
    + Maintainable
    + Readable
    + Extensible

## Requirements
- Python 2
- Numpy
- SciPy
- scikit-learn
- Arcade Learning Environment
- Game-rom which has to be placed in the rom-directory

## Run
Execute:
```
python main.py
```
You can change the hyperparameters directly in the *main.py* file.
A result-directory will be created where the agent's results will be stored.

The trained agent can be saved to the hard-drive by setting:
```
SAVE_QEC_TABLE = True
```
The stored Q<sup>EC</sup>-table can then be used to load the agent back into
memory.
**Warning:** The Q<sup>EC</sup>-tables can become several gigabytes large
depending on the action-buffer-size and the training-time.

## Notes
- VAE is not implemented. Only random projection.
- Currently, the algorithm might still do not work exactly like in the paper.
- The agent is trained on the CPU and not GPU.
