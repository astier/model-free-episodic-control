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
If desired the agents Q<sup>EC</sup> table can also be stored by setting:
```
SAVE_QEC_TABLE = True
```

**Warning:** The Q<sup>EC</sup>-tables can become very big very quick
(several gigabytes) depending on the action-buffer-size.

## Notes
- Currently, the algorithm might still do not work exactly like in the paper.
- The agent is trained on the CPU and not GPU.
- VAE is not implemented. Only random projection.
