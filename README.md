# Introduction
Implementation of the
**[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)**
paper. This is a fork from
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
* Game roms should be stored in directory *roms* which stays next to this
folder.

    Parent Folder

    ├ Model-Free-Episodic-Control -> source codes + README.md

    └ roms -> game roms

# Running
Within the terminal execute

`python run_episodic_control.py`

To get more running details use

`python run_episodic_control.py -h`.
