# Model-Free Episodic Control

## Description
Implementation of the **[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)** algorithm. This is a fork of the repository from **[sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control)**, whereas his work is a fork of the original work from **[ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control)**. This project is maintained by **[astier](https://github.com/astier/Model-Free-Episodic-Control)**. Feedback is appreciated.

## Environment & Requirements Installation
All dependencies can be found in the file *requirements* of this project. This includes:
- Python 2
- OpenAI Gym

Before the program can be executed you shall install a separated conda environment with the required dependencies. This can be done by first navigating to the directory where you would like to download this project and then executing the following steps:
```
git clone https://github.com/astier/Model-Free-Episodic-Control.git
cd Model-Free-Episodic-Control
conda create --name mfec --file requirements
source activate mfec
pip install gym
pip install gym[atari]
```

## Put your first MFEC-Agent on the Battlefield
Navigate into to the projects top folder and execute:
```
source activate mfec
python main.py
```
The program should load a pre-trained MFEC-Agent and show a display where you can watch the MFEC-Agent play the atari-game _Q*Bert_. Some information should be printed regularly on the terminal like the average reward he got in a certain episode. After every epoch, the agent's results are stored in a results directory. Also, the agents QEC-table is stored there after every epoch as a file with the extension *.pkl* which latter can be used to load the agent back into memory.

## Train your first MFEC-Agent to be Combat-Ready
To train your own agent from scratch you simply have to change the variable *QEC_TABLE_PATH* which you can find in the *main.py* file from this:
> QEC_TABLE_PATH = 'example_agent_rambo.pkl'

to this:
> QEC_TABLE_PATH = ''

It might also be advisable to turn off the display so the training will be faster. Just change the variable *DISPLAY_SCREEN* from this:
> DISPLAY_SCREEN = True

to this:
> DISPLAY_SCREEN = False

That's it. Now you can track the agent's performance which will be printed regularly in the terminal. Remember the agent is saved to the hard-disk after every epoch and will be called something like *qec_num.pkl*. When you cancel the training before its finished you can continue later on by loading your trained agent back into memory. Just change the variable *QEC_TABLE_PATH* from this:
> QEC_TABLE_PATH = ''

to this:
> QEC_TABLE_PATH = 'path_to_your_killer_agent_aka_terminator'

Do I have to tell you how to turn the display on so you can see your own MFEC-Agent live in action?

## Training- and Hyperparameters
Every important aspect of the program like:
- The game which should be played (default = _Q*Bert_)
- Loading of pre-trained agents
- Turning ON/OFF rendering
- Hyperparameters
- Training duration
- etc.

can be configured by a few variables which you can find and change in the file *main.py*. They are located on the top of the file and are written in UPPERCASE. The default settings are those of the paper.

## Notes
- VAE is not implemented. Only random projection.
- Currently, the algorithm might still do not work exactly like in the paper (I'm working on it tho').
- The agent is trained on a CPU and not GPU.
