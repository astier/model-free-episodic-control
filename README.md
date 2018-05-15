# Model-Free Episodic Control

## Description
Implementation of the **[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)** algorithm. This is a fork of the repository from **[sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control)**, whereas his work is a fork of the original work from **[ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control)**. This project is maintained by **[astier](https://github.com/astier/Model-Free-Episodic-Control)**. Feedback is appreciated.

## Current Contributions
- Heavy refactoring and cleaning of the codebase to make it maintainable, readable and extensible
- Bugfixes
- Optimizations
- Make it work more like in the paper (not 100% accomplished but pretty much)

## Environment & Requirements Installation
The specific dependencies can be found in the file *requirements* of this project. The [Arcade-Learning-Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/v0.6.0) is used as the Atari2600 environment framework and has to be installed separately.

Before the program can be executed you shall install a separated conda environment with the required dependencies. This can be done by first navigating to the directory where you would like to download this project and then executing the following steps:
```
git clone https://github.com/astier/Model-Free-Episodic-Control.git
cd Model-Free-Episodic-Control
conda create --name mfec --file requirements
source activate mfec
pip install pygame
```
ALE has to be compiled and installed from source otherwise it is not possible to display the agent in action. This can be done by first navigating to the directory where you would like to download ALE and then executing the following steps:
```
source activate mfec
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON
make -j 4
pip install .
```

## Put your first MFEC-Agent on the Battlefield
Navigate into to the projects top folder and execute:
```
source activate mfec
python main.py
```
The program should load a pretrained MFEC-Agent and show a display where you can watch the MFEC-Agent play the atari-game _Q*Bert_ (in the first level he's bad but gets better in the next two). Some information should be printed regularly on the terminal like the average reward he got in a certain episode. After every epoch, the agent's results are stored in a results directory. Also, the agents QEC-table is stored there after every epoch as a file with the extension *.pkl* which latter can be used to load the agent back into memory.

## Train your first MFEC-Agent to be Combat-Ready
To train your own agent from scratch you simply have to change the variable *QEC_TABLE_PATH* which you can find in the *main.py* file from this:
> QEC_TABLE_PATH = 'example_agent.pkl'

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
- Loading of pretrained agents
- Turning ON/OFF the display/sound
- Hyperparameters
- Training duration
- etc.

can be configured by a few variables which you can find and change in the file *main.py*. They are located on the top of the file and are written in UPPERCASE. The default settings are those of the paper.

## Game Roms
The game rom which you want to play has to be placed in the *roms*-directory. Some examples are already there. To actually play the specific game you have to set the *ROM_FILE_NAME* variable to:
> ROM_FILE_NAME = 'rom_file_name.bin'

You can find all kinds of roms for example here:
- [LOVEROMS](https://www.loveroms.com/roms/atari-2600)
- [AtariAge](https://atariage.com/system_items.html?SystemID=2600&ItemTypeID=ROM)

## Notes
- VAE is not implemented. Only random projection.
- Currently, the algorithm might still do not work exactly like in the paper (I'm working on it tho').
- The agent is trained on a CPU and not GPU.

