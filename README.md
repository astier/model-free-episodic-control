# Model-Free Episodic Control

## Description
Implementation of the **[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460)** algorithm. This is a fork of the repository from **[sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control)**, whereas his work is a fork of the original work from **[ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control)**. This project is maintained on the following GitHub repository: [astier/Model-Free-Episodic-Control](https://github.com/astier/Model-Free-Episodic-Control). Feedback is appreciated.

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
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
source activate mfec
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
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

## Training- and Hyperparameters
You can change the hyperparameters directly in the *main* file. They can be found on the top of the file and are written in UPPERCASE. The default settings are those of the paper.

## Notes
- VAE is not implemented. Only random projection.
- Currently, the algorithm might still do not work exactly like in the paper (I'm working on it tho').
- The agent is trained on the CPU and not GPU.

