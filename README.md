# Model-Free Episodic Control

## Description

Implementation of the [Model-Free Episodic Control](http://arxiv.org/abs/1606.04460) algorithm (except the VAE part). This project is maintained by [astier](https://github.com/astier) and is a fork of the repository from [sudeepraja](https://github.com/sudeepraja/Model-Free-Episodic-Control), whereas his work is a fork of the original work from [ShibiHe](https://github.com/ShibiHe/Model-Free-Episodic-Control). All deviations from the paper which I am aware of are listed below.

## Dependencies

This project is written in *Python 3* and uses *[OpenAI Gym](https://github.com/openai/gym)* for the environments. However, the agent itself is independent of *OpenAI Gym* and can be used with any framework.

I would recommend creating a separate conda environment to install the dependencies. This can be done by first navigating to the directory where you would like to download this project and then executing the following steps:

```sh
git clone https://github.com/astier/Model-Free-Episodic-Control
cd Model-Free-Episodic-Control
conda create -n mfec scikit-learn
conda activate mfec
pip install gym[atari]
```

## Parameters

Every important aspect and parameter of the program can be configured by a few variables which can be found on top of the file *main.py* and which are written in UPPERCASE. The default settings are those of the paper.

## Put your first Agent on the Battlefield

Navigate into to the projects top folder and execute:

```sh
source activate mfec
python main.py
```

The program should load a pre-trained agent and show a display where you can watch him play the atari game _Q*Bert_. Some information should be printed regularly on the terminal like the average reward he got in a certain epoch. After every epoch, the agents' results are stored in the directory named *agents*. Also, the agent is stored after every epoch as a file with the extension *.pkl* which latter can be used to load him back into memory.

## Train your first Agent to be Combat-Ready

To train your own agent from scratch you simply have to set the variable *AGENT_PATH* to an empty string:
> AGENT_PATH = ""

It is also advisable to turn off the rendering so the training will be faster. Just set the variable *RENDER* to false:
> RENDER = False

Now you can track the agent's performance which will be printed regularly on the terminal. When you cancel the training before its finished you can continue later on by loading your trained agent back into memory. Just change the variable *AGENT_PATH* to the path where you stored your agent:
> AGENT_PATH = "path_to_your_agent/agent.pkl"

## Deviations

All deviations from the paper which I am aware of are listed here.

- VAE is NOT implemented. Only random projection.
- The paper mentions that the agent starts at one of 30 possible initial states. However, in this implementation, the agent starts in a state whichever is generated by the environment at the beginning of an episode.
- The paper does not specify which KNN-Algorithm is used. This project implements KNN as a KDTree. The search-tree is rebuilt frequently.
- When an action has to be chosen and multiple actions with the same maximum value exist then an action is chosen randomly from this set of actions. The paper does not describe what the algorithm does in such a case.
- The paper does not describe what happens when an estimation via KNN has to be performed when the size of the respective action-buffer is smaller than *k*. This implementation returns in such a case *float(inf)*. This ensures that this action gets executed and the action-buffer gets filled with k elements as fast as possible.
