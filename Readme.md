# Author: Tanmay Aggarwal

# Project Collaboration and Competition

### Introduction

In this project, I have trained two agents to play tennis in a Unity environment using deep reinforcement learning.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

Specifically,
- After each episode, the rewards that each agent received are added (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the `p3_collab-compet/` folder, and unzip (or decompress) the file.

### Installation dependencies

The following dependencies are needed to run this project:
    - Unity environment
    - Anaconda environment
    - Pytorch
    - Unity ML agents

You can download the required Unity environment from one of the links above that best match your operating system.

The following commands can be executed in the terminal for the remaining dependencies:
    - conda create --name Tennis python=3.6
    - source activate Tennis
    - conda install -y pytorch -c pytorch
    - pip install unityagents

### Instructions

Follow the instructions in `Tennis-Solved.ipynb` to get started with training your own agent!  
