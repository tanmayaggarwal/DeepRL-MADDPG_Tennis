# Author: Tanmay Aggarwal

# Project Collaboration and Competition

### Overview

In this project, I use the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm (initially described by OpenAI in https://arxiv.org/pdf/1706.02275.pdf) to train two agents to control rackets to bounce a ball over a net within a Unity3D environment. There are six main files in this project:
1. Tennis-Solved.ipynb
2. maddpg.py
3. model.py
4. buffer.py
5. OUNoise.py
6. utilities.py

The Jupyter Notebook 'Tennis-Solved.ipynb' is the main file used to train and run the agent. The following macro parameters are used in the training process:
- n_episodes: 5000
- episode_length: 1000
- print_every: 100
- random_seed: 48

The 'maddpg.py' file is used to define the MADDPG Agent class and the MADDPG Agent Trainer class.
The 'OUNoise.py' file is used to define the Noise process class.
The 'buffer.py' file is used to define the ReplayBuffer class.
The 'utilities.py' file is used to define some helper functions.

The following hyperparameters are used in the training process:

- BUFFER_SIZE = 1e6                 # replay buffer size
- BATCH_SIZE = 256                  # minibatch size
- GAMMA = 0.95                       # discount factor
- TAU = 1e-2                        # for soft update of target parameters
- LR_ACTOR = 1e-3                   # learning rate of the actor
- LR_CRITIC = 1e-3                  # learning rate of the critic
- UPDATE_EVERY = 1               # Frequency of learning

For the OUNoise process, the following parameters were used:
- mu = 0.
- Theta = 0.15
- Sigma = 0.1

A replay buffer is used to sample experiences randomly to update the neural network parameters.

The 'model.py' file is where the Actor and the Critic neural network models are defined. These neural networks are being used as function approximators that guide the agent's actions at each time step (i.e., the policy). The Actor model directly maps each state with actions. Meanwhile, the Critic network determines the Q-value for each (state, action) pair. The next-state Q values are calculated with the target value (critic) network and the target policy (actor) network. Meanwhile, the original Q value is calculated with the local value (critic) network, not the target value (critic) network.

The overall architecture of the model is relatively simple with two fully connected hidden layers with 64 and 64 nodes each, respectively. ReLU activation function is used in between each layer in the network. An Adam optimizer is used to minimize the loss function. I have used the Mean Squared Error loss function in this model.

The target networks are updated via "soft updates" controlled by the hyperparameter Tau.

Finally, I have used the Ornstein-Uhlenbeck Process to add noise to the action output in order to encourage exploration in the continuous action space.

## Description of the MADDPG algorithm

As described above, in this project, I have used the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train two agents to control rackets to bounce a ball over a net.

The motivation behind using this algorithm stems from the characteristics of the Unity Tennis environment. Specifically, this environment requires the agents to learn from a high dimensional state space and perform actions in continuous action space.

Value function algorithms such as Deep Q-Learning (DQN) only work in discrete and low-dimensional action spaces. Meanwhile, policy gradient algorithms such as REINFORCE tend to be sample inefficient and can often converge to local optima.

MADDPG is a deep reinforcement learning algorithm which makes use of Actor-Critic networks, similar to Deep Deterministic Policy Gradient (DDPG) algorithms. Both DDPG and MADDPG concurrently learns:
1. The policy that guides the agent's action at each time step (i.e., the policy function using the Actor neural network), and
2. The Q-function which estimates the expected reward associated with taking a particular action in a given time step (i.e., the action value function using the Critic neural network).

The key difference between DDPG and MADDPG is that MADDPG incorporates the actions taken by all agents during training (instead of training each agent to learn from its own action only). This is accomplished by letting the Critic network of each agent see the actions taken by all the agents and use it to guide the Actor networks to come up with a better policy.

In MADDPG, the Actor network takes the environment state as input and returns the action. Meanwhile, the Critic network takes states and actions of all agents as input and returns the Q-value.

## Discussion of the hyperparameters

Training this model was a highly iterative process requiring several refinements to the hyperparameters. Specifically, the model required a delicate balance between exploration (controlled via the OUNoise process) and exploitation (controlled by the learning rates).

Discussion for some of the key hyperparameters is as follows:
- Random seed: To ensure scores could be compared across test runs, a fixed seed was used. The model was found to be relatively sensitive to the seed value and several various seeds were tested before finalizing a seed value of 48.
- Batch size: the goal here was to find a balance between a batch size which was large enough to meaningfully estimate a gradient while yet being small enough to not overburden the training process. The final batch size of 256 was found to be effective in solving the environment.
- Gamma: various values between 0.9-0.99 were tested. Eventually, 0.95 gave the best result ensuring long term rewards were being discounted but within reason.
- Tau: given Tau controls the frequency of soft updates between the local and target networks, initially smaller values were evaluated to ensure a slow learning process for the target networks (and avoid significantly large learning steps at any given sample set). However, through iterations, it was found that the model learned more efficiently with a relatively high value of Tau (final value was set at 1e-2).
- Learning rates (actor and critic): various values were explored ranging from 1e-3 to 4e-4. Eventually, the model trained most effectively with the final learning rate of 1e-3. No significant improvement was noted when using a slightly different learning rate between the actor and critic.
- UPDATE_EVERY: unlike the original paper on MADDPG, it was found that this model learns better when the update frequency is 1.
- OUNoise variables (particularly, sigma): The model performance significantly improved when sigma was reduced so as to avoid a large standard deviation in the OUNoise process (that guided the exploration of the agent). Relatedly, a normal distribution was found to be more effective in the OUNoise process compared to a uniform distribution.

### Plot of rewards

The environment is solved (i.e., the average score of 0.5 is reached) in 762 episodes.

![Rewards Image](/Plot_of_Rewards.png)

### Future improvements

The following future improvements should be considered to further improve effectiveness of the agents (i.e., get a higher average score and / or lower training time):
1. Using batch normalization and dropout in the actor and critic neural network to further stabilize learning.
2. Using a larger batch size / higher learning rate to improve the learning process.
3. Implementing prioritized experience replay to increase sampling efficiency.
4. Experimenting with using parameter space noise as compared to noise on action.
4. Implementing a distributed learning algorithm such as Proximal Policy Optimization (PPO), Asynchronous Advantage Actor-Critic (A3C), or Distributed Distributional Deterministic Policy Gradients (D4PG) to train the agents in parallel.
