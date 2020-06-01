# main code that contains the neural network setup and the policy / critic updates
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utilities import flatten
from buffer import ReplayBuffer

from model import Actor, Critic
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, state_size, fc1, fc2, action_size, num_agents, agent_index, gamma, tau, lr_actor, lr_critic, random_seed):

        self.state_size = state_size
        self.action_size = action_size
        self.agent_index = agent_index
        self.seed = random.seed(random_seed)
        self.gamma = gamma
        self.tau = tau
        self.iter = 0
        self.noise = OUNoise(self.action_size, random_seed)
        self.learn_step = 0

        # Create Critic network
        self.local_critic = Critic(self.state_size * num_agents, self.action_size * num_agents, random_seed, fc1, fc2).to(device)
        self.target_critic = Critic(self.state_size * num_agents, self.action_size * num_agents, random_seed, fc1, fc2).to(device)
        self.critic_optimizer = Adam(self.local_critic.parameters(), lr=lr_critic)
        # Create Actor network
        self.local_actor = Actor(self.state_size, self.action_size, random_seed, fc1, fc2).to(device)
        self.target_actor = Actor(self.state_size, self.action_size, random_seed, fc1, fc2).to(device)
        self.actor_optimizer = Adam(self.local_actor.parameters(), lr=lr_actor)

    def act(self, state, add_noise=True, noise_weight=1):
            """Get the actions to take under the supplied states
            Parameters:
                state (array_like): Game state provided by the environment
                add_noise (bool): Whether we should apply the noise
                noise_weight (int): How much weight should be applied to the noise
            """
            state = torch.from_numpy(state).float().to(device)
            # Run inference in eval mode
            self.local_actor.eval()
            with torch.no_grad():
                action = self.local_actor(state).cpu().data.numpy()
            self.local_actor.train()
            # add noise if true
            if add_noise:
                action += self.noise.sample() * noise_weight
            return np.clip(action, -1, 1)

    def reset(self):
        """Resets the noise"""
        self.noise.reset()

    def learn(self, agents, experience):
        # update the critics and the actors of all the agents
        num_agents = len(agents)
        states, actions, rewards, next_states, dones = experience

        #--------------- Update critic -------------------------------------#
        # Get predicted next_state actions and Q values from target model
        next_actions = torch.zeros((len(states), num_agents, self.action_size)).to(device)
        for i, agent in enumerate(agents):
            next_actions[:, i] = agent.target_actor(states[:, i, :])

        # flatten state and action
        critic_states = flatten(next_states)
        next_actions = flatten(next_actions)

        # calculate target and expected Q values
        Q_targets_next = self.target_critic(critic_states, next_actions)
        Q_targets = rewards[:, self.agent_index, :] + (self.gamma * Q_targets_next * (1 - dones[:, self.agent_index, :]))
        Q_expected = self.local_critic(flatten(states), flatten(actions))

        # compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # perform gradient clipping
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1)
        self.critic_optimizer.step()

        c1 = critic_loss.item()

        #-------------- Update actor --------------------#

        # Compute actor loss
        predicted_actions = torch.zeros((len(states), num_agents, self.action_size)).to(device)
        predicted_actions.data.copy_(actions.data)
        predicted_actions[:, self.agent_index] = self.local_actor(states[:, self.agent_index])
        actor_loss = -self.local_critic(flatten(states), flatten(predicted_actions)).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # perform gradient clipping
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), 1)
        self.actor_optimizer.step()

        a1 = actor_loss.item()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step == 0:
            # Start local and target with same parameters
            self._copy_weights(self.local_critic, self.target_critic)
            self._copy_weights(self.local_actor, self.target_actor)
        else:
            self.soft_update(self.local_critic, self.target_critic, self.tau)
            self.soft_update(self.local_actor, self.target_actor, self.tau)

        self.learn_step += 1

        return a1, c1

    def _copy_weights(self, source_network, target_network):
        """Copy source network weights to target"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def checkpoint(self):
        """Checkpoint actor and critic models"""
        torch.save(self.local_critic.state_dict(), 'checkpoint_critic_{}.pth'.format(self.agent_index))
        torch.save(self.local_actor.state_dict(), 'checkpoint_actor_{}.pth'.format(self.agent_index))
        torch.save(self.critic_optimizer.state_dict(), 'checkpoint_critic_{}_optimizer.pth'.format(self.agent_index))
        torch.save(self.actor_optimizer.state_dict(), 'checkpoint_actor_{}_optimizer.pth'.format(self.agent_index))


class MADDPGAgentTrainer():
    def __init__(self, state_size, fc1, fc2, action_size, num_agents, gamma, tau, lr_actor, lr_critic, buffer_size, batch_size, update_every, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau

        # initialise all agents
        self.agents = [MADDPG(state_size, fc1, fc2, action_size, self.num_agents, i, self.gamma, self.tau, lr_actor, lr_critic, random_seed=random_seed) for i in range(self.num_agents)]
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, random_seed)
        self.learn_step = 0

    def act(self, states, add_noise=True):
        """Executes act on all the agents
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(states[i], add_noise)
            actions.append(action)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.learn_step += 1
        # store a single entry for each step i.e the experience of each agent for a step gets stored as single entry.
        states = np.expand_dims(states, 0)
        actions = np.expand_dims(np.array(actions).reshape(self.num_agents, self.action_size),0)
        rewards = np.expand_dims(np.array(rewards).reshape(self.num_agents, -1),0)
        dones = np.expand_dims(np.array(dones).reshape(self.num_agents, -1),0)
        next_states = np.expand_dims(np.array(next_states).reshape(self.num_agents, -1), 0)

        self.memory.add(states, actions, rewards, next_states, dones)

        # Get agent to learn from experience if we have enough data/experiences in memory
        if len(self.memory) < self.batch_size:
            return
        if not self.learn_step % self.update_every == 0:
            return
        experiences = self.memory.sample()
        actor_losses = []
        critic_losses = []
        for agent in self.agents:
            actor_loss, critic_loss = agent.learn(self.agents, experiences)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

    def reset(self):
        """Resets the noise for each agent"""
        for agent in self.agents:
            agent.reset()

    def checkpoint(self):
        """Checkpoint actor and critic models"""
        for agent in self.agents:
            agent.checkpoint()
