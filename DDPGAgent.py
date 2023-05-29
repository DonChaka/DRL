import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy import ndarray


# from utils import ReplayBuffer

class ReplayBuffer(object):
    def __init__(self, mem_size, state_shape, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, _state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = _state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        return np.exp(x) / np.exp(x).sum()

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        _states = self.new_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, _states, done


def conv_block(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
    )


class Critic(nn.Module):
    def __init__(self, beta, input_dims, n_actions, dense1_dims, dense2_dims, name, fname='tmp\ddpg\\pytorch'):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.dense1_dims = dense1_dims
        self.dense2_dims = dense2_dims
        self.checkpoint_file = os.path.join(fname, name + '_ddpg')

        self.conv1 = conv_block(1, 32, 3)
        self.conv2 = conv_block(32, 64, 3)
        self.conv3 = conv_block(64, 64, 3)
        self.conv5 = conv_block(64, 128, 1)

        self.dense1 = nn.Linear(24192, self.dense1_dims)
        f1 = 1 / np.sqrt(self.dense1.weight.data.size()[0])
        nn.init.uniform_(self.dense1.weight.data, -f1, f1)
        nn.init.uniform_(self.dense1.bias.data, -f1, f1)
        self.norm1 = nn.LayerNorm(self.dense1_dims)

        self.dense2 = nn.Linear(self.dense1_dims, self.dense2_dims)
        f2 = 1 / np.sqrt(self.dense2.weight.data.size()[0])
        nn.init.uniform_(self.dense2.weight.data, -f2, f2)
        nn.init.uniform_(self.dense2.bias.data, -f2, f2)
        self.norm2 = nn.LayerNorm(self.dense2_dims)

        self.action_value = nn.Linear(self.n_actions, dense2_dims)

        f3 = 0.003
        self.q = nn.Linear(self.dense2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0')

        self.to(self.device)

    def forward(self, state, action):
        state = self.conv1(state)
        state = self.conv2(state)
        state = self.conv3(state)
        state = self.conv5(state)
        state = state.view(state.size(0), -1)
        state_value = self.dense1(state)
        state_value = self.norm1(state_value)
        state_value = F.relu(state_value)
        state_value = self.dense2(state_value)
        state_value = self.norm2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, alfa, input_dims, n_binary_actions, n_continous_actions, dense1_dims, dense2_dims, name,
                 fname='tmp/ddpg/pytorch'):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.n_binary_actions = n_binary_actions
        self.n_continuous_actions = n_continous_actions
        self.dense1_dims = dense1_dims
        self.dense2_dims = dense2_dims
        self.checkpoint_file = os.path.join(fname, name + '_ddpg')

        self.conv1 = conv_block(1, 32, 3)
        self.conv2 = conv_block(32, 64, 5)
        self.conv3 = conv_block(64, 64, 3)
        self.conv5 = conv_block(64, 128, 1)

        self.dense1 = nn.Linear(21504, self.dense1_dims)
        f1 = 1 / np.sqrt(self.dense1.weight.data.size()[0])
        nn.init.uniform_(self.dense1.weight.data, -f1, f1)
        nn.init.uniform_(self.dense1.bias.data, -f1, f1)
        self.norm1 = nn.LayerNorm(self.dense1_dims)

        self.dense2 = nn.Linear(self.dense1_dims, self.dense2_dims)
        f2 = 1 / np.sqrt(self.dense2.weight.data.size()[0])
        nn.init.uniform_(self.dense2.weight.data, -f2, f2)
        nn.init.uniform_(self.dense2.bias.data, -f2, f2)
        self.norm2 = nn.LayerNorm(self.dense2_dims)

        f3 = 0.003
        self.bin_actions = nn.Linear(self.dense2_dims, self.n_binary_actions)
        nn.init.uniform_(self.bin_actions.weight.data, -f3, f3)
        nn.init.uniform_(self.bin_actions.bias.data, -f3, f3)

        self.con_actions = nn.Linear(self.dense2_dims, self.n_continuous_actions)
        nn.init.uniform_(self.con_actions.weight.data, -f3, f3)
        nn.init.uniform_(self.con_actions.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alfa)
        self.device = T.device('cuda:0')
        self.to(self.device)

    def forward(self, state):
        state = self.conv1(state)
        state = self.conv2(state)
        state = self.conv3(state)
        state = self.conv5(state)
        state = T.flatten(state, 1)
        state = self.dense1(state)
        state = self.norm1(state)
        state = T.relu(state)
        state = self.dense2(state)
        state = self.norm2(state)
        state = T.relu(state)
        binary = T.sigmoid(self.bin_actions(state))

        continuous = self.con_actions(state)

        output = T.cat([binary, continuous], dim=1)

        return output

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, alpha, beta, input_dims, n_actions, n_binary_actions, n_continuous_actions, tau, gamma=0.99, mem_size=20_000, dense1_size=400,
                 dense2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_binary_actions = n_binary_actions
        self.n_continuous_actions = n_continuous_actions
        self.max_action = 100
        self.min_action = -100

        self.actor = Actor(
            alpha, input_dims, n_binary_actions, n_continuous_actions, dense1_size, dense2_size, name='Actor'
        )
        self.target_actor = Actor(
            alpha, input_dims, n_binary_actions, n_continuous_actions, dense1_size, dense2_size, name='TargetActor'
        )

        self.critic = Critic(beta, input_dims, n_actions, dense1_size, dense2_size, name='Critic')
        self.target_critic = Critic(beta, input_dims, n_actions, dense1_size, dense2_size, name='TargetCritic')

        self.noise = 0.02

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[
                name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[
                name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def choose_action(self, state, train=True):
        self.actor.eval()
        state = np.expand_dims(state, 0)
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)

        if train:
            mu += T.normal(mean=0.0, std=self.noise, size=[self.n_actions]).to(self.actor.device)

        mu = T.clip(mu, self.min_action, self.max_action)
        actions = mu.cpu().detach().numpy()
        actions[:, :self.n_binary_actions] = (actions[:, :self.n_binary_actions] > 0.5)

        self.actor.train()
        return actions[0], mu.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, _state, terminal):
        self.memory.store_transition(state, action, reward, _state, terminal)

    def save_model(self):
        print('----- Saving models -----')
        self.actor.save_model()
        self.target_actor.save_model()
        self.critic.save_model()
        self.target_critic.save_model()

    def load_model(self):
        print('----- Loading models -----')
        self.actor.load_model()
        self.target_actor.load_model()
        self.critic.load_model()
        self.target_critic.load_model()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, state_, terminal = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        terminal = T.tensor(terminal).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(state_)
        critic_value_ = self.target_critic.forward(state_, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for i in range(self.batch_size):
            target.append(reward[i] + self.gamma * critic_value_[i] * terminal[i])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
