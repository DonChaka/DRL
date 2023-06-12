import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy import ndarray
import os
from tqdm import tqdm


class ReplayBuffer(object):
    def __init__(self, mem_size, state_shape, n_actions, n_vars):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.vars_memory = np.zeros((self.mem_size, n_vars*3), dtype=np.float32)
        self._state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self._vars_memory = np.zeros((self.mem_size, n_vars*3), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, variables, action, reward, _state, _variables, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.vars_memory[index] = variables
        self._state_memory[index] = _state
        self._vars_memory[index] = _variables
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
        variables = self.vars_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        _states = self._state_memory[batch]
        _variables = self._vars_memory[batch]
        done = self.terminal_memory[batch]

        return states, variables, actions, rewards, _states, _variables, done

def conv_block(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
    )

class DQN(nn.Module):
    def __init__(self, lr, state_shape, n_actions, name, fname='tmp/ddql'):
        super(DQN, self).__init__()

        self.save_file_name = name + '_ddql'
        self.save_dir = fname

        self.conv1 = conv_block(state_shape[0], 32, 9)
        self.conv2 = conv_block(32, 32, 7)
        self.conv3 = conv_block(32, 64, 5)
        self.conv4 = conv_block(64, 128, 3)
        self.conv5 = conv_block(128, 128, 3)
        self.conv6 = conv_block(128, 256, 1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3632, 256)
        self.fc2 = nn.Linear(256, 512)
        self.output = nn.Linear(512, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0') if T.cuda.is_available() else T.device('cpu')
        self.to(self.device)

    def forward(self, state, variables):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.conv4(state))
        state = F.relu(self.conv5(state))
        state = F.relu(self.conv6(state))
        state = self.flatten(state)
        state = T.cat([state, variables], dim=1)

        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions = self.output(state)

        return actions

    def save_model(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        T.save(self.state_dict(), os.path.join(save_dir, self.save_file_name))

    def load_model(self, load_dir=None):
        if load_dir is None:
            load_dir = self.save_dir
        self.load_state_dict(T.load(os.path.join(load_dir, self.save_file_name)))


class DDQNAgent:
    def __init__(self, input_dims, n_actions, learning_rate, update_rate=500):
        self.state_size = input_dims
        self.action_size = n_actions
        self.memory = ReplayBuffer(15_000, input_dims, n_actions, 16)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0005
        self.network_sync_rate = update_rate
        self.update_cntr = 0
        self.q = DQN(learning_rate, input_dims, n_actions, 'q_eval')
        self.q_target = DQN(learning_rate, input_dims, n_actions, 'q_target')
        self.update_weights(1)
        self.evaluate = False
        self.tau = 0.0025

    def store_transition(self, state, variables, action, reward, _state, _vars, terminal):
        self.memory.store_transition(state, variables, action, reward, _state, _vars, terminal)

    def choose_action(self, state, variables):
        if np.random.random() <= self.epsilon and not self.evaluate:
            action = np.random.choice(self.action_size)
        else:
            state = np.expand_dims(state, 0)
            variables = np.expand_dims(variables, 0)
            state = T.tensor(state, dtype=T.float32).to(self.q.device)
            variables = T.tensor(variables, dtype=T.float32).to(self.q.device)
            actions = self.q.forward(state, variables)
            action = T.argmax(actions).item()

        return action

    def get_best_action(self, state, variables):
        state = T.tensor(state).to(self.q.device)
        variables = T.tensor(variables, dtype=T.int16).to(self.q.device)
        actions = self.q.forward(state, variables)
        action = T.argmax(actions).item()

        return action

    def learn(self, batch_size=64):
        if self.memory.mem_cntr < batch_size:
            return

        states, variables, actions, rewards, _states, _variables, done = self.memory.sample_buffer(batch_size)

        states = T.tensor(states).to(self.q.device)
        _states = T.tensor(_states).to(self.q.device)
        variables = T.tensor(variables, dtype=T.int16).to(self.q.device)
        _variables = T.tensor(_variables, dtype=T.int16).to(self.q.device)

        q_pred = self.q(states, variables)
        with T.no_grad():
            q_next = self.q_target.forward(_states, _variables).cpu().detach().numpy()
            q_target = q_pred.cpu().detach().numpy().copy()

            max_actions = np.argmax(q_next, axis=1)

            batch_index = np.arange(batch_size, dtype=np.int32)

            q_target[batch_index, actions] = rewards + self.gamma * q_next[batch_index, max_actions] * (1 - done)
            q_target = T.tensor(q_target).to(self.q.device)

        # q_pred = self.q(states, variables)
        self.q.optimizer.zero_grad()
        loss = self.q.loss(q_pred, q_target).to(self.q.device)
        loss.backward()
        self.q.optimizer.step()

        self.update_weights()
        self.update_epsilon_value()

    def update_epsilon_value(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self, path=None):
        print('----- Saving models -----')
        self.q.save_model(path)
        self.q_target.save_model(path)

    def load_model(self, path=None):
        print('----- Loading models -----')
        self.q.load_model(path)
        self.q_target.load_model(path)



    def update_weights(self, tau=None):
        if tau is None:
            tau = self.tau

        online_params = self.q.named_parameters()
        target_params = self.q_target.named_parameters()

        online_state_dict = dict(online_params)
        target_state_dict = dict(target_params)

        for name in target_state_dict:
            target_state_dict[name] = tau * online_state_dict[name].clone() + (1 - tau) * target_state_dict[name].clone()

        self.q_target.load_state_dict(target_state_dict)