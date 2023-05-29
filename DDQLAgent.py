import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy import ndarray
from tqdm import tqdm


class ReplayBuffer(object):
    def __init__(self, mem_size, state_shape):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, _state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = _state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
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


class DQN(nn.Module):
    def __init__(self, lr, state_shape, n_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(state_shape[0], 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        # self.conv4 = nn.Conv2d(128, 256, 1, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 256)
        self.fc2 = nn.Linear(256, 512)
        self.output = nn.Linear(512, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0') if T.cuda.is_available() else T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = self.flatten(state)

        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions = self.output(state)

        return actions


class DDQNAgent:
    def __init__(self, state_size, action_size, learning_rate, update_rate=500):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(500_000, state_size)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005
        self.network_sync_rate = update_rate
        self.update_cntr = 0
        self.q = DQN(learning_rate, state_size, action_size)
        self.q_target = DQN(learning_rate, state_size, action_size)
        self.update_weights()
        self.evaluate = False

    def store_transition(self, state, action, reward, _state, done):
        self.memory.store_transition(state, action, reward, _state, done)

    def choose_action(self, state):
        if np.random.random() <= self.epsilon and not self.evaluate:
            action = np.random.choice(self.action_size)
        else:
            state = T.tensor(state).to(self.q.device)
            actions = self.q.forward(state)
            action = T.argmax(actions).item()

        return action

    def get_best_action(self, state):
        state = T.tensor(state).to(self.q.device)
        actions = self.q.forward(state)
        action = T.argmax(actions).item()

        return action

    def learn(self, batch_size):
        if self.memory.mem_cntr < batch_size:
            return

        states, actions, rewards, _states, done = self.memory.sample_buffer(batch_size)

        states = T.tensor(states).to(self.q.device)
        _states = T.tensor(_states).to(self.q_target.device)

        q_test = self.q(states)
        with T.no_grad():
            q_next = self.q_target.forward(_states).cpu().detach().numpy()
            q_target = q_test.cpu().detach().numpy().copy()

            max_actions = np.argmax(q_next, axis=1)

            batch_index = np.arange(batch_size, dtype=np.int32)

            q_target[batch_index, actions] = rewards + self.gamma * q_next[batch_index, max_actions] * (1 - done)
            q_target = T.tensor(q_target).to(self.q.device)

        q_pred = self.q(states)
        self.q.optimizer.zero_grad()
        loss = self.q.loss(q_pred, q_target).to(self.q.device)
        loss.backward()
        self.q.optimizer.step()

    def update_epsilon_value(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def update_weights(self):
        self.q_target.load_state_dict(self.q.state_dict())
