from collections import deque
import gym
import numpy as np
import random
from tqdm import tqdm
from numpy import ndarray
from env.FrozenLakeMDP import frozenLake
from env.FrozenLakeMDPExtended import frozenLakeExtended
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer(object):
    def __init__(self, mem_size, state_shape):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
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
        max_mem = min(self.mem_cntr, self.mem_size) - 1

        probs = self._softmax(self.reward_memory[:max_mem])
        batch = np.random.choice(max_mem, batch_size, replace=False)
        batch[-1] = self.mem_cntr % self.mem_size

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        _states = self.new_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, _states, done

    def ready(self, batch_size: int) -> bool:
        return self.mem_cntr-1 <= batch_size


class DQNAgent: # Pytorch
    def __init__(self, action_size, state_size, learning_rate, model):
        self.action_size = action_size
        self.memory = ReplayBuffer(1000, state_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = learning_rate
        self.q = model
        self.evaluate = False

    def remember(self, state, action, reward, _state, done):
        self.memory.store_transition(state, action, reward, _state, done)

    def get_action(self, state):
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
        _states = T.tensor(_states).to(self.q.device)


        q_next = self.q(_states).cpu().detach().numpy()
        q_pred = self.q(states)
        q_target = q_pred.cpu().detach().numpy().copy()

        max_actions = np.argmax(q_next, axis=1)

        batch_index = np.arange(batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * q_next[batch_index, max_actions] * (1-done)
        q_target = T.tensor(q_target).to(self.q.device)

        # q_pred = T.tensor(q_pred, requires_grad=True).to(self.q.device)

        self.q.optimizer.zero_grad()
        loss = self.q.loss(q_pred, q_target).to(self.q.device)
        loss.backward()
        self.q.optimizer.step()

    def update_epsilon_value(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min


class DQN(nn.Module):
    def __init__(self, lr, state_shape, n_actions, fc1, fc2):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_shape, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.output = nn.Linear(fc2, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0')
        self.to(self.device)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions = self.output(state)

        return actions


def check_action_values(agent: DQNAgent, state_size: int):
    rets = []
    for i_state in range(state_size):
        state = np.zeros(state_size, dtype=np.float32)
        state[i_state] = 1
        state = T.tensor(state).to(agent.q.device)
        rets.append(agent.q(state).cpu().detach().numpy())
    rets = np.array(rets).reshape(state_size, 4)
    with np.printoptions(precision=4, suppress=True):
        print(rets)


env = frozenLake("8x8")

state_size = env.get_number_of_states()
action_size = len(env.get_possible_actions(None))
learning_rate = 0.0005
model = DQN(learning_rate, state_size, action_size, 32, 32)
agent = DQNAgent(action_size, state_size, learning_rate, model)
# agent.epsilon = 0.75

done = False
batch_size = 128
EPISODES = 60
counter = 0
for e in range(EPISODES):
    summary = []
    if not e % 10:
        check_action_values(agent, state_size)
    for i in tqdm(range(100), desc=f'Epoch: {e}'):
        total_reward = 0
        i_state = env.reset()

        state = np.zeros(state_size, dtype=np.float32)
        state[i_state] = 1

        for time in range(1000):
            action = agent.get_action(state)
            _i_state, reward, done, _ = env.step(action)
            total_reward += reward

            _state = np.zeros(state_size, dtype=np.float32)
            _state[_i_state] = 1

            if np.allclose(state, _state):
                reward = -1
                done = True

            if done and not reward:
                reward = -1

            agent.remember(state, action, reward, _state, done)
            agent.learn(batch_size)

            state = _state
            if done:
                break

        summary.append(total_reward)
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(e, np.mean(summary), agent.epsilon))
    agent.update_epsilon_value()

    if np.mean(summary) > 0.9:
        print("You Win!")
        break