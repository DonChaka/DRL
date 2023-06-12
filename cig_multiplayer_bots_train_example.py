#!/usr/bin/env python3

#####################################################################
# This script presents how to play a deathmatch game with built-in bots.
#####################################################################

import os
import pickle
from random import choice
import time

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
from tqdm import tqdm
import os
from datetime import datetime

from DDPGAgent import Agent
from utils import StateWrapper

train_epochs = 100
learning_steps_per_epoch = 5_000
replay_memory_size = 5000
n_bots = 1

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (1, 160, 120)
episodes_to_watch = 10

save_model = True
pretrain_on_data = False
load_model = True
skip_learning = False
train = True


game = vzd.DoomGame()
# Use CIG example config or your own.
game.load_config("./multi.cfg")

# Start multiplayer game only with your AI
# (with options that will be used in the competition, details in cig_mutliplayer_host.py example).
game.add_game_args(
    "-host 1 "
    # This machine will function as a host for a multiplayer game with this many players (including this machine).
    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
    # "-port 5029 "  # Specifies the port (default is 5029).
    # "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
    "-deathmatch "  # Deathmatch rules are used for the game.
    "+timelimit 10.0 "  # The game (episode) will end after this many minutes have elapsed.
    "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
    "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
    "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 0 "  # Sets delay between respawns (in seconds, default is 0).
    "+viz_nocheat 1"
)

# Bots are loaded from file, that by default is bots.cfg located in the same dir as ViZDoom exe
# Other location of bots configuration can be specified by passing this argument
game.add_game_args("+viz_bots_path ./bots.cfg")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")

### Change game mode 
# game.set_mode(vzd.Mode.ASYNC_PLAYER)
# game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
# game.set_mode(vzd.Mode.SPECTATOR)
game.set_mode(vzd.Mode.PLAYER)
game.set_window_visible(False)
game.set_doom_map('map03')
game.set_ticrate(35)
game.set_console_enabled(True)
game.init()

# Three example sample actions
actions = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]
last_frags = 0

# Play with this many bots
bots = 5

# Run this many episodes
episodes = 25_000

### DEFINE YOUR AGENT HERE (or init)

agent = Agent(alpha=0.0001, beta=0.0002, input_dims=(1, resolution[1], resolution[2] * 3),
              tau=0.0025, n_actions=9, n_binary_actions=7, n_continuous_actions=2)

if pretrain_on_data:
    with open("./full_memory_buffer", "rb") as f:
        agent.memory = pickle.load(f)
    for i in tqdm(range(5_000)):
        agent.learn()

load_dir = f"./tmp/ddpg/model2load"

if load_model:
    agent.load_model(load_dir)

# Define rewards
hp_change_reward = 0.5
hit_count_reward = 2
kill_count_reward = 5
hits_taken_reward = 0.5
armor_change_reward = 0.5
frag_count_reward = 5


def calculate_reward(reward, vars, _vars, done):
    if done:
        return reward
    if _vars[0] < vars[0]:
        reward += hp_change_reward * np.sign(_vars[0] - vars[0])
    if _vars[1] > vars[1]:
        reward += hit_count_reward
    if _vars[2] > vars[2]:
        reward += kill_count_reward
    if _vars[3] > vars[3]:
        reward += hits_taken_reward
    if _vars[5] != vars[5]:
        reward += armor_change_reward * np.sign(_vars[5] - vars[5])
    if _vars[15] != vars[15]:
        reward += frag_count_reward

    return reward


for i in range(episodes):
    print("Episode #" + str(i + 1))
    state_stack = StateWrapper(resolution, game.get_state().game_variables.shape)
    state_stack(game.get_state().screen_buffer, game.get_state().game_variables)

    if save_model:
        # imprint timestamp into save path in hh:mm format
        current_time = datetime.now().strftime("%H-%M")
        os.makedirs(f"./tmp/ddpg/{current_time}", exist_ok=True)

        agent.save_model(f"./tmp/ddpg/{current_time}")
        # agent.save_model(f"models/{int(time())}")
        # agent.save_model()

    game.send_game_command("removebots")
    for _ in range(bots):
        game.send_game_command("addbot")

    # Change the bots difficulty
    # Valid args: 1, 2, 3, 4, 5 (1 - easy, 5 - very hard)
    game.send_game_command("pukename change_difficulty 1")

    progress_bar = tqdm(unit=" iteration", unit_scale=True, desc=f'Episode #{i + 1}')

    # Play until the game (episode) is over.
    while not game.is_episode_finished():
        progress_bar.update(1)

        state, variables = state_stack()
        action, net_ret = agent.choose_action((state, variables), train=train)
        reward = game.make_action(action)
        s = game.get_state()
        if game.is_episode_finished():
            break

        done = game.is_player_dead()

        if not done:
            _state, _variabels = state_stack(s.screen_buffer, s.game_variables)
        else:
            _state, _variabels = state_stack.reset()

        reward = calculate_reward(reward, variables, _variabels, done)

        # TRAIN YOUR AGENT HERE
        agent.store_transition(state, variables, net_ret, reward, _state, _variabels, done)
        agent.learn()

        # Check if player is dead
        if game.is_player_dead():
            print("Player died.")
            state_stack(s.screen_buffer, s.game_variables)
            game.respawn_player()
            break

    print("Episode finished.")
    print("************************")

    print("Results:")
    server_state = game.get_server_state()
    for j in range(len(server_state.players_in_game)):
        if server_state.players_in_game[j]:
            print(
                server_state.players_names[j]
                + ": "
                + str(server_state.players_frags[j])
            )
    print("************************")

    # Starts a new episode. All players have to call new_episode() in multiplayer mode.
    game.new_episode()

game.close()
