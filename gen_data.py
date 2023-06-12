#!/usr/bin/env python3

#####################################################################
# This script presents how to play a deathmatch game with built-in bots.
#####################################################################

import os
from random import choice
import time
# from agent_example import Agent
import vizdoom as vzd
from tqdm import tqdm

from DDPGAgent import Agent, ReplayBuffer
from utils import StateWrapper
import pickle

import numpy as np

game = vzd.DoomGame()
# Use CIG example config or your own.
game.load_config("./multi.cfg")

import csv

# Q-learning settings
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



# Start multiplayer game only with your AI
# (with options that will be used in the competition, details in cig_mutliplayer_host.py example).
game.add_game_args(
    "-host 1 "
    # This machine will function as a host for a multiplayer game with this many players (including this machine).
    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
    # "-port 5029 "  # Specifies the port (default is 5029).
    # "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
    "-deathmatch "  # Deathmatch rules are used for the game.
    "+timelimit 1.0 "  # The game (episode) will end after this many minutes have elapsed.
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
game.set_mode(vzd.Mode.SPECTATOR)
# game.set_mode(vzd.Mode.PLAYER)

game.set_ticrate(35)
# game.set_console_enabled(True)
game.init()

# Three example sample actions

# Define rewards
hp_change_reward = 0.5
hit_count_reward = 2
kill_count_reward = 5
hits_taken_reward = 0.5
armor_change_reward = 0.5
frag_count_reward = 5


def calculate_reward(reward, vars, _vars, done):
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

last_frags = 0

# Play with this many bots
bots = 5

# Run this many episodes
episodes = 5  # 25_000

### DEFINE YOUR AGENT HERE (or init)

memory = ReplayBuffer(15_000, (1, resolution[1], resolution[2] * 3), 9, 16)

filepath = 'framebuffer_action.pickle'
# try:
#     replay_memory = load_json(filepath)
# except:
#     print("json file not exist or is empty")

# replay_memory = load_json(filepath)


iteration = 0
prog_bar = tqdm(total=15_000)
while iteration < 15_000:
    state_stack = StateWrapper(resolution, game.get_state().game_variables.shape)
    state_stack(game.get_state().screen_buffer, game.get_state().game_variables)
    ### Add specific number of bots
    # edit this file to adjust bots).
    game.send_game_command("removebots")
    for i in range(bots):
        game.send_game_command("addbot")

    game.send_game_command("pukename change_difficulty 1")

    while not game.is_episode_finished():
        prog_bar.update(1)
        iteration += 1
        # Get the state.
        state, variables = state_stack()

        game.advance_action()
        action = game.get_last_action()
        # print(f"action: {action}")
        action_reward = 0
        if game.is_episode_finished():
            break

        s = game.get_state()
        done = game.is_player_dead()
        _state, _variabels = state_stack(s.screen_buffer, s.game_variables)

        reward = calculate_reward(action_reward, variables, _variabels, done)
        memory.store_transition(state, variables, action, reward, _state, _variabels, done)

        # Check if player is dead
        if game.is_player_dead():
            # print("Player died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()
            break

    game.new_episode()

with open(filepath, 'wb') as f:
    pickle.dump(memory, f)


game.close()
