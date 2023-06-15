#!/usr/bin/env python3

#####################################################################
# This script presents how to join and play a deathmatch game,
# that can be hosted using cig_multiplayer_host.py script.
#####################################################################

import os
from random import choice
from DDQLAgent import DDQNAgent as Agent
from utils import StateWrapper
import vizdoom as vzd
import numpy as np

n_actions = 7
game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
resolution = (1, 160, 120)
agent = Agent(input_dims=(1, resolution[1], resolution[2] * 3), n_actions=2 ** 7, learning_rate=0.0001)

load_dir = f"./tmp/ddql/model2load"

agent.load_model(load_dir)

# game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Join existing game.
game.add_game_args(
    "-join 127.0.0.1 -port 5029"
)  # Connect to a host for a multiplayer game.

game.add_game_args("+name Szymon_Kuzik_DDQL +colorset 7")

game.set_mode(vzd.Mode.ASYNC_PLAYER)
# game.set_window_visible(False)
game.init()

player_number = int(game.get_game_variable(vzd.GameVariable.PLAYER_NUMBER))
last_frags = 0

state_stack = StateWrapper(resolution, game.get_state().game_variables.shape)

while not game.is_episode_finished():

    # Get the state.
    state_stack(game.get_state().screen_buffer, game.get_state().game_variables)
    s = game.get_state()

    state, variables = state_stack()
    action = agent.get_best_action(state, variables)
    action = np.array([action])
    action_vector = np.flip(((action[:, None] & (1 << np.arange(n_actions))) > 0).astype(int)).reshape(-1)

    # Make your action.
    game.make_action(action_vector)

    # Check if player is dead
    if game.is_player_dead():
        # Use this to respawn immediately after death, new state will be available.
        game.respawn_player()

game.close()
