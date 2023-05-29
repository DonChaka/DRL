#!/usr/bin/env python3

#####################################################################
# This script presents how to host a deathmatch game.
#####################################################################

import os
from random import choice

import vizdoom as vzd

from utils import StateWrapper

from DDPGAgent import Agent as DDPGAgent

target_resolution = (1, 160, 120)
state_stack = StateWrapper(target_resolution)
agent = DDPGAgent(alpha=0.0001, beta=0.0002, input_dims=(1, target_resolution[1], target_resolution[2]*3), tau=0.005,
                  n_actions=9, n_binary_actions=7, n_continuous_actions=2)

game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "cig.cfg"))

game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Host game with options that will be used in the competition.
game.add_game_args(
    "-host 4"
    # This machine will function as a host for a multiplayer game with this many players (including this machine).
    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
    "-port 5029 "  # Specifies the port (default is 5029).
    "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
    "-deathmatch "  # Deathmatch rules are used for the game.
    # "+timelimit 10.0 "  # The game (episode) will end after this many minutes have elapsed.
    # "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
    "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
    # "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 10 "  # Sets delay between respawns (in seconds, default is 0).
    "+viz_nocheat 1"
)  # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

# This can be used to host game without taking part in it (can be simply added as argument of vizdoom executable).
# game.add_game_args("+viz_spectator 1")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name DQN +colorset 0")

# During the competition, async mode will be forced for all agents.
# game.set_mode(vzd.Mode.PLAYER)
game.set_mode(vzd.Mode.ASYNC_PLAYER)

# game.set_window_visible(False)

game.init()

state_stack.reset()
while not game.is_episode_finished():

    # Get the state.
    s = game.get_state()
    state_stack(s.screen_buffer)
    state = state_stack()

    action = agent.choose_action(state, True)
    reward = game.make_action(action)

    if game.is_player_dead():
        reward -= 100


    state_stack(s.screen_buffer)
    _state = state_stack()
    frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
    if frags != last_frags:
        last_frags = frags
        print("Player " + str(player_number) + " has " + str(frags) + " frags.")

