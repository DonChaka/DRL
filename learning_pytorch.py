# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

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

from DDPGAgent import Agent
from utils import StateWrapper

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

save_model = True
load_model = True
skip_learning = False

# Configuration file path
config_file_path = './multi.cfg'
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args(
        "-host 1 "
        # This machine will function as a host for a multiplayer game with this many players (including this machine).
        # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
        # "-port 5029 "  # Specifies the port (default is 5029).
        # "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
        "-deathmatch "  # Deathmatch rules are used for the game.
        "+timelimit 5.0 "  # The game (episode) will end after this many minutes have elapsed.
        "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
        "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
        "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
        "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
        "+sv_nocrouch 1 "  # Disables crouching.
        "+viz_respawn_delay 0 "  # Sets delay between respawns (in seconds, default is 0).
        "+viz_nocheat 1"
    )
    game.init()
    print("Doom initialized.")

    return game


def add_bots(game, n_bots=1):
    game.send_game_command('removebots')
    for _ in range(n_bots):
        game.send_game_command('addbot')


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        state_wrapper = StateWrapper(resolution, game.get_state().game_variables.shape)
        while not game.is_episode_finished():
            state = state_wrapper(game.get_state().screen_buffer)
            action, _ = agent.choose_action(state)

            game.make_action(action, frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        state_stack = StateWrapper(resolution, game.get_state().game_variables.shape)
        state_stack(game.get_state().screen_buffer, game.get_state().game_variables)
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch, leave=False):
            s = game.get_state()
            state, variables = state_stack()
            action, net_ret = agent.choose_action((state, variables), train=True)
            reward = game.make_action(action, frame_repeat)

            done = game.is_episode_finished()

            if not done:
                _state, _variabels = state_stack(s.screen_buffer, s.game_variables)
            else:
                _state, _variabels = state_stack.reset()

            agent.store_transition(state, variables, net_ret, reward, _state, _variabels, done)
            agent.learn()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()
                state_stack(s.screen_buffer, s.game_variables)

            global_step += 1

        train_scores = np.array(train_scores)

        if save_model:
            print("Saving the network weights to: ", agent.actor.save_file_name)
            agent.save_model()

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        # test(game, agent)

        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()

    # Initialize our agent with the set parameters
    agent = Agent(alpha=0.0001, beta=0.0002, input_dims=(1, resolution[1], resolution[2] * 3),
                  tau=0.0025, n_actions=9, n_binary_actions=7, n_continuous_actions=2)

    if load_model:
        agent.load_model()

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game=game,
            agent=agent,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        state_stack = StateWrapper(resolution, game.get_state().game_variables.shape)
        while not game.is_episode_finished():
            state = state_stack(game.get_state().screen_buffer)
            action, net_ret = agent.choose_action(state)

            game.set_action(action)
            for __ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
