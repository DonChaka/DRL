#!/usr/bin/env python3

# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

import itertools as it
import os
import random
from collections import deque
from time import sleep, time
from typing import Union

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import ndarray
from tqdm import trange

import vizdoom as vzd

from utils import StateWrapper
from tqdm import tqdm
from DDPGAgent import Agent as DDPGAgent

target_resolution = (1, 160, 120)

# Q-learning settings
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (1, 160, 120)
episodes_to_watch = 10
n_bots=1

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

# Configuration file path
config_file_path = 'vizdoom/scenarios/multi.cfg'
# config_file_path = os.path.join(vzd.scenarios_path, "multi.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


def create_simple_game(n_bots=1):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_doom_map("map01")
    game.set
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.add_game_args("+viz_bots_path ./assets/bots.cfg")
    game.add_game_args("+viz_bots_path bots.cfg")
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
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()
    add_bots(game, n_bots)
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        state_stack = StateWrapper(resolution)
        while not game.is_episode_finished():
            state = state_stack(game.get_state().screen_buffer)
            action, _ = agent.choose_action(state, False)

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
    game.close()
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()


def run(game, agent, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()
    state_stack = StateWrapper(resolution)
    for epoch in range(num_epochs):
        game.new_episode()
        add_bots(game, n_bots)

        state_stack.reset()

        train_scores = []
        global_step = 0

        print("\nEpoch #" + str(epoch + 1))

        s = game.get_state()
        state_stack(s.screen_buffer)

        for _ in trange(steps_per_epoch, leave=False):

            state = state_stack()

            action, net_return = agent.choose_action(state, True)
            reward = game.make_action(action, frame_repeat)
            done = game.is_episode_finished()

            if not done:
                _state = state_stack(game.get_state().screen_buffer)
            else:
                _state = state_stack.reset()

            agent.store_transition(state, net_return, reward, _state, done)

            agent.learn()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()
                add_bots(game)
                state_stack.reset()

            global_step += 1

        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(game, agent)
        if save_model:
            print("Saving the network weights")
            agent.save_model()

        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters

    state_stack = StateWrapper(target_resolution)
    agent = DDPGAgent(alpha=0.0001, beta=0.0002, input_dims=(1, target_resolution[1], target_resolution[2] * 3),
                      tau=0.005,
                      n_actions=9, n_binary_actions=7, n_continuous_actions=2)

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
        frame_stack = StateWrapper(resolution)
        while not game.is_episode_finished():
            state = frame_stack(game.get_state().screen_buffer)
            action, _ = agent.choose_action(state, False)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(action)
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
