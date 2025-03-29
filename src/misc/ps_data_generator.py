#!/usr/bin/env python3
# ps_data_generator.py - geneatings perfect game data to train the ppo agent.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
import random
import numpy as np
import torch
import argparse
import pickle
from collections import defaultdict, deque
from tqdm import tqdm

# Environment imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action, encode_hand
from src import config

# Import opponent models
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

# Import training utilities
from src.training.train_utils import load_specific_historical_models

# Import PS and opponent models
from src.model.PS import PerfectSearch


Set up logging and output directory

Set random seeds for reproducibility

Initialize environment with fixed number of players
Define training player and list of opponent players

Initialize opponent pool:
    Load predefined (hardcoded) opponent agents
    Assign each one a label

If historical opponents are to be included:
    Load historical models
    Assign unique labels to each
    Add them to the opponent pool

Determine total number of opponent types

Initialize:
    - List to store all data
    - Dictionary to store statistics

Start generating episodes

For each episode:
    Choose a set of opponents for this episode (randomly or systematically)
    Instantiate those opponents and log combination used

    Reset environment with a seed tied to the episode number

    Initialize PS engine:
        Pass in environment, training player, and current opponents

    Simulate beliefs about opponents (noisy approximations of true types)

    While game is not done:
        If training player’s turn:
            Get observation and valid actions
            Combine belief vectors

            Get internal game info (e.g. hand, table card, etc.)

            Try:
                Use PS to compute:
                    - Action probabilities
                    - Best action
                    - Value estimate of best action

                Store transition with:
                    - Observation and belief
                    - Action dist, best action, value
                    - Action mask and state snapshot
                    - Opponent labels and types

                Step in environment using best action

            On failure:
                Log warning
                Step using random valid action or fallback

        Else:
            For opponent player:
                If PS preplanned action exists:
                    Use it
                Else:
                    Use opponent’s own policy
                Step in environment with action

        Check if episode has ended

    After episode:
        Retrieve game result and reward
        Update each transition with result

        Add all episode transitions to dataset
        Update statistics (wins, combos, actions, etc.)

        If episode count hits save checkpoint:
            Save current data and stats
            Log status
            Clear memory if saving in chunks

After final episode:
    Save full data if not already saved
    Save final stats

Log:
    - Total time and episodes
    - Transitions per episode
    - Final win rate
    - Opponent combo distribution
    - Action distribution