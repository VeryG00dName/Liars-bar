# src/evaluation/gui_evaluate.py

import torch
import logging
import threading
import numpy as np
from openskill.models import PlackettLuce
from pettingzoo.utils import agent_selector

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src import config
from src.eval.evaluate_tournament import (
    evaluate_agents_tournament,
    update_openskill_ratings
)

def get_hidden_dim_from_state_dict(state_dict, layer_prefix='fc1'):
    weight_key = f"{layer_prefix}.weight"
    if weight_key in state_dict:
        return state_dict[weight_key].shape[0]
    else:
        for key in state_dict.keys():
            if key.endswith('.weight') and ('fc' in key or 'layer' in key):
                return state_dict[key].shape[0]
    raise ValueError(f"Cannot determine hidden_dim from state_dict for layer prefix '{layer_prefix}'.")

class TournamentManager:
    def __init__(self, loaded_models, update_callback, log_callback):
        self.loaded_models = loaded_models
        self.update_callback = update_callback
        self.log_callback = log_callback
        self.openskill_model = PlackettLuce(mu=25.0, sigma=25.0/3, beta=25.0/6)
        self.players = {}
        self.is_running = False
        self.current_round = 0
        self.device = torch.device(config.DEVICE)
        self.env = None
        self.current_tournament_wins = {}

    def initialize_players(self, participant_names):
        """
        Initialize players with models from loaded_models and OpenSkill ratings.
        We store each model under 'policy_net', 'value_net', and 'obp_model'.
        """
        self.players = {}
        for name in participant_names:
            if name in self.loaded_models:
                model_data = self.loaded_models[name]
                
                # Extract hidden dimensions dynamically
                policy_hidden_dim = get_hidden_dim_from_state_dict(model_data['policy_net'])
                policy = PolicyNetwork(
                    config.INPUT_DIM,
                    policy_hidden_dim,
                    config.OUTPUT_DIM,
                    use_lstm=True,
                    use_dropout=True,
                    use_layer_norm=True
                ).to(self.device)
                policy.load_state_dict(model_data['policy_net'])
                policy.eval()
                
                value_hidden_dim = get_hidden_dim_from_state_dict(model_data['value_net'])
                value = ValueNetwork(
                    config.INPUT_DIM,
                    value_hidden_dim,
                    use_dropout=True,
                    use_layer_norm=True
                ).to(self.device)
                value.load_state_dict(model_data['value_net'])
                value.eval()
                
                obp = None
                obs_version = None
                if model_data['obp_model'] is not None:
                    obp_hidden_dim = get_hidden_dim_from_state_dict(model_data['obp_model'])
                    
                    # Determine obs_version based on OBP model's input_dim
                    obp_fc1_weight_shape = model_data['obp_model']['fc1.weight'].shape[1]
                    if obp_fc1_weight_shape == 5:
                        obs_version = 1  # OBS_VERSION_1
                        obp_input_dim = 5
                    elif obp_fc1_weight_shape == 4:
                        obs_version = 2  # OBS_VERSION_2
                        obp_input_dim = 4
                    else:
                        raise ValueError(f"Unknown OBP input_dim {obp_fc1_weight_shape} for player {name}")
    
                    obp = OpponentBehaviorPredictor(
                        input_dim=obp_input_dim,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    ).to(self.device)
                    obp.load_state_dict(model_data['obp_model'])
                    obp.eval()
                else:
                    # If no OBP model, default to obs_version=2 and input_dim=4
                    obs_version = 2
                    obp_input_dim = 4
                
                self.players[name] = {
                    'policy_net': policy,
                    'value_net': value,
                    'obp_model': obp,
                    'obs_version': obs_version,
                    'rating': self.openskill_model.rating(name=name),
                    'score': self.openskill_model.rating(name=name).ordinal(),
                    'wins': 0,
                    'games_played': 0
                }
        self.current_tournament_wins = {name: 0 for name in participant_names}

    def log(self, message, error=False):
        """Send logs to the GUI callback."""
        self.log_callback(message, error=error)

    def run_tournament(self, num_rounds=7, games_per_match=11):
        """
        Run the Swiss-style tournament in a separate thread (so it doesn't block the GUI).
        """
        if self.is_running:
            self.log("Tournament already running!", error=True)
            return

        self.is_running = True
        try:
            self.env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
            self._run_swiss_round(num_rounds, games_per_match)
            self._send_final_results()
        except Exception as e:
            self.log(f"Tournament error: {str(e)}", error=True)
        finally:
            self.is_running = False
            self.env = None

    def _run_swiss_round(self, num_rounds, games_per_match):
        """
        Run a full Swiss tournament with proper win tracking.
        """
        player_ids = list(self.players.keys())
        group_size = config.NUM_PLAYERS
        self.log(f"Starting Swiss tournament with {len(player_ids)} players in {num_rounds} rounds")

        for round_num in range(1, num_rounds + 1):
            self.current_round = round_num
            self.log(f"=== Round {round_num} ===")

            # Sort players by current score (descending)
            sorted_players = sorted(player_ids, key=lambda pid: self.players[pid]['score'], reverse=True)
            
            # Create groups of size group_size
            groups = []
            i = 0
            while i < len(sorted_players):
                if i + group_size <= len(sorted_players):
                    groups.append(sorted_players[i:i + group_size])
                else:
                    # Handle remaining players by merging with the last group
                    if groups:
                        groups[-1].extend(sorted_players[i:])
                    else:
                        groups.append(sorted_players[i:])
                i += group_size

            # Process each group
            for group in groups:
                if len(group) < group_size:
                    self.log(f"Skipping small group: {group}", error=True)
                    continue

                try:
                    # Evaluate the group using evaluate_agents_tournament
                    players_in_this_game = {pid: self.players[pid] for pid in group}

                    cumulative_wins, action_counts, game_wins_list, avg_steps = evaluate_agents_tournament(
                        env=self.env,
                        device=self.device,
                        players_in_this_game=players_in_this_game,
                        episodes=games_per_match
                    )

                    # Update tournament win counts
                    for game_wins in game_wins_list:
                        for pid, wins in game_wins.items():
                            if wins > 0:
                                self.current_tournament_wins[pid] += 1
                                self.players[pid]['wins'] += 1

                    # Update OpenSkill ratings
                    group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
                    update_openskill_ratings(
                        players=self.players,
                        group=group,
                        group_ranking=group_ranking,
                        cumulative_wins=cumulative_wins
                    )

                    # Log group results
                    self.log(f"Group {group} results:")
                    for pid in group:
                        self.log(f"{pid}: {cumulative_wins[pid]} wins (Total: {self.current_tournament_wins[pid]})")

                except Exception as e:
                    self.log(f"Error in group {group}: {str(e)}", error=True)

            # Send intermediate results to GUI
            self._update_scores()

    def _update_scores(self):
        """Send intermediate results to GUI."""
        sorted_players = sorted(
            self.players.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        self.update_callback({
            'round': self.current_round,
            'players': [{
                'name': name,
                'score': data['score'],
                'wins': data['wins'],
                'games': data['games_played']
            } for name, data in sorted_players]
        })

    def _send_final_results(self):
        """
        Send final results with accurate win counts and rankings.
        """
        sorted_players = sorted(
            self.players.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        final_results = {
            'final': True,
            'players': [{
                'name': name,
                'score': data['score'],
                'wins': self.current_tournament_wins[name],
                'games': data['games_played']
            } for name, data in sorted_players]
        }

        # Log final results
        self.log("=== Final Tournament Results ===")
        for player in final_results['players']:
            self.log(f"{player['name']}: Score {player['score']:.2f} - Wins {player['wins']}")

        # Send results to GUI
        self.update_callback(final_results)
