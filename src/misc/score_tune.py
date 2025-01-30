import os
import logging
import torch
import optuna
from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.training.train_main import train_agents
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.env.liars_deck_env_utils_2 import decode_action, encode_hand
import numpy as np
import json
import optuna.visualization


def run_obp_inference(obp_model, obs_array, device, num_players):
    if obp_model is None:
        return [0.0] * (num_players - 1)

    opp_feature_dim = config.OPPONENT_INPUT_DIM
    non_opponent_length = 2 + 1 + config.NUM_PLAYERS

    obp_probs = []
    for i in range(num_players - 1):
        opp_vec = obs_array[non_opponent_length + i * opp_feature_dim : non_opponent_length + (i + 1) * opp_feature_dim]
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = obp_model(opp_vec_tensor)
            probs = torch.softmax(logits, dim=-1)
            bluff_prob = probs[0, 1].item()
            obp_probs.append(bluff_prob)
    return obp_probs


class FastEmptyBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, agent):
        hand = self.env.players_hands.get(agent, [])
        table_card = self.env.table_card

        non_table_cards = [card for card in hand if card != table_card and card != "Joker"]
        table_cards = [card for card in hand if card == table_card or card == "Joker"]

        if non_table_cards:
            count = min(len(non_table_cards), 3)
            return 3 + (count - 1)
        elif table_cards:
            count = min(len(table_cards), 3)
            return count - 1
        else:
            return 6  # Challenge action


def evaluate_agents_against_fastempty(env, trained_agents, device, episodes=10):
    def run_games(players_in_this_game):
        wins = {pid: 0 for pid in players_in_this_game}
        action_counts = {pid: {a: 0 for a in range(config.OUTPUT_DIM)} for pid in players_in_this_game}

        for _ in range(episodes):
            obs, infos = env.reset()
            episode_rewards = {pid: 0.0 for pid in players_in_this_game}

            while env.agent_selection is not None:
                agent = env.agent_selection

                if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                    env.step(None)
                    continue

                observation_dict = env.observe(agent)
                observation = observation_dict[agent]
                player_data = players_in_this_game[agent]

                if player_data.get('fast_empty_bot', False):
                    action = player_data['bot'].select_action(agent)
                else:
                    obp_probs = run_obp_inference(player_data['obp_model'], observation, device, env.num_players)
                    final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)
                    obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        probs, _ = player_data['policy_net'](obs_tensor, None)
                    probs = torch.clamp(probs, min=1e-8)
                    probs /= probs.sum(dim=-1, keepdim=True)
                    m = torch.distributions.Categorical(probs)
                    action = m.sample().item()

                action_counts[agent][action] += 1
                env.step(action)
                reward = env.rewards.get(agent, 0.0)
                episode_rewards[agent] += reward

            max_reward = max(episode_rewards.values())
            winners = [pid for pid, rw in episode_rewards.items() if rw == max_reward]
            for winner in winners:
                wins[winner] += 1 / len(winners)

        return wins, action_counts

    players_set_1 = {
        'player_0': {'fast_empty_bot': True, 'bot': FastEmptyBot(env)},
        'player_1': trained_agents['player_1'],
        'player_2': trained_agents['player_2']
    }

    players_set_2 = {
        'player_0': trained_agents['player_0'],
        'player_1': {'fast_empty_bot': True, 'bot': FastEmptyBot(env)},
        'player_2': trained_agents['player_2']
    }

    wins_set_1, action_counts_set_1 = run_games(players_set_1)
    wins_set_2, action_counts_set_2 = run_games(players_set_2)

    combined_wins = {}
    for pid in set(wins_set_1.keys()).union(wins_set_2.keys()):
        combined_wins[pid] = wins_set_1.get(pid, 0) + wins_set_2.get(pid, 0)

    combined_action_counts = {}
    all_players = set(action_counts_set_1.keys()).union(action_counts_set_2.keys())
    for pid in all_players:
        combined_action_counts[pid] = {a: action_counts_set_1.get(pid, {}).get(a, 0) + action_counts_set_2.get(pid, {}).get(a, 0) for a in range(config.OUTPUT_DIM)}

    total_games = episodes * 2
    trained_wins = combined_wins.get('player_2', 0) + combined_wins.get('player_1', 0) + combined_wins.get('player_0', 0)
    win_rate = trained_wins / total_games

    return win_rate, combined_action_counts


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scoring_params = {
        "play_reward_per_card": trial.suggest_categorical("play_reward_per_card", [0, 1, 2]),
        'play_reward': trial.suggest_categorical('play_reward', [0, 1, 2]),
        'invalid_play_penalty': trial.suggest_categorical('invalid_play_penalty', [-3, -2, -1, 0]),
        'challenge_success_challenger_reward': trial.suggest_categorical('challenge_success_challenger_reward', [5, 10, 15]),
        'challenge_success_claimant_penalty': trial.suggest_categorical('challenge_success_claimant_penalty', [-5, -4, -3]),
        'challenge_fail_challenger_penalty': trial.suggest_categorical('challenge_fail_challenger_penalty', [-3, -2, -1]),
        'challenge_fail_claimant_reward': trial.suggest_categorical('challenge_fail_claimant_reward', [4, 5, 6]),
        'forced_challenge_success_challenger_reward': trial.suggest_categorical('forced_challenge_success_challenger_reward', [8, 10, 12]),
        'forced_challenge_success_claimant_penalty': trial.suggest_categorical('forced_challenge_success_claimant_penalty', [-4, -3, -2]),
        'invalid_challenge_penalty': trial.suggest_categorical('invalid_challenge_penalty', [-5, -4, -3]),
        'termination_penalty': trial.suggest_categorical('termination_penalty', [-5, -4, -3]),
        'game_win_bonus': trial.suggest_categorical('game_win_bonus', [10, 15, 20]),
        'game_lose_penalty': trial.suggest_categorical('game_lose_penalty', [-5, -4, -3]),
        'hand_empty_bonus': trial.suggest_categorical('hand_empty_bonus', [1, 2, 3])
    }

    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    env.scoring_params = scoring_params
    obs, infos = env.reset()
    agents = env.agents
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)

    training_results = train_agents(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=False,
        log_tensorboard=False
    )

    trained_agents = training_results.get('agents', {})
    if not trained_agents:
        logging.error(f"Trial {trial.number} pruned due to missing trained agents.")
        raise optuna.TrialPruned()

    win_rate, action_usage_counts = evaluate_agents_against_fastempty(
        env, trained_agents, device, episodes=10
    )

    challenge_amount = getattr(config, 'challenge_amount', 5)
    action_6_violation = any(actions.get(6, 0) <= challenge_amount for pid, actions in action_usage_counts.items() if pid in trained_agents)

    if action_6_violation:
        win_rate *= 0.5

    logging.info(f"Trial {trial.number}: Win Rate = {win_rate}, Params = {scoring_params}")
    return win_rate


def save_results_to_file(study, filename=config.OPTUNA_RESULTS_FILE):
    results = {
        "best_trial": {
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "number": study.best_trial.number,
        },
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Optuna results saved to {filename}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("BayesianOptimization")

    storage_url = "sqlite:///optuna_study2.db"
    study = optuna.create_study(
        direction='maximize',
        storage=storage_url,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=150, n_jobs=1)

    best_trial = study.best_trial
    logger.info(f"Best trial parameters: {best_trial.params}")
    logger.info(f"Best trial score (Win Rate): {best_trial.value}")

    save_results_to_file(study)

    try:
        fig_optimization = optuna.visualization.plot_optimization_history(study)
        fig_optimization.write_image("optimization_history.png")
        logger.info("Saved optimization history plot to 'optimization_history.png'.")

        fig_param_importances = optuna.visualization.plot_param_importances(study)
        fig_param_importances.write_image("param_importances.png")
        logger.info("Saved parameter importances plot to 'param_importances.png'.")
    except Exception as e:
        logger.error(f"Failed to save plots: {e}")


if __name__ == "__main__":
    main()
