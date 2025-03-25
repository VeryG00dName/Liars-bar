import random
import numpy as np

# Import the environment and new observations helper.
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils import get_new_observations

def main():
    # Create the environment with 3 players and render mode set to 'human'
    env = LiarsDeckEnv(num_players=3, render_mode='human')
    # Reset the environment and get initial observations and infos.
    obs, infos = env.reset(seed=42)
    
    print("\n=== Initial Game State ===")
    env.render()

    # Print the initial new observations for all agents.
    new_obs = get_new_observations(env)
    print("\nInitial new observations:")
    for agent, obs_vector in new_obs.items():
        print(f"{agent}: {obs_vector}")

    # Play the game until no agent remains (i.e. env.agent_selection is None).
    while env.agent_selection is not None:
        current_agent = env.agent_selection

        # Obtain and print the new observation for the current agent.
        new_obs = get_new_observations(env, agent_specific=current_agent)
        print(f"\nAgent '{current_agent}' new observation: {new_obs[current_agent]}")

        # Choose a random action if the agent is active; if terminated or eliminated, use None.
        if env.terminations.get(current_agent, False) or env.truncations.get(current_agent, False):
            action = None
        else:
            action = random.choice(range(env.action_spaces[current_agent].n))
        print(f"Agent '{current_agent}' takes action: {action}")

        # Take the action in the environment.
        env.step(action)

        # Render the current game state.
        env.render()

    print("\nGame ended.")

if __name__ == "__main__":
    main()
