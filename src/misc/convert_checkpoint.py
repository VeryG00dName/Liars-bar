import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import torch
import torch.optim as optim

# Import the save_checkpoint function and config from train_utils.py
from src.training.train_utils import save_checkpoint
from src import config
# Import the models from shen_models
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel

def main():
    parser = argparse.ArgumentParser(
        description="Convert train_belief_space_policy checkpoint format to train_with_belief_rollout format."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoints (opponent_belief_model.pth and bsp_* subfolder)"
    )
    parser.add_argument(
        "--bsp_subdir",
        type=str,
        default="bsp_20250330_192915",
        help="Subdirectory inside checkpoint_dir where belief_space_policy_best.pth is located"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode number to record in the new checkpoint"
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    bsp_dir = os.path.join(checkpoint_dir, args.bsp_subdir)
    bsp_path = os.path.join(bsp_dir, "belief_space_policy_best.pth")
    opponent_belief_path = os.path.join(checkpoint_dir, "opponent_belief_model.pth")

    if not os.path.exists(bsp_path):
        print(f"Belief policy checkpoint not found at {bsp_path}")
        return
    if not os.path.exists(opponent_belief_path):
        print(f"Opponent belief model checkpoint not found at {opponent_belief_path}")
        return

    # Load the belief-space policy checkpoint
    bsp_checkpoint = torch.load(bsp_path, map_location="cpu", weights_only=False)
    # Load the opponent belief model state dict
    obm_state = torch.load(opponent_belief_path, map_location="cpu", weights_only=False)

    # Extract the hyperparameters and training info from the bsp checkpoint.
    # (These keys were saved by train_belief_space_policy.)
    epoch = bsp_checkpoint.get("epoch", 0)
    obs_dim = bsp_checkpoint["obs_dim"]
    belief_dim = bsp_checkpoint["belief_dim"]
    output_dim = bsp_checkpoint["output_dim"]
    hidden_dim = bsp_checkpoint["hidden_dim"]
    num_opponent_types = bsp_checkpoint["num_opponent_types"]

    # Instantiate BeliefSpacePolicy and load weights.
    belief_policy = BeliefSpacePolicy(
        belief_dim=belief_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    belief_policy.load_state_dict(bsp_checkpoint["model_state_dict"])

    # Instantiate a policy optimizer and load its state.
    policy_optimizer = optim.Adam(belief_policy.parameters())
    policy_optimizer.load_state_dict(bsp_checkpoint["optimizer_state_dict"])

    # Instantiate OpponentBeliefModel.
    # Use a fixed event feature dimension of 5, max sequence length from config,
    # and hidden dimension defined as config.HIDDEN_DIM // 4.
    belief_model = OpponentBeliefModel(
        event_feature_dim=5,
        max_seq_length=config.MAX_SQUENCE_LENGTH,
        hidden_dim=config.HIDDEN_DIM // 4,
        num_opponent_types=10
    )
    belief_model.load_state_dict(obm_state)

    # We don't have a saved optimizer for the belief model, so we set it to None.
    belief_optimizer = None

    # The evaluation logic expects the training agent key to be "player_0".
    training_agent = "player_0"
    # Use the provided episode number (or you can choose to use the one from bsp_checkpoint)
    episode = args.episode

    # Save the combined checkpoint in the new format.
    save_checkpoint(
        {training_agent: belief_policy},
        None,
        {training_agent: policy_optimizer},
        None,
        belief_model,
        belief_optimizer,
        episode,
        checkpoint_dir=checkpoint_dir
    )
    print(f"Combined checkpoint saved in {checkpoint_dir} at episode {episode}.")

if __name__ == "__main__":
    main()
