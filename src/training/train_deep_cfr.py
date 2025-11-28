import torch
import torch.optim as optim
import os
import time
import argparse
from pathlib import Path

from src.model.deep_cfr_model import DeepCFRNetwork
from src.training.dcr_extras import ReservoirBuffer

from src.misc.lb import CFRManager

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Models
    adv_net = DeepCFRNetwork(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_len
    ).to(device)
    
    pol_net = DeepCFRNetwork(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_len
    ).to(device)
    
    # 2. Optimizers
    adv_opt = optim.Adam(adv_net.parameters(), lr=args.lr)
    pol_opt = optim.Adam(pol_net.parameters(), lr=args.lr)
    
    # 3. Buffers
    adv_buffer = ReservoirBuffer(args.buffer_size, args.obs_dim, args.max_seq_len, args.action_dim, device)
    pol_buffer = ReservoirBuffer(args.buffer_size, args.obs_dim, args.max_seq_len, args.action_dim, device)
    
    # 4. C++ CFR Manager
    cfr_manager = CFRManager(
        num_players=args.num_players,
        max_depth=100,
        num_workers=1024
    )
    
    # Training Loop
    for iteration in range(args.num_iterations):
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        
        # A. Traversals (Data Collection)
        # We need to save the current Advantage Net for C++ to use in traversals (for Opponent Node inference)
        # We pass state_dict directly to C++!
        cfr_manager.load_advantage_network(adv_net.state_dict())
        
        nodes_visited = cfr_manager.run_cfr_traversal(args.num_traversals)
        print(f"  Traversals completed. Nodes visited (approx): {nodes_visited}")
        
        # Collect samples
        regret_samples, strategy_samples = cfr_manager.collect_samples()
        
        # Add to buffers
        # regret_samples is a list of tensors: [obs, act, agent, pos, mask, target, weight]
        if regret_samples:
            adv_buffer.add(
                regret_samples[0], regret_samples[1], regret_samples[2], 
                regret_samples[3], regret_samples[4], regret_samples[5], 
                weight=1.0 # Weighting handled in loss or here?
            )
            
        if strategy_samples:
            # Linear weighting for strategy: t
            pol_buffer.add(
                strategy_samples[0], strategy_samples[1], strategy_samples[2], 
                strategy_samples[3], strategy_samples[4], strategy_samples[5], 
                weight=float(iteration + 1)
            )
        
        print(f"  Buffer sizes: Adv {len(adv_buffer)}, Pol {len(pol_buffer)}")
        # Loss = MSE(predicted_regret, target_regret)
        train_network(adv_net, adv_opt, adv_buffer, args.batch_size, args.train_steps_adv, "Advantage")
        
        # C. Train Policy Network
        # Loss = CrossEntropy(predicted_logits, target_strategy) or MSE
        # Deep CFR paper uses MSE on probabilities usually, but CrossEntropy on logits is fine if target is distribution.
        # If target is average strategy (distribution), MSE is standard.
        train_network(pol_net, pol_opt, pol_buffer, args.batch_size, args.train_steps_pol, "Policy")
        
        # D. Save Checkpoints
        if (iteration + 1) % args.save_interval == 0:
            torch.save(adv_net.state_dict(), f"checkpoints/adv_net_{iteration+1}.pt")
            torch.save(pol_net.state_dict(), f"checkpoints/pol_net_{iteration+1}.pt")

def train_network(net, opt, buffer, batch_size, steps, label):
    net.train()
    total_loss = 0
    for i in range(steps):
        batch = buffer.sample(batch_size)
        if batch is None:
            break
            
        obs, act, agent, pos, mask, target, weights = batch
        
        opt.zero_grad()
        output = net(obs, act, agent, pos, mask)
        
        # MSE Loss weighted by iteration
        loss = (output - target).pow(2).mean(dim=-1) * weights
        loss = loss.mean()
        
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
    
    print(f"{label} Loss: {total_loss / steps if steps > 0 else 0:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dim", type=int, default=16) # Aligned to 8 bytes in C++
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=480)
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--num_traversals", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_steps_adv", type=int, default=1000)
    parser.add_argument("--train_steps_pol", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=10)
    
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    train(args)
