import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from src import config
from src.model.new_models import StrategyTransformer
from src.training.train_transformer import build_field_vocab, EventEncoder, convert_memory_to_features

# ---------------------------
# Setup device and paths
# ---------------------------
device = torch.device(config.DEVICE)
data_path = "opponent_training_data.pkl"
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")

# ---------------------------
# Load one sample from training data
# ---------------------------
with open(data_path, "rb") as f:
    training_data = pickle.load(f)
# For visualization, pick the first sample.
sample_memory, sample_label = training_data[0]

# ---------------------------
# Build categorical vocabularies (must match training)
# ---------------------------
response2idx = build_field_vocab(training_data, "response")
action2idx = build_field_vocab(training_data, "triggering_action")

# ---------------------------
# Instantiate event encoder and transformer model
# ---------------------------
event_encoder = EventEncoder(
    response_vocab_size=len(response2idx),
    action_vocab_size=len(action2idx),
    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
).to(device)

strategy_transformer = StrategyTransformer(
    num_tokens=config.STRATEGY_NUM_TOKENS,
    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
    nhead=config.STRATEGY_NHEAD,
    num_layers=config.STRATEGY_NUM_LAYERS,
    strategy_dim=config.STRATEGY_DIM,
    num_classes=config.STRATEGY_NUM_CLASSES,
    dropout=config.STRATEGY_DROPOUT,
    use_cls_token=True
).to(device)
# Override the token embedding so that continuous embeddings pass through.
strategy_transformer.token_embedding = torch.nn.Identity()

# Load checkpoint if available.
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"])
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
    print("Loaded checkpoint from", checkpoint_path)
else:
    print("Checkpoint not found. Using untrained model.")

strategy_transformer.eval()
event_encoder.eval()

# ---------------------------
# Set up hooks to capture per-head attention weights
# ---------------------------
# Dictionary to store attention weights from each transformer layer.
attention_weights = {}

def get_hook(name):
    def hook(module, inputs, output):
        # The output from self-attention is a tuple: (attn_output, attn_weights)
        # Save the per-head attention weights.
        attention_weights[name] = output[1].detach().cpu().numpy()
    return hook

# For each encoder layer, override the forward method to force no averaging.
for i, layer in enumerate(strategy_transformer.transformer_encoder.layers):
    # Save the original forward method.
    orig_forward = layer.self_attn.forward

    # Define a new forward method that forces need_weights=True and average_attn_weights=False.
    def new_forward(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return orig_forward(query, key, value, **kwargs)

    # Overwrite the forward method.
    layer.self_attn.forward = new_forward

    # Register hook on the self-attention module.
    layer.self_attn.register_forward_hook(get_hook(f"layer_{i}"))

# ---------------------------
# Process a sample through the model
# ---------------------------
# Convert the memory (list of event dictionaries) to features.
features = convert_memory_to_features(sample_memory, response2idx, action2idx)
# Create a tensor with shape (1, seq_len, 4)
input_tensor = torch.tensor(features, dtype=torch.float, device=device).unsqueeze(0)

# Use the event encoder to project raw events to token embeddings.
projected = event_encoder(input_tensor)
# Forward pass through the transformer (attention hooks will capture the weights).
with torch.no_grad():
    embedding, _ = strategy_transformer(projected)

# ---------------------------
# Visualize per-head attention weights for each layer
# ---------------------------
for name, attn in attention_weights.items():
    # attn is now expected to have shape: (batch, num_heads, seq_len, seq_len)
    if attn.ndim == 4:
        num_heads = attn.shape[1]
        for head in range(num_heads):
            head_attn = attn[0, head]  # take the first (and only) sample's attention for this head
            plt.figure(figsize=(6, 5))
            plt.imshow(head_attn, cmap="viridis")
            plt.title(f"{name} - Head {head}")
            plt.xlabel("Key positions")
            plt.ylabel("Query positions")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
    else:
        print(f"Attention for {name} has unexpected shape: {attn.shape}")
