import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
from src.model.new_models import StrategyTransformer
from src.training.train_vs_hardcoded import convert_memory_to_tokens, vocab
from src import config

# Load transformer model
device = torch.device(config.DEVICE)
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

# Load Transformer Checkpoint
transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
if os.path.exists(transformer_checkpoint_path):
    strategy_transformer.load_state_dict(torch.load(transformer_checkpoint_path, map_location=device))
    print(f"Loaded transformer from {transformer_checkpoint_path}")
else:
    print("No transformer checkpoint found. Using untrained model.")

strategy_transformer.eval()

# Load Training Data
data_path = ("opponent_training_data2.pkl")
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found!")
    exit()

with open(data_path, "rb") as f:
    training_data = pickle.load(f)

print(f"Total Samples Loaded: {len(training_data)}")

# Extract Bot Labels
bot_counts = defaultdict(int)
for _, bot_type in training_data:
    bot_counts[bot_type] += 1

print("\nHardcoded Bot Distribution in Training Data:")
for bot, count in bot_counts.items():
    print(f"{bot}: {count} samples")

# Assign Colors to Each Bot Type
bot_types = list(bot_counts.keys())
colors = plt.cm.get_cmap("tab10", len(bot_types))  # Generate a colormap with unique colors

bot_color_map = {bot: colors(i) for i, bot in enumerate(bot_types)}

# Extract Features & Labels
opponent_embeddings = []
labels = []
label_colors = []

for i, (features, bot_type) in enumerate(training_data):
    token_seq = convert_memory_to_tokens(features, vocab)

    if len(token_seq) > 0:
        token_tensor = torch.tensor(token_seq, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            embedding, _ = strategy_transformer(token_tensor)
        opponent_embeddings.append(embedding.cpu().numpy().flatten())
        labels.append(f"Sample {i} - {bot_type}")
        label_colors.append(bot_color_map[bot_type])  # Assign color based on bot type

# Convert to numpy array
embeddings_array = np.array(opponent_embeddings)

if len(embeddings_array) < 2:
    print("Not enough samples to visualize. Need at least 2 opponent embeddings.")
    exit()

# Reduce dimensions with PCA first
pca_components = min(10, len(embeddings_array), embeddings_array.shape[1])
pca = PCA(n_components=pca_components)
pca_embeddings = pca.fit_transform(embeddings_array)

# Reduce further with t-SNE
perplexity = min(5, len(pca_embeddings) - 1)
reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
embedded_2d = reducer.fit_transform(pca_embeddings)

# Plot embeddings grouped by bot type
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    plt.scatter(embedded_2d[i, 0], embedded_2d[i, 1], color=label_colors[i], label=label if i < 10 else None)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Transformer Strategy Embeddings (Colored by Hardcoded Bot Type)")
plt.legend(loc="best", fontsize=8)
plt.grid()
plt.show()
