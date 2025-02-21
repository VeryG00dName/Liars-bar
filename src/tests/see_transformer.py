# src/tests/see_transformer.py
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict, Counter
from random import shuffle

# Import the transformer model and configuration.
from src.model.new_models import StrategyTransformer
from src import config

# Import helper functions and the EventEncoder from the training module.
from src.training.train_transformer import build_field_vocab, EventEncoder

# Set device.
device = torch.device(config.DEVICE)

# Toggle extra debug prints.
debug_mode = False

# ---------------------------
# Instantiate the Transformer Model.
# ---------------------------
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

# Override token_embedding so the transformer receives the projected embeddings.
strategy_transformer.token_embedding = torch.nn.Identity()

# ---------------------------
# Load and balance training data.
# (This is the same file used during training.)
# ---------------------------
data_path = "opponent_training_data.pkl"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found!")
    exit()

with open(data_path, "rb") as f:
    training_data = pickle.load(f)

def balance_training_data(data):
    label_counts = Counter(label for _, label in data)
    min_count = min(label_counts.values())
    balanced = []
    for label in label_counts:
        samples = [sample for sample in data if sample[1] == label]
        shuffle(samples)
        balanced.extend(samples[:min_count])
    return balanced

training_data = balance_training_data(training_data)
print(f"Total Samples Loaded (after balancing): {len(training_data)}")

# Print the distribution of bot labels.
bot_counts = defaultdict(int)
for _, bot_type in training_data:
    bot_counts[bot_type] += 1
print("\nHardcoded Bot Distribution in Training Data:")
for bot, count in bot_counts.items():
    print(f"{bot}: {count} samples")

# ---------------------------
# Build vocabularies for the categorical fields.
# (These should match those used during training.)
# ---------------------------
response2idx = build_field_vocab(training_data, "response")
action2idx = build_field_vocab(training_data, "triggering_action")

if debug_mode:
    print("\nDEBUG: Response vocabulary:")
    for key, val in response2idx.items():
        print(f"  {key}: {val}")
    print("\nDEBUG: Action vocabulary:")
    for key, val in action2idx.items():
        print(f"  {key}: {val}")

# ---------------------------
# Instantiate the Event Encoder.
# ---------------------------
event_encoder = EventEncoder(
    response_vocab_size=len(response2idx),
    action_vocab_size=len(action2idx),
    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
).to(device)

# ---------------------------
# Load the checkpoint for the transformer, event encoder, label mapping, and categorical mappings.
# ---------------------------
transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
if os.path.exists(transformer_checkpoint_path):
    checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
    
    # Load model states.
    strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=True)
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
    print(f"Loaded transformer and event encoder from {transformer_checkpoint_path}")
    
    # Load label mapping; if not present, rebuild from training data.
    if "label_mapping" in checkpoint:
        label_mapping = checkpoint["label_mapping"]
        label2idx = label_mapping["label2idx"]
        idx2label = label_mapping["idx2label"]
        print("Loaded label mapping from checkpoint.")
    else:
        unique_labels = sorted(list({bot_type for _, bot_type in training_data}))
        label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx2label = {idx: label for label, idx in label2idx.items()}
        print('failed to load label mapping from checkpoint')
        print("Rebuilt label mapping from training data (sorted order).")
    
    # Load the categorical mappings; if not present, use the ones built earlier.
    if "response2idx" in checkpoint and "action2idx" in checkpoint:
        response2idx = checkpoint["response2idx"]
        action2idx = checkpoint["action2idx"]
        print("Loaded response and action mappings from checkpoint.")
    else:
        print('failed to load response and action mappings from checkpoint')
        print("Using rebuilt response and action mappings from training data.")
else:
    print("No transformer checkpoint found. Using untrained model.")
    unique_labels = sorted(list({bot_type for _, bot_type in training_data}))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}

# Set models to evaluation mode.
strategy_transformer.eval()
event_encoder.eval()

print("\nMapping from Bot Label to Class Index (Evaluation):")
for label, idx in label2idx.items():
    print(f"{label}: {idx}")

# ---------------------------
# Define a function to convert memory events to 4D feature vectors.
# Each event should be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
# ---------------------------
def convert_memory_to_features(memory, response_mapping, action_mapping):
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        # Map the categorical features using the provided mappings.
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

if debug_mode:
    print("\nDEBUG: First sample of raw training data:")
    sample_events, sample_label = training_data[0]
    print("Raw events:", sample_events)
    print("Label:", sample_label)
    sample_features = convert_memory_to_features(sample_events, response2idx, action2idx)
    print("Converted features:", sample_features)

# ---------------------------
# For visualization, assign a color to each bot type.
# ---------------------------
bot_types = list(bot_counts.keys())
colors = plt.colormaps["tab20"](np.linspace(0, 1, len(bot_types)))
bot_color_map = {bot: colors[i] for i, bot in enumerate(bot_types)}

# ---------------------------
# Process each training sample:
# - Convert events to features.
# - Pass through the event encoder.
# - Feed to the transformer to get embeddings and predictions.
# Also collect data for visualization and debugging.
# ---------------------------
opponent_embeddings = []
labels_for_plot = []     # For t-SNE plot labels
label_colors = []        # For plot colors
predicted_classes = []   # Numeric predictions from the model
ground_truth_labels = [] # Actual bot type strings

# For extra debugging, collect raw feature statistics and per-sample debug info.
all_features = []
sample_debug_info = []  # Store debug info for the first few samples

for i, (features, bot_type) in enumerate(training_data):
    try:
        features_cont = convert_memory_to_features(features, response2idx, action2idx)
    except Exception as e:
        print(f"Error converting sample {i}: {e}")
        continue
    if len(features_cont) > 0:
        all_features.extend(features_cont)
        feature_tensor = torch.tensor(features_cont, dtype=torch.float, device=device).unsqueeze(0)
        
        # Collect debugging info for the first few samples.
        if debug_mode and i < 5:
            sample_debug = {}
            sample_debug["original_features"] = features_cont
            sample_debug["tensor_shape"] = feature_tensor.shape
        
        projected = event_encoder(feature_tensor)
        if debug_mode and i < 5:
            sample_debug["projected_output"] = projected.detach().cpu().numpy()
            # Also record summary stats of the projected outputs.
            sample_debug["projected_mean"] = projected.mean().item()
            sample_debug["projected_std"] = projected.std().item()
        
        with torch.no_grad():
            embedding, logits = strategy_transformer(projected)
        if debug_mode and i < 5:
            sample_debug["transformer_embedding"] = embedding.detach().cpu().numpy()
            sample_debug["logits"] = logits.detach().cpu().numpy()
            # Compute softmax probabilities.
            softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
            sample_debug["softmax"] = softmax_probs.detach().cpu().numpy()
            sample_debug_info.append(sample_debug)
        
        opponent_embeddings.append(embedding.cpu().numpy().flatten())
        labels_for_plot.append(f"Sample {i} - {bot_type}")
        label_colors.append(bot_color_map[bot_type])
        pred_class = logits.argmax(dim=1).item()
        predicted_classes.append(pred_class)
        ground_truth_labels.append(bot_type)

if debug_mode:
    print("\nDEBUG: Detailed info for first 5 samples:")
    for idx, info in enumerate(sample_debug_info):
        print(f"\nSample {idx}:")
        print("Original Features:", info["original_features"])
        print("Tensor Shape:", info["tensor_shape"])
        print("Projected Output (first row):", info["projected_output"][0])
        print("Projected Mean:", info["projected_mean"], "Std:", info["projected_std"])
        print("Transformer Embedding (flattened, first 10 values):", info["transformer_embedding"].flatten()[:10], "...")
        print("Logits:", info["logits"])
        print("Softmax Probabilities:", info["softmax"])

# ---------------------------
# Print predicted class distribution.
# ---------------------------
print("\nDistribution of Predicted Classes:")
pred_dist = Counter(predicted_classes)
for cls, count in pred_dist.items():
    print(f"Class {cls}: {count} samples")

# ---------------------------
# Compute per-label and overall classification accuracy.
# ---------------------------
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
skipped_labels = set()

for gt_label, pred in zip(ground_truth_labels, predicted_classes):
    if gt_label not in label2idx:
        skipped_labels.add(gt_label)
        #print(f"Warning: Ground truth label '{gt_label}' not found in label2idx mapping. Skipping sample for accuracy computation.")
        continue
    total_counts[gt_label] += 1
    if pred == label2idx[gt_label]:
        correct_counts[gt_label] += 1

print("\nTransformer Classification Accuracy per Label:")
for label in sorted(label2idx, key=label2idx.get):
    total = total_counts[label]
    correct = correct_counts[label]
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"{label}: {accuracy:.2f}% ({correct}/{total})")

overall_correct = sum(correct_counts.values())
overall_total = sum(total_counts.values())
overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0.0
print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")

# ---------------------------
# Debug: Print raw input feature statistics.
# ---------------------------
all_features = np.array(all_features)
print("\nInput Feature Statistics (columns: response_idx, action_idx, penalties, card_count):")
print("Min:", np.min(all_features, axis=0))
print("Max:", np.max(all_features, axis=0))
print("Mean:", np.mean(all_features, axis=0))

# ---------------------------
# Visualization: Reduce dimensions with PCA and t-SNE, then plot.
# ---------------------------
if len(opponent_embeddings) < 2:
    print("Not enough samples to visualize. Need at least 2 opponent embeddings.")
    exit()

embeddings_array = np.array(opponent_embeddings)
pca_components = min(10, embeddings_array.shape[1])
pca = PCA(n_components=pca_components)
pca_embeddings = pca.fit_transform(embeddings_array)

perplexity = min(5, len(pca_embeddings) - 1)
reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
embedded_2d = reducer.fit_transform(pca_embeddings)

plt.figure(figsize=(10, 6))
# Create a dictionary to collect one scatter handle per unique ground truth label.
legend_handles = {}
for i, gt_label in enumerate(ground_truth_labels):
    x, y = embedded_2d[i, 0], embedded_2d[i, 1]
    # If this label hasn't been added to the legend yet, store its scatter handle.
    if gt_label not in legend_handles:
        scatter_handle = plt.scatter(x, y, color=label_colors[i], label=gt_label)
        legend_handles[gt_label] = scatter_handle
    else:
        plt.scatter(x, y, color=label_colors[i])
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Transformer Strategy Embeddings (Colored by Bot Type)")
plt.legend(
    handles=list(legend_handles.values()),
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),  # Shift the legend outside the plot
    fontsize=8
)
plt.tight_layout()  # Helps ensure nothing is clipped
plt.grid()
plt.show()
