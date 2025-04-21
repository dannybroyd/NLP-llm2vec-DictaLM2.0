import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("/vol/joberant_nobck/data/NLP_368307701_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/output/mntp_simcse/dictalm2.0-instruct/checkpoint-1000/trainer_state.json") as f:
    data = json.load(f)

log_history = data["log_history"]

# Extract metrics
steps = []
train_loss = []
lr = []
grad_norm = []

for entry in log_history:
    step = entry.get("step")
    if "loss" in entry:
        train_loss.append((step, entry["loss"]))
    if "learning_rate" in entry:
        lr.append((step, entry["learning_rate"]))
    if "grad_norm" in entry:
        grad_norm.append((step, entry["grad_norm"]))

# Unzip extra metrics
train_steps, train_losses = zip(*train_loss) if train_loss else ([], [])
lr_steps, lrs = zip(*lr) if lr else ([], [])
grad_steps, grads = zip(*grad_norm) if grad_norm else ([], [])

# Plotting
plt.figure(figsize=(14, 6))

# Train Loss Only
plt.subplot(1, 3, 1)
plt.plot(train_steps, train_losses, marker='x', linestyle='--', label="Train Loss")
plt.title("Train Loss over Steps")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Learning Rate
if lr:
    plt.subplot(1, 3, 2)
    plt.plot(lr_steps, lrs, marker='o', color='purple')
    plt.title("Learning Rate over Steps")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True)

# Gradient Norm
if grad_norm:
    plt.subplot(1, 3, 3)
    plt.plot(grad_steps, grads, marker='o', color='orange')
    plt.title("Gradient Norm over Steps")
    plt.xlabel("Step")
    plt.ylabel("Grad Norm")
    plt.grid(True)

plt.tight_layout()
plt.savefig("training_metrics_simcse.png")
