import re
import matplotlib.pyplot as plt

# Define file paths
train_file = "train_stats.txt"
inference_file = "inference_stats.txt"

# Regex to capture the metrics: Epoch, Loss, Accuracy, Precision, F1
pattern = re.compile(r"Epoch\s+(\d+):\s+Loss=([\d.]+),\s+Accuracy=([\d.]+),\s+Precision=([\d.]+),\s+F1=([\d.]+)")

def parse_stats(filepath):
    epochs, loss, accuracy, precision, f1 = [], [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                loss.append(float(m.group(2)))
                accuracy.append(float(m.group(3)))
                precision.append(float(m.group(4)))
                f1.append(float(m.group(5)))
    return epochs, loss, accuracy, precision, f1

# Parse the data from the training and inference log files
train_epochs, train_loss, train_acc, train_prec, train_f1 = parse_stats(train_file)
inf_epochs, inf_loss, inf_acc, inf_prec, inf_f1 = parse_stats(inference_file)

# Define a color scheme for the four metrics
metric_colors = {
    "Loss": "tab:red",
    "Accuracy": "tab:blue",
    "Precision": "tab:green",
    "F1": "tab:purple"
}

# Create a figure with a 2x4 grid of subplots; figsize=(8,4) at dpi=100 gives 800x400 pixels.
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=100)

# List of metrics and corresponding data for training and inference
metrics = [
    ("Loss", train_loss, inf_loss),
    ("Accuracy", train_acc, inf_acc),
    ("Precision", train_prec, inf_prec),
    ("F1", train_f1, inf_f1)
]

# Plot for each metric
for col, (metric, train_data, inf_data) in enumerate(metrics):
    # Top row: Training metrics with discrete markers and no connecting lines.
    ax_train = axes[0, col]
    ax_train.plot(train_epochs, train_data, marker='x', linestyle='None', 
                  label=f"Training {metric}", color=metric_colors[metric])
    ax_train.set_title(f"Training {metric}")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel(metric)
    ax_train.legend()
    ax_train.grid(True)
    
    # Bottom row: Inference metrics with discrete markers.
    ax_inf = axes[1, col]
    ax_inf.plot(inf_epochs, inf_data, marker='*', linestyle='None', 
                label=f"Inference {metric}", color=metric_colors[metric])
    ax_inf.set_title(f"Inference {metric}")
    ax_inf.set_xlabel("Epoch")
    ax_inf.set_ylabel(metric)
    ax_inf.legend()
    ax_inf.grid(True)

fig.tight_layout()
plt.show()
