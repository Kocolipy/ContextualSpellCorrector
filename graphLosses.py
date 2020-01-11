import torch
import pathlib
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# Graph the training and validation losses against epochs
# Print the validation losses at every 50 epoch interval

source = pathlib.Path(r"D:\MLNLP\PCA")
ckpt_path = source / "ckpt_0.00001_800_500_0"
val_path = source / "val_0.00001_800_500_0"

num_ckpts=len(os.listdir(ckpt_path))
training_losses = []
val_loss = []
successrates = []
for i in range(1, num_ckpts+1):
    success = 0
    checkpoint = torch.load(str(ckpt_path / str(i)))
    training_losses.append(checkpoint["loss"])

    with open(val_path/str(i), "r") as f:
        val = float(f.readline().split(" ")[2])
        sentences = f.readlines()
        for line in sentences:
            obj = json.loads(line)
            success += 1 if obj["label"] in obj["prediction"] else 0
    successrates.append(success/float(len(sentences)))
    val_loss.append(val)
for i in range(len(val_loss)):
    if (i+1) % 50 == 0:
        print("ckpt", str(i+1), val_loss[i], successrates[i])

x = np.arange(351, num_ckpts+1)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(x, val_loss[350:], color=color, label="Loss")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(x, successrates[350:], color=color, label="Accuracy")
ax2.tick_params(axis='y', labelcolor=color)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

plt.legend(lines, labels, loc="right")
plt.title("Validation Loss and Accuracy")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()