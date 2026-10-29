import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("song2.mp3_features.csv")       # the only input

cols = 16
x = df["second"] % cols                      # column: the remainder
y = df["second"] // cols                     # row: full rows before it

size = df["loudness"] / df["loudness"].max() * 1500   # loudest second = biggest circle
colour = np.log2(df["pitch_hz"])                       # octaves → equal colour steps

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("black")             # black canvas
ax.scatter(x, -y, s=size, c=colour, cmap="plasma", alpha=0.8)
ax.set_aspect("equal")                       # keep circles round
ax.axis("off")                               # no axes: it's a painting, not a chart

plt.savefig("painting2.png", dpi=150, facecolor="black", bbox_inches="tight")
plt.show()