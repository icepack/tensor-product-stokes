import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

with open(args.input, "r") as input_file:
    data = json.load(input_file)

fig, ax = plt.subplots()
ax.set_title("Empirical inf-sup constants")
ax.set_xscale("log", base=2)
ax.set_ylim((0.0, 1.0))
ax.set_xlabel("Mesh spacing")
ax.set_ylabel("inf-sup constant")
for entry in data:
    element = entry["element"]
    hs, λs = zip(*entry["results"])
    ax.scatter(hs, λs, label=element)
ax.legend()
fig.savefig(args.output, bbox_inches="tight")
