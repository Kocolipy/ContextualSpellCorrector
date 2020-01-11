import numpy as np
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import matplotlib.pyplot as plt
import pathlib
import json

dir = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Data")
file = dir / "fasttextvocab.txt"

### Split data into hard and easy data sets
easy = dir/ "fteasy.txt"
hard = dir/ "fthard.txt"

# with open(str(file), "r") as f:
#     easyfp = open(str(easy), "w+")
#     hardfp = open(str(hard), "w+")
#     lines = f.readlines()
#     for i, jsli in enumerate(lines):
#         line = json.loads(jsli)
#         mistake = line["mistake"]
#         label = line["label"]
#         d = damerau_levenshtein_distance(mistake, label)
#         dnorm = normalized_damerau_levenshtein_distance(mistake, label)
#         if d <= 4 and dnorm <= 0.5:
#             easyfp.write(jsli)
#         else:
#             hardfp.write(jsli)
#     easyfp.close()
#     hardfp.close()

### Plot graphs
# dist = []
# distnorm = []
# with open(str(file), "r") as f:
#     lines = f.readlines()
#     for i, jsli in enumerate(lines):
#         line = json.loads(jsli)
#         mistake = line["mistake"]
#         label = line["label"]
#         dist.append(damerau_levenshtein_distance(mistake, label))
#         distnorm.append(normalized_damerau_levenshtein_distance(mistake, label))

# plt.hist(distnorm, bins=20)
# plt.show()

# bins = np.bincount(dist)
# print(bins)
# x = np.arange(0,len(bins))
# plt.bar(x, bins)
# plt.xlim(left=0)
# plt.show()

### Evaluate performance by distance
threshold = 10
dir = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset")
file = dir / "bertvocabresults.txt"

counts = [0]*15
total = [0]*15
with open(str(file), "r") as f:
    lines = f.readlines()
    for i, jsli in enumerate(lines):
        line = json.loads(jsli)
        mistake = line["mistake"]
        label = line["label"]
        distance = damerau_levenshtein_distance(mistake, label)
        total[distance] += 1
        predictions = line["predictions"][:threshold]
        if label in predictions:
            counts[distance] += 1

print(counts)
print(total)
total = np.array(total)
counts = np.array(counts)
print(counts/total)
