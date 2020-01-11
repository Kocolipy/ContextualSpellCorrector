import numpy as np
from collections import defaultdict
from pyxdameraulevenshtein import damerau_levenshtein_distance
import os
import pathlib
import json


dir = pathlib.Path(os.getcwd()).parent / "Dataset"
file = dir / "fasttextvocab.txt"

# Evaluate the edit distance distribution of the data set
count = defaultdict(int)
with open(str(file), "r") as f:
    lines = f.readlines()
    for i, jsli in enumerate(lines):
        line = json.loads(jsli)
        mistake = line["mistake"]
        label = line["label"]
        count[damerau_levenshtein_distance(mistake, label)] += 1

for k,v in sorted(count.items()):
    print("Distance of {0}:".format(k), v/len(lines)*100)