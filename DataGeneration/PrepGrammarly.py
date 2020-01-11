import subprocess
import os
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import json
import pathlib

# Organise test set into a format more accessible for Grammarly testing

dir = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab")
filename = "test_aspell.txt"
out_dir = dir / "grammarly"
out_dir.mkdir(exist_ok=True)

with open(dir / filename) as f:
    lines = f.readlines()
    for i, jsline in enumerate(lines):
        line = json.loads(jsline)
        distance = damerau_levenshtein_distance(line["mistake"], line["label"])
        (out_dir / str(distance)).mkdir(exist_ok=True)
        sentence = line["sentence"].replace("[MASK]", line["mistake"])[6:-6]
        with open(str(out_dir / str(distance) / (str(i) + "_" + line["label"] + ".txt")), "w+") as f:
            f.write(sentence)
