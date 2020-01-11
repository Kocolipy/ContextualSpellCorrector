import numpy as np
import json
import os
import pathlib

for fold in range(5):
    source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\{0}".format(fold))
    encodings = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\encodings")

    with open(source/"train.dct", "r") as f:
        indices = json.load(f)

    data = []
    for i in indices:
        with open(encodings/str(i), "r") as f:
            obj = json.load(f)
            data.append(obj["data"])

    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    value = {"mean": mean.tolist(), "std": std.tolist()}

    with open(source/"norm", "w+") as f:
        json.dump(value, f)
