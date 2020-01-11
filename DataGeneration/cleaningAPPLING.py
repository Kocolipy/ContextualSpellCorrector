import os
import pathlib
import re
import json

source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\misspelling\APPLING1DAT.643")
outfile = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\appling.txt")

def forTraining(sentence, mistake, mistake_index, label):
    data = {"mistake": mistake.lower(), "label": label.lower()}
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "a")
with open(source, "r") as f:
    for line in f.readlines():
        if line and line[0] != "$":
            line = list(filter(lambda w: w, line.strip().split(" ")))
            savefile.write(json.dumps(forTraining(line[2:], line[0], line.index("*")-2,  line[1])) + "\n")