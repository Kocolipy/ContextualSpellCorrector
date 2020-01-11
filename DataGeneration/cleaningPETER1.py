import os
import pathlib
import re
import json
import numpy

source = pathlib.Path(r"C:\Users\Nicholas\Downloads\misspelling\PETERS1DAT.643")
senten = pathlib.Path(r"C:\Users\Nicholas\Downloads\misspelling\PETERSEN.txt")
outfile = pathlib.Path(r"C:\Users\Nicholas\Downloads\Training\PETERS1.txt")

def forTraining(sentence, mistake, mistake_index):
    data = {"mistake": mistake.lower(), "label": sentence[mistake_index].lower()}
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "w+")

def parseLine(line):
    a = list(filter(lambda w: w, line.strip().split(" ")))
    dct = {}
    for i in range(0, len(a), 2):
        try:
            x = int(a[i])
            dct[x] = a[i+1]
        except Exception:
            continue
    return dct

correct = {}
sentences = {}
masked = {}
with open(senten, "r") as f:
    for line in f.readlines():
        line = list(filter(lambda w: w, line.strip().split(" ")))
        line[-1] = line[-1][:-1]
        key = int(line[0])
        label = line[1].lower()
        sentence = [w.lower() for w in line[2:]]
        masked_index = sentence.index(label)

        correct[key] = label
        sentences[key] = sentence
        masked[key] = masked_index


with open(source, "r") as f:
    start = False
    for line in f.readlines():
        if "%" in line and "T" in line:
            start=True
        elif "$" in line or "%" in line:
            start=False

        if start:
            maps = parseLine(line)
            if maps:
                for k, v in maps.items():
                    if k > 77:
                        continue
                    if v == "-":
                        print("skipping")
                        continue
                    savefile.write(json.dumps(forTraining(sentences[k], v, masked[k])) + "\n")
