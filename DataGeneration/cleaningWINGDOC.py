import os
import pathlib
import re
import json

source = pathlib.Path(r"C:\Users\Nicholas\Downloads\misspelling\WINGDAT.643")
outfile = pathlib.Path(r"C:\Users\Nicholas\Downloads\Training\WING.txt")

def forTraining(sentence, mistake, mistake_index, label):
    data = {"mistake": mistake.lower(), "label": label.lower()}
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "w+")
with open(source, "r") as f:
    for line in f.readlines():
        if line and "*" in line:
            line = list(filter(lambda w: w.strip() and w != "SQ" and w != "EQ", line.strip().split(" ")))
            mistake = line[0]
            label = line[1].strip("(").strip(")")

            star_index = line.index("*")
            if label[0].isupper():
                line = line[star_index:]
            else:
                line = line[2:]
            star_index = line.index("*")
            savefile.write(json.dumps(forTraining(line, mistake, star_index,  label)) + "\n")