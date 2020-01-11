import os
import pathlib
import re
import json

source = pathlib.Path(r"path\to\CSpell Training\GoldStd-NonWord")
corrected = pathlib.Path(r"path\to\CSpell Training\GoldStd-RealWord")
dir = pathlib.Path(os.getcwd())
outfile = pathlib.Path(dir / "CSpell.txt")

def cleanData(fp):
    clean = []
    data = " ".join([line.strip() for line in f.readlines()])
    data = re.split('[.!?]', data)
    for d in data:
        if d:
            clean.append(d.strip())
    return clean


def forTraining(sentence, mistake, mistake_index):
    data = {"mistake": mistake.lower(), "label": sentence[mistake_index].lower()}
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "a")
for i in os.listdir(source):

    with open(source/i, "r") as f:
        original = cleanData(f)

    with open(corrected / i, "r") as f:
        correct = cleanData(f)

    for j in range(len(original)):
        x = list(filter(lambda w: w, original[j].split(" ")))
        y = list(filter(lambda w: w, correct[j].split(" ")))

        if (len(x) == len(y)):
            errors = []
            for k in range(len(x)):
                if x[k] != y[k]:
                    errors.append(k)
            if errors:
                for e in errors:
                    savefile.write(json.dumps(forTraining(y, x[e], e)) + "\n")

