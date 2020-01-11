import os
import pathlib
import re
import json

source = pathlib.Path(r"path\to\PETERS2DAT.643")
dir = pathlib.Path(os.getcwd())
outfile = pathlib.Path(dir / "PETER2.txt")

### Contains multiple errors in sentences
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
            line = list(filter(lambda w: w, re.split("[(),[\ ]", line.strip())))
            if "_" not in line[0] and "_" not in line[1]:
                savefile.write(json.dumps(forTraining(line[2:], line[0], line.index("*")-2,  line[1])) + "\n")