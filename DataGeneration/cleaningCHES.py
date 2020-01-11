import os
import pathlib
import re
import json
import numpy

source = pathlib.Path(r"path\to\CHESDAT.643")
dir = pathlib.Path(os.getcwd())
outfile = pathlib.Path(dir / "CHES.txt")

def forTraining(sentence, mistake, mistake_index):
    data = {"mistake": mistake.lower(), "label": sentence[mistake_index].lower()}
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "w+")

words = 'I OFTEN VISITED my AUNT .  She lived in  a  MAGNIFICENT \
HOUSE  OPPOSITE the GALLERY .  I REMEMBER her SPLENDID PURPLE \
CURTAINS .  She WROTE POETRY .  The PROBLEM was  nobody  could \
UNDERSTAND  it.   Her  LATEST  POEMS  had words like prunty , \
slimber , grondel , blomp.  I WANTED to LAUGH  but  I  had  to \
PRETEND  to  like  them.  However , I REALLY like the SPECIAL \
REFRESHMENT .  THERE was BLUE JUICE , CAKE and BISCUITS .  When \
I left , my STOMACH was full and I was happy and CONTENTED .'

keywords = [""]
for i in words.split(" "):
    if i.isupper() and i != "I":
       keywords.append(i.lower())

words = words.lower()
sentences = words.split(".")[:-1]
sentences = [list(filter(lambda w:w, s.split(" "))) for s in sentences]

indices = [4, 8, 12,14, 16, 18, 21, 24, 29,31]

def parseLine(line):
    a = list(filter(lambda w: w, line.strip().split(" ")))[1:]
    dct = {}
    for i in range(0, len(a), 2):
        try:
            x = int(a[i])
            dct[x] = a[i+1]
        except Exception:
            continue
    return dct

with open(source, "r") as f:
    for line in f.readlines():
        maps = parseLine(line)
        if maps:
            for k, v in maps.items():
                for i, limit in enumerate(indices):
                    if k < limit:
                        s = i
                        break
                mask_index = sentences[s].index(keywords[k])
                if v == "..":
                    print("Skipping")
                    continue
                savefile.write(json.dumps(forTraining(sentences[s], v, mask_index)) + "\n")
