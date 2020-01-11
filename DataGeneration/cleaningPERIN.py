import os
import pathlib
import re
import json
import numpy

source = pathlib.Path(r"C:\Users\Nicholas\Downloads\misspelling\PERIN1DAT.643")
outfile = pathlib.Path(r"C:\Users\Nicholas\Downloads\Training\PERIN.txt")

def forTraining(sentence, mistake, mistake_index):
    data = {"mistake": mistake.lower(), "label": sentence[mistake_index].lower()}
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "w+")

words = 'If you are aged 16-19 and unemployed you should take advantage of the special training schemes run by the government for unemployed young people. Enquire at your local Jobcentre about the different schemes available. You can choose to work for an employer on the spot to get experience of a particular type of job or you can work on a special project. Or you may prefer to work in Community Industry. There are also courses run to help you choose which kind of work suits you best and courses to train you for a particular job at operator or semi-skilled level.'
sentences = words.split(".")[:-1]
sentences = [list(filter(lambda w:w, s.split(" "))) for s in sentences]
dct = {}
for i, w in enumerate(list(filter(lambda w: w, " ".join(words.split(".")[:-1]).split(" ")))):
    if i < 4:
        dct[i+1] = w
    elif i == 4:
        continue
    else:
        dct[i] = w

indices = [len(s) for s in sentences]
indices = numpy.cumsum(indices)


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
                mask_index = sentences[s].index(dct[k])
                savefile.write(json.dumps(forTraining(sentences[s], v, mask_index)) + "\n")
