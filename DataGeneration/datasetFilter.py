import os
import pathlib
import json
import utils

def removeDuplicates(in_file, out_file):
    file = open(out_file, "w+")

    with open(in_file, "r") as f:
        lines = f.readlines()

    seen = set()
    for i, jsonline in enumerate(lines):
        line = json.loads(jsonline)
        mistake = line["mistake"]
        label = line["label"]
        if (mistake, label, line["sentence"]) not in seen:
            file.write(jsonline)
        seen.add((mistake, label, line["sentence"]))
    file.close()


def removeNonFastTextLabels(in_file, out_file):
    ft_model = utils.loadFastText()

    data = []
    with open(in_file, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    edited = []
    for i in data:
        if i["label"] in ft_model.wv.index2word:
            edited.append(i)

    with open(out_file, "w+") as f:
        for line in edited:
            f.write(json.dumps(line) + "\n")


def removeNonBERTLabels(in_file, out_file):
    tokenizer, _ = utils.loadBERT()

    data = []
    with open(in_file, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    edited = []
    for i in data:
        if i["label"] in tokenizer.vocab:
            edited.append(i)

    with open(out_file, "w+") as f:
        for line in edited:
            f.write(json.dumps(line) + "\n")


source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset")
in_file = source/"collated.txt"

if not os.path.exists(source/"nodup.txt"):
    removeDuplicates(in_file, source/"noduplicates.txt")

if not os.path.exists(source/"fasttextvocab.txt"):
    removeNonFastTextLabels(in_file, source/"fasttextvocab.txt")
