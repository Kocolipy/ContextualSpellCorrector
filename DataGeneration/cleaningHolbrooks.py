import os
import pathlib
import re
import json

source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\misspelling\holbrook.txt")
outfile = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Data\holbrook.txt")

def forTraining(sentence, mistake, label):
    data = {"mistake": mistake.lower(), "label": label.lower()}
    mistake_index = sentence.index(label)
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "a")
with open(source, "r") as f:
    lines = f.readlines()
    paras = [0]
    sentences = []
    for i, line in enumerate(lines):
        if line == "\n":
            paras.append(i)
    for i in range(len(paras) - 1):
        paragraph = " ".join([s.strip() for s in lines[paras[i]:paras[i+1]]])
        paragraph = paragraph.replace(",", " ,")
        periods = [m.start() for m in re.finditer('\.', paragraph)]
        periods = [0] + list(filter(lambda j: paragraph[max(0, j - 2):j+1] != "Mr.", periods))
        ps = []
        for index in range(len(periods)-1):
            ps.append(paragraph[periods[index]+1: periods[index+1]])
        ps.append(paragraph[periods[-1]:])
        sentences += ps

        # print(list(filter(lambda x: paragraph[x-2:x] == "Mr." for x in periods)))
    count = 0
    for line in sentences:
        if r"</ERR>" in line:
            line = line.split(" ")
            errs = list(filter(lambda x: line[x] == '<ERR', range(len(line))))
            correct = line[:errs[0]]
            mistakes = {}
            for i, e in enumerate(errs):
                truth = line[e+1].split("=")[1][:-1]
                mistake = line[e+2]
                if line[e+3] == "</ERR>" and len(truth.split("-")) == 1:
                    mistakes[truth] = mistake
                    correct.append(truth)
                    if i < len(errs)-1:
                        correct += line[e+4:errs[i+1]]
                    else:
                        correct += line[errs[-1]+4:]
                else:
                    correct = []
                    break
            if correct:
                if len(correct) > 15:
                    count += 1
                    for k, v in mistakes.items():
                        print("{\"mistake\": " + v + "\", \"label\": \"" + k + "\", \"sentence\": \"[CLS][SEP]\"}")
                    print(" ".join(correct))
                # if len(correct) <= 15:
                #     for k,v in mistakes.items():
                #         savefile.write(json.dumps(forTraining(correct, v, k)) + "\n")


        #     line = list(filter(lambda w: w, line.strip().split(" ")))
        #     print(line)
        #     print(line.index("<ERR"))
        #