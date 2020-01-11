import os
import pathlib
import re
import json
import numpy

source = pathlib.Path(r"path\to\NFER2DAT.643")
dir = pathlib.Path(os.getcwd())
outfile = pathlib.Path(dir / "NFER2.txt")

def forTraining(sentence, mistake, mistake_index):
    data = {"mistake": mistake.lower(), "label": sentence[mistake_index].lower()}
    sentence = [w.lower() for w in sentence]
    sentence[mistake_index] = "[MASK]"
    data["sentence"] = "[CLS] " + " ".join(sentence) + " [SEP]"
    return data


savefile = open(outfile, "w+")


words = '1 we, 2 will,  3 be,  4 coming,  5 to,  6 leeds,  7 on, \
8 sunday,  9 with,  10 our,  11 two, 12 sons, 13 as, 14 you, \
15 were, 16 not, 17 there, 18 last, 19 time, 20 we, 21 came, \
22 we,  23 are,  24 looking,  25 forward,  26 to, 27 seeing, \
28 you, 29 again, 30 best, 31 wishes.'

a = list(filter(lambda w: w, words.split(" ")))

dct1 = {}
for i in range(0, len(a), 2):
    num = int(a[i])
    word = a[i+1].replace(",", "").replace(".", "")
    dct1[num] = word

strings = ["We will be coming to Leeds on Sunday with our two sons", "As you were not there last time we came , we are looking forward to seeing you again", "Best Wishes",
           "My FRIENDS , Joan and Arthur live near the STATION", "They have two children , both still BABIES",
           "Yesterday I WALKED over to their house to help Arthur do some DIGGING", "I was a bit late because I had been COOKING myself HALF a pound of steak and VARIOUS vegetables , including fried POTATOES",
           "I am not used to DINING on so much , it must be ADMITTED , but I had RECEIVED a bonus that week and I wanted to celebrate", "In fact his progress was hardly NOTICEABLE"]
strings = [s.lower() for s in strings]
sentences1 = [list(filter(lambda w: w, s.split(" "))) for s in strings]
indices1 = [13, 30, 51, 53, 54, 56, 60, 63, 64]


word = 'My FRIENDS , Joan and Arthur  live  near  the  STATION . \
They  have  two  children,  both  still BABIES .  Yesterday I \
WALKED over to their house to help Arthur do  some  DIGGING . \
I  was  a  bit late because I had been COOKING myself HALF a \
pound of  steak  and  VARIOUS  vegetables,  including  fried \
POTATOES .   I  am  not used to DINING on so much, it must be \
ADMITTED , but I had RECEIVED a bonus that week and I  wanted \
to  celebrate.   Arthur  still  had not dug very much of the \
garden when I got there .  In fact his progress  was  hardly \
NOTICEABLE .'

count = 51
for i in word.split(" "):
    if i.isupper() and i != "I":
        dct1[count] =(i.lower())
        count += 1


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
                for i, limit in enumerate(indices1):
                    if k < limit:
                        s = i
                        break
                mask_index = sentences1[s].index(dct1[k])
                savefile.write(json.dumps(forTraining(sentences1[s], v, mask_index)) + "\n")
