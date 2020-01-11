import torch
import json
import numpy as np
import pathlib
import utils
import os
import matplotlib.pyplot as plt
import pickle as pk
from sklearn.decomposition import PCA

data_source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset")
filename = "fasttextvocab"

for fold in range(5):
    fold_dir = data_source / filename / str(fold)
    benc_dir = data_source / filename / "bertencodings"

    with open(fold_dir / "train.dct", "r") as f:
        indi = json.load(f)
    lst = []
    for i in indi:
        with open(benc_dir/str(i), "r") as f:
            obj = json.load(f)
            lst.append(np.array(obj["data"]))

    lst = np.array(lst)
    print(lst.shape)
    print("Finishing loading")

    pca = PCA(n_components=250).fit(lst)
    pk.dump(pca, open(str(fold_dir/"pca"),"wb+"))