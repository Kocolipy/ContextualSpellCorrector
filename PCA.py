import json
import numpy as np
import pathlib
import os
import pickle as pk
from sklearn.decomposition import PCA

cwd = pathlib.Path(os.getcwd())
data_source = cwd / "Dataset"
filename = "fasttextvocab"

for fold in range(5):
    fold_dir = data_source / filename / str(fold)
    benc_dir = data_source / filename / "contextemb"

    with open(fold_dir / "train.dct", "r") as f:
        indi = json.load(f)
    lst = []
    for i in indi:
        with open(benc_dir/str(i), "r") as f:
            obj = json.load(f)
            lst.append(np.array(obj["data"]))

    lst = np.array(lst)
    print("Finishing loading data")

    # # Plot cumulative variance
    # pca = PCA().fit(lst)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative Explained Variance');

    pca = PCA(n_components=250).fit(lst)
    pk.dump(pca, open(str(fold_dir/"pca"),"wb+"))
    print("Finishing training PCA for fold", fold)
