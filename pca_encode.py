import torch
import json
import numpy as np
import os
import pathlib
import utils
import pickle as pk

# Takes data set of json strings (mistake, ground truth, masked sentence)
# Obtain context embeddings using BERT
# Obtain mistake and ground truth embedding with fastText
# Dimensionality reduction of context embedding with fastText
# Dump the concatenated (context, mistake) embeddings, ground truth and original json string into files.

data_source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset")
filename = "fasttextvocab"

# Load fastText model
ft_model = utils.loadFastText()

for fold in range(1):
    fold_dir = data_source / filename / str(fold)
    benc_dir = data_source / filename / "bertencodings"
    out_dir = fold_dir / "encodings"
    out_dir.mkdir(parents=True, exist_ok=True)

    pca = pk.load(open(str(fold_dir / "pca"), 'rb'))

    with open(fold_dir / "train.dct", "r") as f:
        indi = json.load(f)
    with open(fold_dir / "val.dct", "r") as f:
        indi += json.load(f)
    with open(data_source / filename / "test_index.txt", "r") as f:
        indi += json.load(f)

    for count, doc in enumerate(indi):
        if os.path.exists(out_dir/str(doc)):
            continue
        if (count + 1) % int(len(indi)/100) == 0:
            print(str((count+1)/len(indi) * 100) + "% completed")


        with open(benc_dir/str(doc), "r") as f:
            obj = json.load(f)

        line = obj["original"]
        original = json.loads(line)
        data = pca.transform(np.array(obj["data"]).reshape(1,-1))
        sentenceEmb = torch.tensor(data.squeeze()).float()

        # Ground Truth embedding
        ground_truth = ft_model.wv[original["label"]].tolist()

        # FastText embeddings
        mistake_emb = torch.from_numpy(ft_model.wv[original['mistake']])

        input = torch.cat((sentenceEmb, mistake_emb), 0)

        # Dump input embedding, ground truth embedding and original sample to json file
        sample = {"data": input.tolist(), "label": ground_truth, "original": line}
        with open(out_dir/str(doc), "w+") as file:
            json.dump(sample, file)

