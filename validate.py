import argparse
import torch
import pathlib
import os
import gensim
import FFNN
import json
import EmbeddingDataset
from torch.utils.data import DataLoader
import sys

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print("Loading Data")
    cwd = pathlib.Path(os.getcwd())
    # out = cwd
    out = pathlib.Path(r"D:\MLNLP\LR2")
    ckpt_path = out / "ckpt_0.00001_2"
    ckpts = os.listdir(str(ckpt_path))

    val_path = out / "val_0.00001_2"
    val_path.mkdir(exist_ok=True)

    checkpoint = torch.load(str(ckpt_path / str(ckpts[0])))

    hyperparams = {"hidden_size": checkpoint["hidden_size"],
                   'drop_out': checkpoint["drop_out"],
                   'batch_size': checkpoint["batch_size"]}
    current_fold = checkpoint["fold"]

    # source = cwd
    source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab")
    data_source = source / "encodings"
    dct_source = source / str(current_fold) / "val.dct"

    val_data = EmbeddingDataset.EmbeddingDataset(data_source, dct_source)
    dataloader = DataLoader(val_data, batch_size=hyperparams["batch_size"],
                            shuffle=True, num_workers=hyperparams["batch_size"])

    print("Loading FastText ...")
    ft_modelpath = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP") / "cc.en.300.bin"
    # Load vectors directly from the file
    ftt = gensim.models.fasttext.load_facebook_model(str(ft_modelpath))

    for ckpt in ckpts:
        if os.path.exists(str(val_path/ str(ckpt))):
            continue
        checkpoint = torch.load(str(ckpt_path / str(ckpt)))

        # ff_model = FFNN.FFNN(hyperparams["hidden_size"])
        ff_model = FFNN.FFNNTwo(hyperparams["hidden_size"])
        ff_model.load_state_dict(checkpoint["model_state_dict"])
        print("Loading Checkpoint", ckpt)
        ff_model.eval()
        ff_model.to(device)

        # specify loss function
        criterion = torch.nn.L1Loss(reduction="sum")

        print("Beginning Validation ...")

        val_loss = 0.0
        success = 0
        count = 0
        results = []
        for data, label, line in dataloader:
            data = data.to(device)
            label = label.to(device)

            output = ff_model(data)
            loss = criterion(output, label)
            val_loss += loss.item()

            output = output.cpu().detach().numpy()

            predictions = [ftt.wv.similar_by_vector(o, topn=10, restrict_vocab=100000) for o in output]
            predictions = [[p for (p, v) in single] for single in predictions]

            line["prediction"] = predictions
            results.append(line)
            count += hyperparams["batch_size"]
            # if count % (hyperparams["batch_size"]*50) == 0:
            #     print(count, "files processed.")

        # print validation statistics
        total_loss = val_loss / len(val_data)
        print(total_loss)

        # Save validation data to file
        filename = "{0}".format(ckpt)
        with open(str(val_path / filename), "a+") as f:
            f.write("Total Loss: {:.6f} \n".format(total_loss))
            for line in results:
                vals = [dict(zip(line, t)) for t in zip(*line.values())]
                for v in vals:
                    f.write(json.dumps(v) + "\n")