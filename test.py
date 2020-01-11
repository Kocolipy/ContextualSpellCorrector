import argparse
import torch
import pathlib
import os
import FFNN
import json
import EmbeddingDataset
from torch.utils.data import DataLoader
import utils
import sys


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print("Loading Checkpoint")
    out = pathlib.Path(r"D:\MLNLP\PCA")
    ckpt_path = out / "ckpt_0.00001_800_500_0" / "1500"

    checkpoint = torch.load(str(ckpt_path))
    hyperparams = {"hidden_size": checkpoint["hidden_size"],
                   'drop_out': checkpoint["drop_out"],
                   'batch_size': checkpoint["batch_size"]}
    current_fold = checkpoint["fold"]
    ff_model = FFNN.FFNNTwo(hyperparams["hidden_size"])
    ff_model.load_state_dict(checkpoint["model_state_dict"])
    ff_model.eval()
    ff_model.to(device)

    test_dir = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab")

    # Name of results which would be saved
    results_file = test_dir / "results.txt"

    # Load Test Data
    test_data = test_dir / "test_index.txt"
    test_encodings = test_dir / str(current_fold) / "encodings"
    val_data = EmbeddingDataset.EmbeddingDataset(test_encodings, test_data)
    dataloader = DataLoader(val_data, batch_size=hyperparams["batch_size"],
                            shuffle=True, num_workers=hyperparams["batch_size"])

    ftt = utils.loadFastText()

    # specify loss function
    criterion = torch.nn.L1Loss(reduction="sum")

    print("Beginning Evaluation ...")
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
        if count % (hyperparams["batch_size"]*50) == 0:
            print(count, "files processed.")

    # print statistics
    total_loss = val_loss / len(val_data)
    print(total_loss)

    with open(str(results_file), "w+") as f:
        f.write("Total Loss: {:.6f} \n".format(total_loss))
        for line in results:
            vals = [dict(zip(line, t)) for t in zip(*line.values())]
            for v in vals:
                f.write(json.dumps(v) + "\n")

    # Evaluate accuracy
    with open(str(results_file), "r") as f:
        f.readline() # Ignore evaluation loss
        samples = f.readlines()

        for t in [1, 3, 5, 10]:
            success, counts = utils.accuracy(samples, t)
            print("Top {0} predictions".format(t), success, counts)