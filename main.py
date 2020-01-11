import argparse
import torch
import pathlib
import os
import FFNN
import EmbeddingDataset
from torch.utils.data import DataLoader
import sys

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Train Feedforward Neural Network')
    # parser.add_argument('hidden_size', metavar='hidden_size', type=int, nargs='+',
    #                     help='Size of hidden layer')
    parser.add_argument('hidden_size1', metavar='hidden_size1', type=int, nargs='+',
                        help='Size of hidden layer')
    parser.add_argument('hidden_size2', metavar='hidden_size2', type=int, nargs='+',
                        help='Size of hidden layer')
    # parser.add_argument('hidden_size3', metavar='hidden_size3', type=int, nargs='+',
    #                     help='Size of hidden layer')
    parser.add_argument('--fold', dest='fold', type=int, nargs='?',
                        help='The fold number for cross validation (Default: 0)')
    parser.add_argument('--drop_out', dest='drop_out', type=float, nargs='?',
                        help='Drop out probability for training (Default: 0.5)')
    parser.add_argument('--batch_size', dest='batch_size', type=int, nargs='?',
                        help='Batch size for training  (Default: 10)')
    parser.add_argument('--ckpt', dest='ckpt', type=int, nargs='?',
                        help='Checkpoint to begin from')
    parser.add_argument('--epochs', dest='epochs', type=int, nargs='?',
                        help='Number of epochs to run for  (Default: 500)')

    args = parser.parse_args()
    hyperparams = {"hidden_size": (args.hidden_size1[0], args.hidden_size2[0]),
                   "drop_out": args.drop_out if args.drop_out else 0.5,
                   "batch_size": args.batch_size if args.batch_size else 10,
                   "n_epochs": args.epochs if args.epochs else 500}

    current_fold = args.fold if args.fold else 0
    ckpt = args.ckpt

    cwd = pathlib.Path(os.getcwd())
    # cwd = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP")
    cwd = pathlib.Path(r"D:\MLNLP\LR2")
    # ckpt_path = cwd / "ckpt_0.0001"
    ckpt_path = cwd / "ckpt_0.00001_{0}".format(current_fold)
    ckpt_path.mkdir(exist_ok=True)

    # data_path = cwd / "fasttextvocab"
    data_path = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab")
    data_encodings = data_path / "encodings"
    dct_source = data_path / str(current_fold) / "train.dct"

    train_data = EmbeddingDataset.EmbeddingDataset(data_encodings, dct_source)
    dataloader = DataLoader(train_data, batch_size=hyperparams["batch_size"],
                            shuffle=True, num_workers=hyperparams["batch_size"])
    print("Data loaded")

    ff_model = FFNN.FFNNTwo(hyperparams["hidden_size"], hyperparams["drop_out"])
    # ff_model = FFNN.FFNNThree(hyperparams["hidden_size"], hyperparams["drop_out"])

    # specify loss function
    criterion = torch.nn.L1Loss(reduction="sum")

    # specify optimizer
    optimizer = torch.optim.Adam(ff_model.parameters(), lr=0.00001)

    if ckpt:
        checkpoint = torch.load(str(ckpt_path / str(ckpt)))
        ff_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loading Checkpoint", ckpt)
    ff_model.to(device)

    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    print("Beginning training ...")
    ff_model.train()  # prep model for training

    for epoch in range(hyperparams["n_epochs"]):
        # monitor training loss
        train_loss = 0.0
        count = 0
        for data, label, _ in dataloader:
            data = data.to(device)
            label = label.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = ff_model(data)
            # calculate the loss
            loss = criterion(output, label)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            # update running training loss
            train_loss += loss.item()

            # count += hyperparams["batch_size"]
            # if count % (hyperparams["batch_size"]*100) == 0:
            #     print(count, "files processed.")
        # print training statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_data)

        if (epoch + 1 + (ckpt if ckpt else 0)) % 1 == 0:
            torch.save({
                'model_state_dict': ff_model.state_dict(),
                'loss': train_loss,
                'hidden_size': hyperparams["hidden_size"],
                'drop_out': hyperparams["drop_out"],
                'batch_size': hyperparams["batch_size"],
                'fold': current_fold,
                'optimizer_state_dict': optimizer.state_dict()
                }, str(ckpt_path / str(epoch + 1 + (ckpt if ckpt else 0))))

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch + 1 + (ckpt if ckpt else 0),
            train_loss
        ))