import argparse
import torch
import pathlib
import FFNN
import json
import EmbeddingDataset
from torch.utils.data import DataLoader
import os
import utils

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Train Feedforward Neural Network')
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
                        help='Number of epochs to run for  (Default: 1000)')

    args = parser.parse_args()
    hyperparams = {"hidden_size": (args.hidden_size1[0],args.hidden_size2[0]),
                   "drop_out": args.drop_out if args.drop_out else 0.5,
                   "batch_size": args.batch_size if args.batch_size else 10,
                   "n_epochs": args.epochs if args.epochs else 2500}

    current_fold = args.fold if args.fold else 0
    ckpt = args.ckpt

    ### Data source
    cwd = pathlib.Path(os.getcwd())
    data_path = cwd / "Dataset"/ "fasttextvocab"
    data_encodings = data_path / str(current_fold) / "encodings"
    dct_source = data_path / str(current_fold) / "train.dct"
    val_source = data_path / str(current_fold) / "val.dct"

    train_data = EmbeddingDataset.EmbeddingDataset(data_encodings, dct_source)
    dataloader = DataLoader(train_data, batch_size=hyperparams["batch_size"],
                            shuffle=True, num_workers=hyperparams["batch_size"])

    val_data = EmbeddingDataset.EmbeddingDataset(data_encodings, val_source)
    val_dataloader = DataLoader(val_data, batch_size=hyperparams["batch_size"],
                            shuffle=True, num_workers=hyperparams["batch_size"])
    print("Data loaded")

    save_path = data_path / str(current_fold)

    # Location to save checkpoints
    ckpt_path = save_path / "ckpt"
    ckpt_path.mkdir(exist_ok=True)

    # Location to save validation results
    val_path = save_path / "val"
    val_path.mkdir(exist_ok=True)

    ff_model = FFNN.FFNNTwo(hyperparams["hidden_size"], hyperparams["drop_out"])

    # Load FastText model
    ftt = utils.loadFastText()

    # specify loss function
    criterion = torch.nn.L1Loss(reduction="sum")

    optimizer = torch.optim.Adam(ff_model.parameters(), lr=0.00001, weight_decay=0.01)

    # if checkpoint is stated, load from checkpoint
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

    for epoch in range(hyperparams["n_epochs"]):
        ff_model.train()  # prep model for training

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

        ff_model.eval()  # prep model for evaluation
        val_loss = 0.0
        count = 0
        results = []
        for data, label, line in val_dataloader:
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

        # print validation statistics
        total_loss = val_loss / len(val_data)
        print("Validation loss:", total_loss)

        # Save validation data to file
        with open(str(val_path / str(epoch + 1 + (ckpt if ckpt else 0))), "w+") as f:
            f.write("Total Loss: {:.6f} \n".format(total_loss))
            for line in results:
                vals = [dict(zip(line, t)) for t in zip(*line.values())]
                for v in vals:
                    f.write(json.dumps(v) + "\n")

