from torch.utils.data import Dataset
import torchvision
import torch
import json


class EmbeddingDataset(Dataset):
    def __init__(self, encoding_source, indices):
        self.encodings = encoding_source
        with open(str(indices), "r") as f:
            self.data_seq = json.load(f)

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.data_seq[idx]
        with open(str(self.encodings/str(file)), 'r') as f:
            obj = json.load(f)

        data = torch.tensor(obj["data"])
        label = torch.tensor(obj["label"])
        line = json.loads(obj["original"])
        return data, label, line


class EmbeddingDatasetWithNorm(Dataset):
    def __init__(self, encoding_source, indices, norm):
        self.encodings = encoding_source
        with open(str(indices), "r") as f:
            self.data_seq = json.load(f)

        with open(str(norm), "r") as f:
            data = json.load(f)
        self.mean = torch.tensor(data["mean"])
        self.std = torch.tensor(data["std"])

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.data_seq[idx]
        with open(str(self.encodings/str(file)), 'r') as f:
            obj = json.load(f)
        data = torch.tensor(obj["data"])
        data = (data - self.mean)/self.std
        label = torch.tensor(obj["label"])
        line = json.loads(obj["original"])
        return data, label, line
