import torch
import pathlib
import os
import gensim
import FFNN
import json
import EmbeddingDataset
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

# Test if different approach to the problem would work (EDIT: it did not work)
# Calculate the cosine similiarities between the fastText embedding of each word of the BERT vocabulary with the mistake embedding
# Use the context embedding as weights for the cosine similarities
# Return the top 10 scoring embedding as predictions

print("Loading Bert Matrix ...")
bert_matrix_path = pathlib.Path(r"D:\MLNLP\Bert_FastText.mat")
with open(bert_matrix_path, "r") as f:
    bert_matrix = torch.tensor(json.load(f))

print("Loading FastText ...")
source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP")
ft_modelpath = source / "cc.en.300.bin"
# Load vectors directly from the file
ft_model = gensim.models.fasttext.load_facebook_model(ft_modelpath)


# Load pre-trained model tokenizer (vocabulary)
print("Loading BERT model ...")
bert_modelpath = r'bert-base-uncased'
bert_matrix_path = pathlib.Path(r"D:\MLNLP\Bert_FastText.mat")
tokenizer = BertTokenizer.from_pretrained(bert_modelpath)

# Load pre-trained model (weights)
bert_model = BertForMaskedLM.from_pretrained(bert_modelpath)
bert_model.eval()

dataset = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\bertvocab.txt")
out = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\bertvocabresults.txt")
outfile = open(out, "w")
with open(dataset, "r") as f:
    score = 0
    lines = f.readlines()
    for count, line in enumerate(lines):
        d = json.loads(line)
        if (count + 1) % int(len(lines) / 100) == 0:
            print(str((count + 1) / len(lines) * 100) + "% completed")

        ### FastText embeddings
        mistake_emb = torch.from_numpy(ft_model.wv[d['mistake']])

        similiarityscore = torch.stack([torch.nn.CosineSimilarity(dim=0)(i, mistake_emb) for i in bert_matrix])

        ### BERT embeddings
        tokenized_text = tokenizer.tokenize(d['sentence'])
        masked_index = tokenized_text.index("[MASK]")
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors z
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        predictions = bert_model(tokens_tensor, segments_tensors)[0, masked_index]

        top10 = torch.argsort(-1*similiarityscore*predictions)[:10].tolist()
        toppredictions = tokenizer.convert_ids_to_tokens(top10)

        d["predictions"] = toppredictions
        outfile.write(json.dumps(d) + "\n")