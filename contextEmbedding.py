import torch
import json
import numpy as np
import os
import pathlib
import utils

# Takes data set of json strings (mistake, ground truth, masked sentence)
# Obtain context embeddings using BERT
# Obtain mistake and ground truth embedding with fastText
# Dimensionality reduction of context embedding with fastText
# Dump the concatenated (context, mistake) embeddings, ground truth and original json string into files.

cwd = pathlib.Path(os.getcwd())
data_source = cwd / "Dataset"
filename = "fasttextvocab"
source_file = data_source / (filename + ".txt")

out_dir = data_source / filename / "contextemb"
out_dir.mkdir(parents=True, exist_ok=True)

# Load pre-trained model tokenizer (vocabulary) and model (weights)
tokenizer, bert_model = utils.loadBERT()

print("Encoding Data")
with open(source_file, "r") as f:
    lines = f.readlines()
    for count, line in enumerate(lines):
        d = json.loads(line)
        if (count + 1) % int(len(lines)/100) == 0:
            print(str((count+1)/len(lines) * 100) + "% completed")

        # BERT embeddings
        # Tokenize context
        tokenized_text = tokenizer.tokenize(d['sentence'])
        masked_index = tokenized_text.index("[MASK]")
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        predictions = bert_model(tokens_tensor, segments_tensors)[0, masked_index].detach().numpy()

        # Dump input embedding, ground truth embedding and original sample to json file
        sample = {"data": predictions.tolist(), "original": line}
        with open(out_dir/ str(count), "w+") as file:
            json.dump(sample, file)