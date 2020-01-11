import torch
import json
import numpy as np
import pathlib
import utils

# Takes data set of json strings (mistake, ground truth, masked sentence)
# Obtain context embeddings using BERT
# Obtain mistake and ground truth embedding with fastText
# Dimensionality reduction of context embedding with fastText
# Dump the concatenated (context, mistake) embeddings, ground truth and original json string into files.

data_source = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset")
filename = "fasttextvocab"
source_file = data_source / (filename + ".txt")
out_dir = data_source / filename / "encodings"
out_dir.mkdir(parents=True, exist_ok=True)

# Load fastText model
ft_model = utils.loadFastText()

# Load pre-trained model tokenizer (vocabulary) and model (weights)
tokenizer, bert_model = utils.loadBERT()

# Load bert matrix (bert vocabulary mapped to fastText embeddings)
bert_matrix = utils.loadBERTMatrix()

print("Encoding Data")
with open(source_file, "r") as f:
    lines = f.readlines()
    for count, line in enumerate(lines):
        d = json.loads(line)
        if (count + 1) % int(len(lines)/100) == 0:
            print(str((count+1)/len(lines) * 100) + "% completed")

        # Ground Truth embedding
        ground_truth = ft_model.wv[d["label"]].tolist()

        # FastText embeddings
        mistake_emb = torch.from_numpy(ft_model.wv[d['mistake']])

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
        predictions = bert_model(tokens_tensor, segments_tensors)[0, masked_index]
        probs = torch.nn.functional.softmax(predictions).detach().numpy()

        # Use top 10000 values (out of 30522) to speed up process
        topindices = np.argsort(-1 * probs)[:10000]
        topprobs = probs[topindices]
        topembs = bert_matrix[topindices]

        # Weighted summation of fastText embeddings (contextual embedding)
        sentenceEmb = torch.stack([topembs[i] * topprobs[i] for i in range(len(topembs))])
        sentenceEmb = torch.sum(sentenceEmb, dim=0).squeeze()

        # Concatenate context and mistake embeddings
        input = torch.cat((sentenceEmb, mistake_emb), 0)

        # Dump input embedding, ground truth embedding and original sample to json file
        sample = {"data": input.tolist(), "label": ground_truth, "original": line}
        with open(out_dir/ str(count), "w+") as file:
            json.dump(sample, file)