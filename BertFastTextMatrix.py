import json
import pathlib
import utils

# Generate the BERT matrix
# a map of every word in the bert vocabulary to their fastText embeddings
# Used for dimensionality reduction of context embedding


# Change this line to change location of matrix file
outfile = pathlib.Path(r"D:\MLNLP\Bert_FastText.mat")

ft_model = utils.loadFastText()
tokenizer, _ = utils.loadBERT()

matrix = []
for i, word in enumerate(tokenizer.vocab):
    matrix.append(ft_model.wv[word].tolist())
    if len(matrix) % 1000 == 0:
        print(len(matrix), "loaded ...")

with open(outfile, "w+") as f:
    json.dump(matrix, f)
