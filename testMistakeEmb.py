import os
import json
import pathlib
import utils
# Test the accuracy of fastText embedding of mistake and using cosine similarity against fastText embeddings

def generatePredictions(results_file):
    """
    Generate file containing predictions produced using fastText embeddings and cosine similarity
    """
    ft_model = utils.loadFastText()
    out_fp = open(results_file, "w+")
    with open(str(data_path / "test.txt"), "r") as f:
        lines = f.readlines()
        for l in lines:
            obj = json.loads(l)
            mistake_emb = ft_model.wv[obj["mistake"]]
            predictions = ft_model.wv.similar_by_vector(mistake_emb, topn=10, restrict_vocab=100000)
            predictions = [p for (p, v) in predictions]
            obj["prediction"] = predictions
            out_fp.write(json.dumps(obj) + "\n")
    out_fp.close()


cwd = pathlib.Path(os.getcwd())
data_path = cwd / "Dataset" / "fasttextvocab"
results_file = data_path / "results_full_mistakeemb.txt"

# Generate file if it does not exist
if not os.path.exists(results_file):
    generatePredictions(results_file)

# Evaluate results
threshold = [1, 3, 5, 10]
with open(results_file, "r") as f:
    lines = f.readlines()
    for t in threshold:
        success, counts = utils.accuracy(lines, t)
        print(t)
        print(success)
        print(counts)