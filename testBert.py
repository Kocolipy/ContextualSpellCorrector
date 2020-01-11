import os
import json
import numpy as np
import pathlib
import utils
# Test the accuracy of reduced sentence embedding and using cosine similarity against fastText embeddings

def generatePredictions():
    """
    Generate file containing predictions produced using fastText embeddings and cosine similarity
    """
    ft_model = utils.loadFastText()

    out_fp = open(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\pureBert.txt", "w+")
    encodings = pathlib.Path(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\encodings")
    with open(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\test_index.txt", "r") as f:
        indices = json.load(f)

    for ind in indices:
        with open(encodings/str(ind)) as f:
            obj = json.load(f)
        original = json.loads(obj["original"])
        sentence_emb = np.array(obj["data"][:300])

        predictions = ft_model.wv.similar_by_vector(sentence_emb, topn=10, restrict_vocab=100000)
        predictions = [p for (p, v) in predictions]

        original["prediction"] = predictions
        out_fp.write(json.dumps(original) + "\n")
    out_fp.close()

threshold = [1, 3, 5, 10]
results_file = r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\pureBert.txt"

# Generate file if it does not exist
if not os.path.exists(results_file):
    generatePredictions()

# Evaluate results
with open(results_file, "r") as f:
    lines = f.readlines()
    for t in threshold:
        success, counts = utils.accuracy(lines, t)
        print(t)
        print(success)
        print(counts)