import os
import json
import utils
# Test the accuracy of fastText embedding of mistake and using cosine similarity against fastText embeddings

def generatePredictions():
    """
    Generate file containing predictions produced using fastText embeddings and cosine similarity
    """
    ft_model = utils.loadFastText()
    out_fp = open(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\purefastText.txt", "w+")
    with open(r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\test.txt", "r") as f:
        lines = f.readlines()
        for l in lines:
            obj = json.loads(l)
            mistake_emb = ft_model.wv[obj["mistake"]]
            predictions = ft_model.wv.similar_by_vector(mistake_emb, topn=10, restrict_vocab=100000)
            predictions = [p for (p, v) in predictions]
            obj["prediction"] = predictions
            out_fp.write(json.dumps(obj) + "\n")
    out_fp.close()

threshold = [1, 3, 5, 10]
results_file = r"C:\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\purefastText.txt"

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