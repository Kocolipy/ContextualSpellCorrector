import os
import pathlib

cwd = pathlib.Path(os.getcwd())

def loadFastText():
    import gensim

    ### Load FastText model
    print("Loading FastText ...")
    ft_modelpath = str(cwd / "cc.en.300.bin")
    # Load vectors directly from the file
    return gensim.models.fasttext.load_facebook_model(ft_modelpath)


def loadBERT():
    from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

    print("Loading BERT ...")
    bert_modelpath = r'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_modelpath)
    model = BertForMaskedLM.from_pretrained(bert_modelpath)
    model.eval()
    return tokenizer, model


def accuracy(samples, threshold):
    """
    samples: list of json strings (mistake, predictions and ground truth)
    Threshold : the number of top X predictions to use

    Evaluate the accuracy of the predictions based on Edit Distance
    """
    from collections import defaultdict
    import json
    from pyxdameraulevenshtein import damerau_levenshtein_distance

    success = defaultdict(int)
    count = defaultdict(int)
    for l in samples:
        obj = json.loads(l)
        distance = damerau_levenshtein_distance(obj["mistake"], obj["label"])
        count[distance] += 1
        if obj["label"] in obj["prediction"][:threshold]:
            success[distance] += 1
    return success, count
