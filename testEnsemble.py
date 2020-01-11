import subprocess
import json
import pathlib
import utils

def combinePredictions(results_file, bad_speller=False):
    samples = []
    model = open(results_file, "r")
    model.readline()
    lines = model.readlines()
    for jsline in lines:
        line = json.loads(jsline)

        # Replace mask with mistake in sentence
        sentence = line["sentence"].replace("[MASK]", line["mistake"])[6:-6]

        dd_process = subprocess.Popen(['echo', sentence], stdout=subprocess.PIPE)
        if bad_speller:
            ssh_process = subprocess.Popen(['aspell', '-a', '--sug-mode=bad-spellers'], stdin=dd_process.stdout, stdout=subprocess.PIPE)
        else:
            ssh_process = subprocess.Popen(['aspell', '-a'], stdin=dd_process.stdout, stdout=subprocess.PIPE)
        dd_process.stdout.close()
        out, err = ssh_process.communicate()

        for i in out.decode("utf-8").split("\n"):
            if i and "&" == i[0]:
                ind = i.index(":")
                if i[2:ind].split(" ")[0] == line["mistake"]:
                    start = ind+2
                    suggestions = [j.lower() for j in i[start:].split(", ")[:2]]

        for i in line["prediction"]:
            if i not in suggestions:
                suggestions.append(i)
                break
        line["prediction"] = suggestions
        samples.append(json.dumps(line))
    return samples


result_file = pathlib.PureWindowsPath(r"\mnt\c\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\results_AG.txt").as_posix()
test_file = pathlib.PureWindowsPath(r"\mnt\c\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab\test_AG.txt").as_posix()

# samples = getASpellPredictions(test_file, bad_speller=False)
samples = combinePredictions(result_file, bad_speller=True)

success, counts = utils.accuracy(samples, 3)
print(success, counts)
