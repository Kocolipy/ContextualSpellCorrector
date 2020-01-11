import subprocess
import json
import pathlib
import utils
import os


def getASpellPredictions(test_file, bad_speller=False):
    samples = []
    with open(test_file) as f:
        lines = f.readlines()
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
                        suggestions = [j.lower() for j in i[start:].split(", ")[:100]]
                        line["prediction"] = suggestions
            samples.append(json.dumps(line))
    return samples


cwd = pathlib.Path(os.getcwd())
data_path = cwd / "Dataset" / "fasttextvocab"
test_file = (data_path / "test_AG.txt").as_posix()
threshold = 20

# samples = getASpellPredictions(test_file, bad_speller=False)
samples = getASpellPredictions(test_file, bad_speller=True)

success, counts = utils.accuracy(samples, threshold)
print("Successful predictions")
for k, v in sorted(success.items()):
    print("Distance {0}:".format(k), v)
print("Total number")
for k, v in sorted(counts.items()):
    print("Distance {0}:".format(k), v)
