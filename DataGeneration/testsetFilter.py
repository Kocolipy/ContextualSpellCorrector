import subprocess
import json
import pathlib


def filterASpell(in_file, in_index, out_file, out_index):
    # Filter out files that Aspell cannot detect mistakes
    file = open(out_file, "w+")
    file_index = open(out_index, "w+")

    indices = []
    with open(in_file, "r") as f:
        with open(in_index.as_posix(), "r") as ind:
            ori_indices = json.load(ind)
        lines = f.readlines()
        for i, jsline in enumerate(lines):
            line = json.loads(jsline)
            sentence = line["sentence"].replace("[MASK]", line["mistake"])[6:-6]
            dd_process = subprocess.Popen(['echo', sentence], stdout=subprocess.PIPE)
            ssh_process = subprocess.Popen(['aspell', 'list'], stdin=dd_process.stdout, stdout=subprocess.PIPE)
            dd_process.stdout.close()
            out, err = ssh_process.communicate()
            mistakes = [i for i in out.decode("utf-8").split("\n")[:-1]]
            if mistakes:
                if line["mistake"] in mistakes:
                    file.write(jsline)
                    indices.append(ori_indices[i])

    json.dump(indices, file_index)
    file_index.close()
    file.close()


# Note: Using a linux subsystem to run ASpell on windows
# Change the path according to platform
# Ensure that cmd line can access aspell.bin
dir = pathlib.PureWindowsPath(r"\mnt\c\Users\Nicholas\Downloads\MLNLP\Dataset\fasttextvocab")
filename = "test"
source = dir / (filename + ".txt")
source_index = dir / (filename + "_index.txt")
filtered = dir / (filename + "_aspell.txt")
filtered_index = dir / (filename + "_index_aspell.txt")

filterASpell(source.as_posix(), source_index.as_posix(),
             filtered.as_posix(), filtered_index.as_posix())

