import os
import pathlib
import json
import random

# Take a dataset, split it into training and test sets (9:1 split)
# Training and Test sets contain both original json strings as well as indices
# Training set is further split into 5 folds and organised into folders for cross validation
# Folds only contain indices

cwd = pathlib.Path(os.getcwd())
data_dir = cwd / "Dataset"

# Data set to be splitted
filename = "fasttextvocab"
dataset = data_dir / "{0}.txt".format(filename)

# Location to save training and test sets
out_dir = data_dir / filename
out_dir.mkdir(exist_ok=True)

### Shuffle dataset into training and test (9:1 split)
# Read the data set
data = []
with open(dataset, "r") as f:
    for line in f.readlines():
        data.append(line)

# Shuffle the data
index = [i for i in range(len(data))]
random.shuffle(index)

# Split the data
cutoff = int(len(data)/10)
test = index[:cutoff]
train = index[cutoff:]

# Save the test set
with open(out_dir / "test.txt", "w+") as f:
    for i in test:
        f.write(data[i])

# Save the indices of the test set
with open(out_dir / "test_index.txt", "w+") as f:
    json.dump(test, f)

# Save the training set
with open(out_dir / "train.txt", "w+") as f:
    for i in train:
        f.write(data[i])

# Save the indices of the training set
with open(out_dir / "train_index.txt", "w+") as f:
    json.dump(train, f)


### Shuffle Train data into 5 folds of training and validation (4:1 split)

# Shuffle training data
data = train
random.shuffle(data)

# Split training data into 5 folds
interval = int(len(data)/5)
start = 0
end = interval
for i in range(5):
    (out_dir/str(i)).mkdir()

    val = data[start:end]
    train = data[0:start] + data[end:]

    with open(out_dir/str(i)/"val.dct", "w+") as f:
        json.dump(val, f)

    with open(out_dir/str(i)/"train.dct", "w+") as f:
        json.dump(train, f)

    start = end
    end = end + interval

    # For the last fold, use all remaining data
    if i == 4:
        end = len(data)