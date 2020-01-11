import torch


### Tried with less layers, results were not as good
class FFNN(torch.nn.Module):
    def __init__(self, hidden_size, p=0.5):
        super(FFNN, self).__init__()
        self.input_size = 600
        self.hidden_size = hidden_size
        self.output_size = 300
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        return x

### Tried with batch norm layers as well, results were not as good
class FFNNTwo(torch.nn.Module):
    def __init__(self, hidden_size, p=0.5):
        super(FFNNTwo, self).__init__()
        self.input_size = 550
        self.hidden_size = hidden_size
        self.output_size = 300
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size[0])
        self.fc2 = torch.nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.fc3 = torch.nn.Linear(self.hidden_size[1], self.output_size)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


### Tried with more layers, results were not as good
class FFNNThree(torch.nn.Module):
    def __init__(self, hidden_size, p=0.5):
        super(FFNNThree, self).__init__()
        self.input_size = 600
        self.hidden_size = hidden_size
        self.output_size = 300
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size[0])
        self.fc2 = torch.nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.fc3 = torch.nn.Linear(self.hidden_size[1], self.hidden_size[2])
        self.fc4 = torch.nn.Linear(self.hidden_size[2], self.output_size)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.leaky_relu(self.fc1(x)))
        x = self.dropout(torch.nn.functional.leaky_relu(self.fc2(x)))
        x = self.dropout(torch.nn.functional.leaky_relu(self.fc3(x)))
        x = self.fc4(x)
        return x