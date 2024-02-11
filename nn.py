import torch
import torch.nn as nn
from features import *
from data import get_train_data
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], NB_CLASSES))
        self.activation = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return self.softmax(x)


def main():
    features = ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features + GEOGRAPHY_FEATURES
    
    net = Net(len(features), [50, 50, 50])
    criterion = nn.CrossEntropyLoss()
    
    train_x, train_y, val_x, val_y = get_train_data(features, n_data=-1)
    
    train_x = torch.tensor(train_x.values, dtype=torch.float32)
    val_x = torch.tensor(val_x.values, dtype=torch.float32)
    
    train_y = torch.zeros((train_y.shape[0], NB_CLASSES), dtype=torch.float32).scatter_(1, torch.tensor(train_y.values).unsqueeze(1), 1)
    # val_y = torch.zeros((val_y.shape[0], NB_CLASSES), dtype=torch.float32).scatter_(1, torch.tensor(val_y.values).unsqueeze(1), 1)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    losses = []
    accuracies = []
    
    pbar = tqdm(range(1000))
    for epoch in pbar:
        optimizer.zero_grad()
        outputs = net(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accuracies.append((outputs.argmax(1) == train_y.argmax(1)).float().mean().item())
        
        diff = abs(outputs.argmax(1).numpy() - train_y.argmax(1).numpy()).sum()
        pbar.set_description(f"loss: {loss.item()} diff: {diff}")
    
    pred_y = net(val_x).argmax(1).numpy()
    print(net(train_x))
    print(net(train_x).argmax(1).numpy())
    print(train_y.argmax(1).numpy())
    score = f1_score(val_y.values, pred_y, average="weighted")
    print('score = ', score)
    
    plt.plot(losses)
    plt.show()
    plt.plot(accuracies)
    plt.show()


if __name__ == "__main__":
    main()
        