import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader
import torchtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_functions import *
from src.model_functions import *


def main():
    batch_size = 3
    X_train, y_train = read_emoji_csv('Data/train_emoji.csv')
    X_test, y_test = read_emoji_csv('Data/test_emoji.csv')
    train_loader, test_loader, embedding_layer, max_length = build_dataloaders(X_train, X_test, y_train, y_test, batch_size)

    embedding_dim = 50
    sequence_length = max_length
    hidden_size = 128
    num_classes = 30
    batch_size = 3
    num_epochs = 25
    learning_rate = 0.001
    num_layers = 2

    model = GRUNet(input_dim=sequence_length, 
                hidden_dim=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                embedding_layer=embedding_layer,
                embedding_dim=embedding_dim,
                batch_size=batch_size).to(torch.device("mps"))

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history, epoch_lst = fit(model=model, 
                         num_epochs=num_epochs, 
                         train_loader=train_loader,
                         test_loader=test_loader,
                         lr=learning_rate,
                         optimizer=optimizer)

    torch.save(model.state_dict(), "model/emoji_model.pt")
    plot_results(history, epoch_lst)



if __name__ == '__main__':
    main()