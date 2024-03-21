import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader
import torchtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def accuracy(outputs, labels):
    """
    Returns accuracy score

    Arguments:
    outputs -- outputs generated by GRU network
    labels -- ground truth labels

    Returns:
    accuracy score
    """
    _, preds = torch.max(outputs, dim=1)
    labels = torch.argmax(labels, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class ModelHelperFunctions(nn.Module):
    """
    Model helper functions to be inherited by GRU model

    Functions:
    training_step -- takes a batch of data, computes training loss
    validate -- takes batch of data, computes validation loss
    validation_results -- combines batch losses and accuracies to get per-epoch loss and accuracy
    result_per_epoch -- prints formatted results per epoch
    """
    def training_step(self, batch):
        lines, labels = batch
        lines, labels = lines.to(torch.float32).to(torch.device("mps")), labels.to(torch.device("mps"))
        output = self(lines)
        loss = F.cross_entropy(output, labels)
        return loss

    def validate(self, batch):
        line, labels = batch
        line, labels = line.to(torch.device("mps")), labels.to(torch.device("mps"))
        output = self(line)
        loss = F.cross_entropy(output, labels)
        acc = accuracy(output, labels)
        return {'test_loss': loss.detach(), 'test_acc': acc}

    def validation_results(self, outputs):
        batch_losses = [x['test_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['test_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'test_loss': epoch_loss.item(), 
                'test_acc': epoch_acc.item()}

    def result_per_epoch(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(
            epoch, result['train_loss'], result['test_loss'], result['test_acc']))



class GRUNet(ModelHelperFunctions):
    """
    Implementation of GRU Network
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, embedding_layer, embedding_dim, batch_size, dropout=0.1):

        super(GRUNet, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embeddings = embedding_layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True) # x must have shape: (batch_size, seq_len, input_size)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, predict=False):
        if predict == True:
            self.batch_size = 1
            hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        else:
            hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim) #.to(torch.device("mps"))
            
        x = self.embeddings(x.to(torch.int64))
        out, _ = self.gru(x, hidden) # shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out



@torch.no_grad()
def evaluate(model, test_loader):
    """
    Model evaluation function

    Arguments:
    model -- model
    test_loader -- test data model hasn't yet seen

    Returns:
    model accuracies and losses
    """
    model.eval()
    outputs = [model.validate(batch) for batch in test_loader]
    return model.validation_results(outputs)


def fit(model, num_epochs, train_loader, test_loader, lr, optimizer):
    """
    Model training function

    Arguments:
    model -- model
    num_epochs -- number of epochs/length of training
    train_loader -- training data
    test_loader -- testing data
    lr -- learning rate
    optimizer -- optimizer function

    Returns:
    history -- list of model losses and accuracies per epoch
    epoch_lst -- list of epochs
    """
    history = []
    epoch_list = []

    for epoch in range(num_epochs):
        epoch_list.append(epoch)

        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Testing Phase
        result = evaluate(model, test_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.result_per_epoch(epoch, result)
        history.append(result)

    return history, epoch_list



def plot_results(history, epoch_lst):
    """
    Graphs train and test loss and accuracy per epoch

    Arguments:
    history -- list of model losses and accuracies per epoch
    epoch_lst -- list of epochs
    """
    accuracies = [x['test_acc'] for x in history]
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['test_loss'] for x in history]
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(epoch_lst, accuracies)
    ax1.set_title('Validation Accuracy vs. No. of epochs')
    ax1.set(ylabel='Accuracy')
    ax2.plot(epoch_lst, train_losses)
    ax2.plot(epoch_lst, val_losses)
    ax2.set_title('Losses vs. No. of epochs')
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.legend(['Training', 'Validation'])
    plt.show()