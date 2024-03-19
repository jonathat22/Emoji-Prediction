import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader
import torchtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def convert_to_one_hot(Y, num_emojis):
    """
    One hot encodes Y labels

    Arguments:
    Y -- array containing the indices of the ground truth emoji
    num_emojis -- the number of classes

    Returns:
    Y -- array with one hot encoded vectors of each target emoji
    """
    Y = np.eye(num_emojis)[Y.reshape(-1)]
    return Y



def read_emoji_csv(path):
    """
    Reads emoji dataset

    Argument:
    path -- location of the dataset

    Returns:
    X -- array of training data sentences
    y -- array of indices representing the target emoji for each X sentence 
    """
    data = pd.read_csv(path, header=None)
    X = data.iloc[:,0].values
    y = data.iloc[:,1].values
    return X, y



def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m,)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]
    X_indices = np.zeros((m, max_len)).astype(np.float32)
    for i in range(m):
        sentence_words = [i.lower() for i in X[i].split()]
        for idx, val in enumerate(sentence_words):
            X_indices[i, idx] = word_to_index[val]
    
    return X_indices



def torch_pretrained_embedding(load=False):
    """
    Creates the embedding layer and loads pre-trained GloVe 50-dimensional vectors

    Returns:
    embedding layer -- pretrained layer
    word2idx -- dictionary mapping each word to its index
    idx2word -- dictionary mapping index to word
    """
    if load == True:
        glove = torch.load("saved_glove.pt")
    else:
        glove = torchtext.vocab.GloVe(name='6B', dim=50)

    word2idx = glove.stoi
    idx2word = glove.itos
    embedding_layer = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
    return embedding_layer, word2idx, idx2word



class EmojiDataset():
    """
    Taken from  https://www.kaggle.com/code/abdoukassimi/classification-rnn-gru
    Learned to use classes to create custom datasets
    """
    def __init__(self, X_features, y_labels):
        self.x_features = torch.from_numpy(X_features)
        self.y_labels = torch.from_numpy(y_labels).type(torch.float32)
        self.len = len(X_features)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_features[index], self.y_labels[index]



def build_dataloaders(X_train, X_test, y_train, y_test, word2idx, max_length, batch_size):
    """
    Constructs train and test data loaders

    Arguments:
    X_train -- training data
    X_test -- testing data
    y_train -- training labels
    y_test -- testing labels

    Returns:
    train_loader -- train DataLoader object
    test_loader -- test DataLoader object
    """
    X_train_indices = sentences_to_indices(X_train, word2idx, max_length)
    X_test_indices = sentences_to_indices(X_test, word2idx, max_length)
    y_train_oh = convert_to_one_hot(y_train, num_emojis = 30)
    y_test_oh = convert_to_one_hot(y_test, num_emojis = 30)
    train_data = EmojiDataset(X_features=X_train_indices, y_labels=y_train_oh)
    test_data = EmojiDataset(X_features=X_test_indices, y_labels=y_test_oh)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader