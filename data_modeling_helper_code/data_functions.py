import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader
import torchtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def convert_to_one_hot(ground_truth_labels, num_classes):
    """
    One hot encodes ground truth labels labels.

    Arguments:
    ground_truth_labels -- array containing the indices of the ground truth emoji
    num_classes -- the number of classes

    Returns:
    one_hot_ground_truth_labels -- array with one hot encoded vectors of each target emoji
    """
    one_hot_ground_truth_labels = np.eye(num_classes)[ground_truth_labels.reshape(-1)]
    return one_hot_ground_truth_labels



def read_emoji_csv(emoji_dataset_filepath):
    """
    Reads emoji dataset.
    This function is used to load both training and testing datasets.

    Argument:
    emoji_dataset_filepath -- location of the dataset

    Returns:
    features -- array of training/testing data sentences
    targets -- array of indices representing the target emoji for each sentence in features 
    """
    loaded_emoji_data = pd.read_csv(emoji_dataset_filepath, header=None)
    features = loaded_emoji_data.iloc[:,0].values
    targets = loaded_emoji_data.iloc[:,1].values
    return features, targets



def sentences_to_indices(features, word_to_index_map, max_sequence_length):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    features -- array of sentences (strings), of shape (number_of_sentences, )
    word_to_index_map -- a dictionary containing the each word mapped to its index
    max_sequence_length -- maximum number of words in a sentence. You can assume every sentence in features is no longer than this. 
    
    Returns:
    features_indices -- array of indices corresponding to words in the sentences from features, of shape (number_of_sentences, max_sequence_length)
    """
    number_of_sentences = features.shape[0]
    features_indices = np.zeros((number_of_sentences, max_sequence_length)).astype(np.float32)
    for sentence in range(number_of_sentences):
        sentence_words = [i.lower() for i in features[sentence].split()]
        for index, word in enumerate(sentence_words):
            features_indices[sentence, index] = word_to_index_map[word]
    
    return features_indices



def torch_pretrained_embedding(load=False):
    """
    Creates the embedding layer and loads pre-trained GloVe 50-dimensional vectors

    Returns:
    embedding layer -- pretrained layer
    word_to_index_map -- dictionary mapping each word to its index
    index_to_word_map -- dictionary mapping index to word
    """
    if load == True:
        glove = torch.load("src/saved_glove.pt")
    else:
        glove = torchtext.vocab.GloVe(name='6B', dim=50)

    word_to_index_map = glove.stoi
    index_to_word_map = glove.itos
    embedding_layer = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
    return embedding_layer, word_to_index_map, index_to_word_map



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



def build_dataloaders(features_training, features_testing, 
                      targets_training, targets_testing, 
                      word_to_index_map, max_sequence_length, batch_size):
    """
    Constructs train and test data loaders

    Arguments:
    features_training -- training data
    features_testing -- testing data
    targets_training -- training labels
    targets_testing -- testing labels

    Returns:
    train_loader -- train DataLoader object
    test_loader -- test DataLoader object
    """
    features_train_indices = sentences_to_indices(features_training, word_to_index_map, max_sequence_length)
    features_test_indices = sentences_to_indices(features_testing, word_to_index_map, max_sequence_length)
    one_hot_targets_train = convert_to_one_hot(targets_training, num_emojis = 30)
    one_hot_targets_test = convert_to_one_hot(targets_testing, num_emojis = 30)
    train_data = EmojiDataset(X_features=features_train_indices, y_labels=one_hot_targets_train)
    test_data = EmojiDataset(X_features=features_test_indices, y_labels=one_hot_targets_test)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader