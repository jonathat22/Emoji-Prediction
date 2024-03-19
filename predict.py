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


max_length  = 10
embedding_layer, word2idx, idx2word = torch_pretrained_embedding()

model = GRUNet(input_dim=max_length, 
                hidden_dim=128,
                num_classes=30,
                num_layers=2,
                embedding_layer=embedding_layer,
                embedding_dim=50,
                batch_size=3).to(torch.device("mps"))

model.load_state_dict(torch.load("emoji_model.pt"))
model.eval()
def predict(sentence_str,
            model=model,
            max_len=max_length, 
            embed_layer=embedding_layer,
            word_to_idx=word2idx,
            idx_to_word=idx2word):
    """
    Returns emoji classification prediction for a sample test sentence
    """
    sentence_str = np.array([sentence_str])
    sentence_str = sentences_to_indices(sentence_str, word2idx, max_length)
    sentence_str = torch.from_numpy(sentence_str)
    with torch.no_grad():
        predictions = model.forward(sentence_str, predict=True)
        _, max_prediction = torch.max(predictions, dim=1)
    result = max_prediction.numpy()

    return result