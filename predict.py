import torch
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader
import torchtext
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from data_modeling_helper_code.data_functions import *
from data_modeling_helper_code.model_functions import *


@st.cache_data
def load_embeddings():
    embedding_layer, word2idx, idx2word = torch_pretrained_embedding(load=True)
    return embedding_layer, word2idx, idx2word


@st.cache_resource
def load_model(_embedding_layer):
    model = GRUNet(input_dim=10, 
                hidden_dim=128,
                num_classes=30,
                num_layers=2,
                embedding_layer=_embedding_layer,
                embedding_dim=50,
                batch_size=3)

    model.load_state_dict(torch.load("emoji_model.pt", map_location=lambda storage, loc: storage))

    return model


def prepare_data(sentence_str, word_to_idx, max_length=10):
    sentence_str = np.array([sentence_str])
    sentence_str = sentences_to_indices(sentence_str, word_to_idx, max_length)
    prepared_data = torch.from_numpy(sentence_str)
    return prepared_data


def predict(prepared_data, model):
    """
    Returns emoji classification prediction for a sample test sentence
    """
    with torch.no_grad():
        predictions = model.forward(prepared_data, predict=True)
        _, max_predictions = torch.topk(predictions, 3, dim=1)
    result = max_predictions.numpy()

    return result