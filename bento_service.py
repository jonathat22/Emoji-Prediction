import bentoml
import torch
import numpy as np
from bentoml.io import Text, NumpyNdarray
from src.data_functions import *
from src.model_functions import *


def create_bento_service_torch(bento_name):
    """
    Create a Bento service for a PyTorch model.
    """
    # Load model
    model = bentoml.pytorch.get("emoji_bentoml_model:latest").to_runner()

    # Create the service
    service = bentoml.Service(bento_name + "_service", runners=[model])

    return model, service

model, service = create_bento_service_torch("emoji_bentoml_service")


@service.api(input=Text(), output=int)
@torch.no_grad()
def predict(sentence):
    """
    Returns emoji classification prediction for a sample test sentence
    """
    max_length  = 10
    embedding_layer, word2idx, idx2word = torch_pretrained_embedding()
    sentence = np.array([sentence])
    sentence = sentences_to_indices(sentence, word2idx, max_length)
    sentence = torch.from_numpy(sentence)
    model.init_local()
    predictions = model.forward.run(sentence, predict=True)
    _, max_prediction = torch.max(predictions, dim=1)
    return max_prediction.numpy()[0]