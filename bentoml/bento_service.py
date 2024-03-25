import bentoml
import torch
import numpy as np
from bentoml.io import Text, NumpyNdarray
from data_modeling_helper_code.data_functions import *
from data_modeling_helper_code.model_functions import *


def create_bento_service_torch(bento_name):
    """
    Create a Bento service for a PyTorch model.
    """
    # Load model
    model = bentoml.pytorch.get(bento_name).to_runner()

    # Create the service
    service = bentoml.Service(bento_name + "_service", runners=[model])

    return model, service

model, service = create_bento_service_torch("emoji_bentoml_model")


@service.api(input=Text(), output=NumpyNdarray())
def predict(sentence_str) -> np.ndarray:
    """
    Returns emoji classification prediction for a sample test sentence
    """
    max_length  = 10
    embedding_layer, word2idx, idx2word = torch_pretrained_embedding()
    sentence_str = np.array([sentence_str])
    sentence_str = sentences_to_indices(sentence_str, word2idx, max_length)
    sentence_str = torch.from_numpy(sentence_str)
    with torch.no_grad():
        predictions = model.forward.run(sentence_str, predict=True)
        _, max_prediction = torch.max(predictions, dim=1)
    result = max_prediction.numpy()

    return result

#print(predict("I could not be more confused"))