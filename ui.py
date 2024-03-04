import streamlit as st
import requests
import numpy as np
import torch
import emoji
from src.map_pred_to_emoji import *


API_ENDPOINT = "http://localhost:3000/predict"

st.title("Emoji Prediction Using Deep Learning")

st.text("Type in a sentence and recieve a sentiment classification prediction in the form of an emoji!")


def predict(sentence):
    """
    A function that sends a prediction request to the API and returns the predicted emoji
    """
    response = requests.post(API_ENDPOINT, headers={"content-type": "text/plain"}, data=sentence)

    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Status: {}".format(response.status_code))


def main():
    sentence = st.text_input("Type in your sentence here")
    if len(sentence.split()) > 0:
        if len(sentence.split()) <= 10: # max sequence length is 10
            prediction = predict(sentence)
            if len(prediction) == 3:
                prediction = int(prediction[1]) # output is [2] or [26], for example, so the if elif statements are extracting the numbers
            elif len(prediction) == 4:
                prediction = int(prediction[1:3])
            emoji_map = build_emoji_df(filepath="Data/full_emoji.csv")
            emoji_name = emoji_map['name'].iloc[prediction]
            st.title(emoji.emojize(f":{emoji_name}:"))
            #st.success(f"Your predicted emoji is {prediction:.3f}")
        else:
            st.write("Sequence is too long. It must have 10 or fewer words.")
            st.write(len(sentence.split()))
    else:
        st.write("You didn't type in a sentence!")
        st.write("\U0001F603")
        




if __name__ == "__main__":
    main()