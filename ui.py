import streamlit as st
import requests
import numpy as np
import torch
import emoji
import unicodeit
from src.map_pred_to_emoji import *

# "http://localhost:3000/predict"
API_ENDPOINT = "http://localhost:3000/predict"

st.set_page_config(layout="wide")

st.title(" \U0001F680 Emoji Prediction Using Deep Learning \U0001F525")

st.write("""
    Natural Language Processing (NLP) is a subfield of machine/deep learning that is concerned with
    building models to process, understand, and generate human language. NLP has become an integral part 
    of the technologies many of us use and depend on. Natural language processing has many use cases 
    such as sentiment analysis, toxicity classification, named entity recognition, and more.
""")

st.write("""
    Emojis visually represent emotions or other objects that are conveyed or discussed in text messages, 
    and can enhance the communication experience between people.
""")

st.write("""
    On social platforms like Twitter, Instagram, and instant messaging, 
    emojis allow people to efficiently convey their feelings in ways they cannot with only words.
""")

st.write("""
    One can think of the emoji prediction problem as a more fun variant of the 
    sentiment classification problem. In this multi-class classification problem,
    instead of the output being a word like "positive" or "negative" the output is an emoji.
    In this app, I implement a Gated Recurrent Unit (GRU) network and use it to 
    predict the most relevant emoji for an input sentence. \U0001F92F
""")

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
            st.write(prediction)
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


#with col2:
with st.expander("All about GRUs \U0001F9E0"):
    col1, col2 = st.columns(2)

    #######################################    RNN Section   ########################################################


    col1.header("Recurrent Neural Network")

    col1.write("""
        The most basic deep learning architecture used for NLP applications is the Recurrent 
        Neural Network (RNN), shown in Figure 1. In an RNN, input sequences (x in the diagram) 
        are concatenated with the previous hidden state (the model's "memory"), then the 
        concatenated vector goes through a tanh activation function and becomes the new hidden state.
        Inputs are processed sequentially and this process is repeated for each time step.
        """)

    col1.image("diagram_images/RNN_diagram.png", caption="Figure 1: Diagram of RNN architecture")

    col1.write("""
        One issue with the basic **RNN** is that it is susceptible to the **vanishing gradient problem**.
        During backpropagation, the gradients are used to update all the weights such that the difference
        between the model's output and the ground truth is minimized. The deeper the model architecture 
        (or in this case, the longer your training sequence), the more likely the gradients become so small
        that the model doesn't learn much anymore in the earlier layers. As a result, for longer sequences, 
        the model can "forget" what it saw earlier in the sequence.
    """)

    col1.write("""
        The vanishing gradient problem is addressed by the developement of Long Short-Term Memory (LSTM)
        and Gated Recurrent Unit (GRU) network architectures.
    """)

    col1.header("Gated Recurrent Unit Network")

    col1.write("""
        The Gated Recurrent Unit (GRU) network is a more evolved version of the basic RNN and has architectural features that can solve the
        vanishing gradient problem.
    """)

    col1.write("""
        Like RNNs, GRUs process data sequentially and passes information as it progates forward.
        A GRU cell uses hidden state ***({})*** and gates that control what information is kept or forgotten.
    """.format(unicodeit.replace('H_t')))


    ##############################################    GRU Section     ##########################################


    col2.image("diagram_images/GRU_diagram.png", caption="Figure 2: Diagram of GRU Cell", width=650)

    col2.write("""
        The GRU architecture contains several key components that allow it selectively keep or forget information over time.

        -- **Hidden state**: vector of numbers that represents the network's memory of the previous inputs, gets updated based 
           on the current input and the previous hidden state
    """)

    col2.write("""
        -- **Reset Gate**: Concatenates the current input and previous hidden state vectors, passes this concatenated vector through the sigmoid function to determine 
           how much of the previous hidden state should be forgotten
    """)

    col2.write("""
        -- **Update Gate**: The Update Gate determines how much of the candidate activation 
           vector will be included in the new hidden state
    """)

    col2.write("""
        --**Candidate Activation Vector**: This is a vector that combines the "reset" version of previous hidden state with
        the current input and is pushed through the tanh function to squeeze values between -1 and 1. This helps regulate the network.
        This vector is used to update next hidden state
    """)

    col2.latex(r'''\tag{Reset Gate} R_t = \sigma(X_tW_{xr} + H_{t-1}W_{hr} + b_r)''')
    col2.latex(r'''\tag{Update Gate} Z_t = \sigma(X_tW_{xz} + H_{t-1}W_{hz} + b_z)''')
    col2.latex(r'''\tag{Candidate} \tilde{H_t} = tanh(X_tW_{xh} + (R_t\odot{H_{t-1}})W_{hh} + b_h)''')
    col2.latex(r'''\tag{Hidden State} H_t = Z_t\odot{H_{t-1}} + (1-Z_t)\odot{\tilde{H_t}}''')
    col2.write("Note: Wxr, Whr, Wxz, Whz, Wxh, Whh are weight matricies, br, bz, bh are bias parameters. These are the parameters that are learned during training.")



    


        


with st.sidebar:
    st.markdown("---")
    st.markdown("# About")
    st.markdown("Here, I use Natual Language Processing to explor the relationship between text and emojis \U0001F913")
    st.markdown("Type in a sentence and recieve a sentiment classification prediction in the form of an emoji!")

    st.markdown("---")
    st.markdown("# References")
    st.write("1. [Dive Into Deep Learning 10.2 Gated Recurrent Units (GRU)](https://d2l.ai/chapter_recurrent-modern/gru.html)")
    st.write("2. [Illustrated Guide to LSTM's and GRU's: A Step by Step Explanation by Michael Phi](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)")
    st.write("3. [Recurrent Neural Networks Cheatsheet by Afshine Amidi and Shervine Amidi](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)")
    st.write("4. [Understanding Gated Recurrent Unit (GRU) in Deep Learning](https://medium.com/@anishnama20/understanding-gated-recurrent-unit-gru-in-deep-learning-2e54923f3e2)")
    st.write("5. [A complete Guide to Natural Language Processing by DeepLearning.AI](https://www.deeplearning.ai/resources/natural-language-processing/)")
    st.write("6. [Unlocking Emotions with AI: Emoji Prediction using LSTM](https://medium.com/@govinnachiran/unlocking-emotions-with-ai-emoji-prediction-using-lstm-e673662fd7e3)")
    st.write("7. [Understanding LSTM Networks via colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)")
    



if __name__ == "__main__":
    main()