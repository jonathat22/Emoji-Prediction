import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def convert_unicode_column(emoji_names_unicode_data):
    """
    reformats emoji unicodes to be Python friendly
    Example: U+1F601 -> "\\U0001F601"

    Arguments:
    emoji_df -- emoji_df generated in from build_emoji_df()

    Returns:
    emoji_df -- emoji_df with adjusted unicode column
    """
    emoji_names_unicode_data['unicode'] = "\n" + emoji_names_unicode_data['unicode'].str.replace("+", "000").astype(str)
    emoji_names_unicode_data['name'] = emoji_names_unicode_data['name'].str.replace(" ", "_").astype(str)
    return emoji_names_unicode_data


def subset_emoji_emoji_names_unicode_data(full_emoji):
    """
    returns subset of emoji_df with the 30 most popular emojis
    according to Emojipedia.com

    Arguments:
    full_emoji -- full_emoji loaded in build_emoji_df()

    Returns:
    emoji_df -- subset of emoji_df
    """

    popular_smileys = ['face with tears of joy',
                   'rolling on the floor laughing',
                   'loudly crying face',
                   'smiling face with heart-eyes',
                   'smiling face with smiling eyes',
                   'smiling face with hearts',
                   'grinning face with sweat',
                   'beaming face with smiling eyes',
                   'face blowing a kiss',
                   'pleading face']

    popular_gestures = ['folded hands',
                        'thumbs up',
                        'clapping hands',
                        'flexed biceps',
                        'raising hands',
                        'ok hand',
                        'victory hand',
                        'waving hand',
                        'backhand index pointing down',
                        'oncoming fist',
                        'vulcan salute']

    other_popular_emojis = ['red heart',
                            'sparkles',
                            'fire',
                            'person facepalming',
                            'person shrugging',
                            'face with symbols on mouth',
                            'thinking face',
                            'fork and knife with plate',
                            'face with rolling eyes',
                            'disappointed face']

    most_popular_emojis = popular_smileys + popular_gestures + other_popular_emojis
    subset_with_popular_emojis = full_emoji['name'].isin(most_popular_emojis)
    subset_with_popular_emojis = full_emoji[subset_with_popular_emojis].reset_index()
    return subset_with_popular_emojis


@st.cache_data
def build_emoji_df(emoji_names_unicode_data_path):
    """
    loads and processes full emoji dataset.
    calls convert_unicode_column() and subset_emoji_df()

    Arguments:
    filepath -- path to the full emoji dataset

    Returns:
    emoji_df -- dataframe containing index, emoji and unicode columns
    """

    full_emoji = pd.read_csv(emoji_names_unicode_data_path)
    most_popular_emojis_df = subset_emoji_emoji_names_unicode_data(full_emoji)
    most_popular_emojis_df = convert_unicode_column(most_popular_emojis_df)
    unnessecary_columns = ["index", "#", "Apple", 
                    "Google", "Facebook", "Windows", 
                    "Twitter", "JoyPixels", "Samsung", 
                    "Gmail", "SoftBank", "DoCoMo", "KDDI"]

    most_popular_emojis_df = most_popular_emojis_df.drop(unnessecary_columns, axis=1)
    return most_popular_emojis_df