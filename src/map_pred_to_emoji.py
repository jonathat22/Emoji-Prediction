import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_unicode_column(emoji_df):
    """
    reformats emoji unicodes to be Python friendly
    Example: U+1F601 -> "\\U0001F601"

    Arguments:
    emoji_df -- emoji_df generated in from build_emoji_df()

    Returns:
    emoji_df -- emoji_df with adjusted unicode column
    """
    emoji_df['unicode'] = "\n" + emoji_df['unicode'].str.replace("+", "000").astype(str)
    emoji_df['name'] = emoji_df['name'].str.replace(" ", "_").astype(str)
    return emoji_df


def subset_emoji_df(full_emoji):
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
    data = full_emoji['name'].isin(most_popular_emojis)
    data = full_emoji[data].reset_index()
    return data


def build_emoji_df(filepath):
    """
    loads and processes full emoji dataset.
    calls convert_unicode_column() and subset_emoji_df()

    Arguments:
    filepath -- path to the full emoji dataset

    Returns:
    emoji_df -- dataframe containing index, emoji and unicode columns
    """

    full_emoji = pd.read_csv(filepath)
    emoji_df = subset_emoji_df(full_emoji)
    emoji_df = convert_unicode_column(emoji_df)
    cols_to_drop = ["index", "#", "Apple", 
                    "Google", "Facebook", "Windows", 
                    "Twitter", "JoyPixels", "Samsung", 
                    "Gmail", "SoftBank", "DoCoMo", "KDDI"]

    emoji_df = emoji_df.drop(cols_to_drop, axis=1)
    return emoji_df