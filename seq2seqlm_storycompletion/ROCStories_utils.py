import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import *

try:  # transformers >= v4
    from transformers.models.bart.modeling_bart import shift_tokens_right
except ModuleNotFoundError:  # transformers < v3
    from transformers.modeling_bart import shift_tokens_right


from tqdm import tqdm, trange
from collections import namedtuple

from typing import List, Dict, Tuple, Callable  # Any, Optional, Union

# ---
# Dataset with missing-sentence-mask processing
# ---
#
# See below as an example:
#
# https://github.com/huggingface/transformers/blob/023f0f3708f73e4fdffb92505296cd7d3d928aef/examples/seq2seq/utils.py
#


class ROCStories_fixed_missing_sentence_mask(Dataset):
    """ Dataset class for validation (dev) and test sets """

    """
    columns=['stories_with_missing_sentence1',
             'stories_with_missing_sentence2',
             'stories_with_missing_sentence3',
             'stories_with_missing_sentence4',
             'missing_id', 'missing_sentence',
             'storyid', 'storytitle']
    """

    def __init__(self, data_path=""):
        assert os.path.isfile(data_path)

        self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        assert type(idx) == int

        row = self.df.iloc[idx].values

        story_lines = row[
            0:4
        ]  # one sentence is lost from the original five-sentence story
        remain_sentences = story_lines

        missing_id = row[4]
        missing_sentence = row[5:6]
        storyid = row[6]

        #
        # Convert missing_sentence from np.array to str.
        # Now, missing_sentence is just one sentence.
        #
        missing_sentence = missing_sentence[0]

        # concatenate sentences with the missing-sentence-mask special token.
        given_context = np.insert(remain_sentences, missing_id, "<missing_sentence>")
        given_context = " ".join(given_context)

        #
        # Outputs
        #
        # given_context (str): concatnated 4 sentences + <missing_sentence> for input. One sentence is lost from the original five-sentence story.
        # missing_sentence (str): One sentence missing from the original five-sentence story.
        # missing_id (int): The position where the sentence is missing (from 0 to 4).
        # remain_sentences (List(str)): given 4 sentences, without concatenation
        # storyid (str) : story id
        # dataset_index (int) : index in dataset (because storyid (str) cannot be treated in torch.Tensor)
        #

        # return given_context, missing_sentence, missing_id, remain_sentences

        return {
            "context": given_context,
            "target_sentence": missing_sentence,
            "missing_id": missing_id,
            "remain_sentences_list": remain_sentences,
            "storyid": storyid,
            "dataset_index": idx,
        }


class ROCStories_random_missing_sentence_mask(Dataset):
    """ Dataset class for train set """

    """
    columns=['sentence1',
             'sentence2',
             'sentence3',
             'sentence4',
             'sentence5',
             'storyid',
             'storytitle']
    """

    def __init__(self, data_path=""):
        assert os.path.isfile(data_path)

        self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        assert type(idx) == int

        row = self.df.iloc[idx].values

        story_lines = row[0:5]

        storyid = row[5]

        #
        # random missing_id
        #
        missing_id = np.random.randint(low=0, high=5)

        missing_sentence = np.array([story_lines[missing_id]], dtype=object)

        #
        # Convert missing_sentence from np.array to str.
        # Now, missing_sentence is just one sentence.
        #
        missing_sentence = missing_sentence[0]

        remain_sentences = np.delete(story_lines, missing_id)

        # concatenate sentences with the missing-sentence-mask special token.
        given_context = np.insert(remain_sentences, missing_id, "<missing_sentence>")
        given_context = " ".join(given_context)

        #
        # Outputs
        #
        # given_context (str): concatnated 4 sentences + <missing_sentence> for input. One sentence is lost from the original five-sentence story.
        # missing_sentence (str): One sentence missing from the original five-sentence story.
        # missing_id (int): The position where the sentence is missing (from 0 to 4).
        # remain_sentences (List(str)): given 4 sentences, without concatenation
        # storyid (str) : story id
        # dataset_index (int) : index in dataset (because storyid (str) cannot be treated in torch.Tensor)
        #

        # return given_context, missing_sentence, missing_id, remain_sentences

        return {
            "context": given_context,
            "target_sentence": missing_sentence,
            "missing_id": missing_id,
            "remain_sentences_list": remain_sentences,
            "storyid": storyid,
            "dataset_index": idx,
        }
