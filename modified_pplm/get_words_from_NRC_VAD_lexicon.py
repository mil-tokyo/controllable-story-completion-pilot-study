import os
import pandas as pd
from typing import List
from transformers.file_utils import cached_path


dir_path = "/path/to/NRC-VAD-Lexicon-Aug2018Release/"
file_name = "NRC-VAD-Lexicon.txt"

NRC_VAD_lexicon_PATH = "/path/to/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"

BAG_OF_WORDS_ARCHIVE_MAP = {
    "legal": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    "military": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    "monsters": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    "politics": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    "positive_words": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    "religion": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    "science": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    "space": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    "technology": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}


class data_from_NRC_VAD_lexicon:
    """ class for get words from NRC-VAD lexicon """

    def __init__(self, data_path="", delimiter="\t"):
        assert os.path.isfile(data_path)

        self.df = pd.read_csv(data_path, delimiter=delimiter)

    def __len__(self):
        return len(self.df)

    def get_samples_with_VAD(
        self, V_min=0.0, V_max=1.0, A_min=0.0, A_max=1.0, D_min=0.0, D_max=1.0
    ):

        print(
            f"get words from lexicon: Valence: {V_min} -- {V_max}, Asoural: {A_min} -- {A_max}, Dominance: {D_min} -- {D_max}"
        )

        sampled_df = self.df.query(
            "@V_min <= Valence <= @V_max & @A_min <= Arousal <= @A_max & @D_min <= Dominance <= @D_max"
        )

        print(f"num  of words: {len(sampled_df)}")

        return sampled_df

    def get_word_lists(
        self, V_min=0.0, V_max=1.0, A_min=0.0, A_max=1.0, D_min=0.0, D_max=1.0
    ):

        sampled_df = self.get_samples_with_VAD(
            V_min=V_min, V_max=V_max, A_min=A_min, A_max=A_max, D_min=D_min, D_max=D_max
        )
        words = sampled_df["Word"].values.tolist()

        return words


def get_bag_of_words_indices(
    bag_of_words_ids_or_paths: List[str],
    tokenizer=None,
    use_NRC_VAD_lexicon=True,
    NRC_VAD_lexicon_instance=None,
    V_min=0.0,
    V_max=1.0,
    A_min=0.0,
    A_max=1.0,
    D_min=0.0,
    D_max=1.0,
) -> List[List[List[int]]]:

    bow_indices = []

    try:
        assert use_NRC_VAD_lexicon is True or bag_of_words_ids_or_paths is not None
    except AssertionError:
        print("Please use NRC-VAD-lexicon or BAG_OF_WORDS_ARCHIVE_MAP")

    if use_NRC_VAD_lexicon:  # NRC-VAD-lexicon
        print("use NRC-VAD-lexicon to make 'bow_indices'")
        lexicon = NRC_VAD_lexicon_instance
        if lexicon is None:
            lexicon = data_from_NRC_VAD_lexicon(NRC_VAD_lexicon_PATH)

        words = lexicon.get_word_lists(
            V_min=V_min, V_max=V_max, A_min=A_min, A_max=A_max, D_min=D_min, D_max=D_max
        )
        # bow_indices.append([tokenizer.encode(word.strip(), add_prefix_space=True) for word in words])
        bow_indices.append(
            [tokenizer.encode(word.strip(), add_special_tokens=False) for word in words]
        )

    else:  # BAG_OF_WORDS_ARCHIVE_MAP (original BoW mode)
        print(f"use_NRC_VAD_lexicon {use_NRC_VAD_lexicon}")
        print("use BAG_OF_WORDS_ARCHIVE_MAP to make 'bow_indices'")
        for id_or_path in bag_of_words_ids_or_paths:
            print(f"word_list: {id_or_path}")
            if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
                filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
            else:
                filepath = id_or_path
            with open(filepath, "r") as f:
                words = f.read().strip().split("\n")
            bow_indices.append(
                [
                    tokenizer.encode(word.strip(), add_special_tokens=False)
                    for word in words
                ]
            )

    return bow_indices
