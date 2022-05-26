import os
import itertools
import argparse
import random
import math
import time

import numpy as np
import pandas as pd

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

from tqdm import tqdm, trange
from collections import namedtuple

from typing import List, Dict, Tuple, Optional, Callable  # Any,  Union

from utils import label_smoothed_nll_loss, lmap, calculate_bleu, calculate_rouge

# ---
# Define my collator
# ---
#
# What is collator?
# See : https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainer#id1
#
# > data_collator (DataCollator, optional) – The function to use to form a batch from a list of elements of train_dataset or eval_dataset. Will default to default_data_collator() if no tokenizer is provided, an instance of DataCollatorWithPadding() otherwise.
#

#
# See below as examples:
#
# https://github.com/huggingface/transformers/blob/023f0f3708f73e4fdffb92505296cd7d3d928aef/examples/seq2seq/finetune_trainer.py
#
# ```
# from utils import (
#     LegacySeq2SeqDataset,
#     Seq2SeqDataCollator,         <-
#     Seq2SeqDataset,
#     assert_all_frozen,
#     build_compute_metrics_fn,
#     check_output_dir,
#     freeze_embeds,
#     freeze_params,
#     lmap,
#     save_json,
#     use_task_specific_params,
#     write_txt_file,
# )
# ```
#
# https://github.com/huggingface/transformers/blob/023f0f3708f73e4fdffb92505296cd7d3d928aef/examples/seq2seq/utils.py
#


logger = getLogger(__name__)


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class StoryCompletion_DataCollator:
    def __init__(
        self,
        tokenizer,
        data_args,
        model_config,
        tpu_num_cores=None,
        with_dataset_index=False,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = model_config.pad_token_id
        self.decoder_start_token_id = model_config.decoder_start_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores

        #
        # v3.4.0 のものでは、 t5 で `Keyword arguments {'add_prefix_space': False} not recognized.`` という warning が出る。
        # master (2020/10/26時点) では修正されている。
        #
        # self.dataset_kwargs = {"add_prefix_space": isinstance(tokenizer, BartTokenizer)}
        self.dataset_kwargs = (
            {"add_prefix_space": True}
            if isinstance(self.tokenizer, BartTokenizer)
            else {}
        )

        self.with_dataset_index = with_dataset_index

        # In this time, we don't use language translation.
        #
        #         if data_args.src_lang is not None:
        #             self.dataset_kwargs["src_lang"] = data_args.src_lang
        #         if data_args.tgt_lang is not None:
        #             self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang
        #

    def __call__(self, batch) -> Dict[str, torch.Tensor]:

        # add storyids to original __call__ function
        dataset_indices = torch.tensor([x["dataset_index"] for x in batch])
        # dataset_indices = trim_batch(dataset_indices, self.pad_token_id)

        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(
                input_ids, self.pad_token_id, attention_mask=attention_mask
            )

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels)
        else:
            decoder_input_ids = shift_tokens_right(
                input_ids=labels,
                pad_token_id=self.pad_token_id,
                decoder_start_token_id=self.decoder_start_token_id,
            )

        if (
            self.with_dataset_index
        ):  # When you need to associate the batch and ROCStories
            batch = {
                "input_ids": input_ids,  # List of token ids to be fed to the encoder.
                "attention_mask": attention_mask,  # List of indices specifying which tokens should be attended to by the model.
                "decoder_input_ids": decoder_input_ids,  #
                "labels": labels,  # List of token ids for tgt_texts
                "dataset_indices": dataset_indices,  # added to original __call__ function
            }
        else:
            batch = {
                "input_ids": input_ids,  # List of token ids to be fed to the encoder.
                "attention_mask": attention_mask,  # List of indices specifying which tokens should be attended to by the model.
                "decoder_input_ids": decoder_input_ids,  #
                "labels": labels,  # List of token ids for tgt_texts
            }
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:

        #
        # What's prepare_seq2seq_batch?
        #
        # https://huggingface.co/transformers/model_doc/bart.html#transformers.BartTokenizer.prepare_seq2seq_batch
        #
        # returns:
        # A BatchEncoding with the following fields:
        #
        # input_ids – List of token ids to be fed to the encoder.
        # attention_mask – List of indices specifying which tokens should be attended to by the model.
        # labels – List of token ids for tgt_texts
        #
        # The full set of keys [input_ids, attention_mask, decoder_input_ids,  decoder_attention_mask], will only be returned if tgt_texts is passed.
        # Otherwise, input_ids, attention_mask will be the only keys.
        #

        try:
            batch_encoding = self.tokenizer.prepare_seq2seq_batch(
                [x["context"] for x in batch],
                tgt_texts=[x["target_sentence"] for x in batch],
                max_length=self.data_args.max_length,
                max_target_length=self.data_args.max_length,
                padding="max_length"
                if self.tpu_num_cores is not None
                else "longest",  # TPU hack
                return_tensors="pt",
                **self.dataset_kwargs,
            )
        except NotImplementedError:

            def prepare_seq2seq_batch(
                tokenizer,
                src_texts: List[str],
                tgt_texts: Optional[List[str]] = None,
                max_length: Optional[int] = None,
                max_target_length: Optional[int] = None,
                padding: str = "longest",
                return_tensors: str = "None",
                truncation=True,
                **kwargs,
            ) -> BatchEncoding:
                kwargs.pop("src_lang", None)
                kwargs.pop("tgt_lang", None)
                if max_length is None:
                    max_length = self.data_args.max_length
                model_inputs: BatchEncoding = tokenizer(
                    src_texts,
                    add_special_tokens=True,
                    return_tensors=return_tensors,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    **kwargs,
                )
                if tgt_texts is None:
                    return model_inputs
                # Process tgt_texts
                if max_target_length is None:
                    max_target_length = max_length
                labels = tokenizer(
                    tgt_texts,
                    add_special_tokens=True,
                    return_tensors=return_tensors,
                    padding=padding,
                    max_length=max_length,
                    truncation=truncation,
                    **kwargs,
                )["input_ids"]
                model_inputs["labels"] = labels
                return model_inputs

            batch_encoding = prepare_seq2seq_batch(
                self.tokenizer,
                [x["context"] for x in batch],
                tgt_texts=[x["target_sentence"] for x in batch],
                max_length=self.data_args.max_length,
                max_target_length=self.data_args.max_length,
                padding="max_length"
                if self.tpu_num_cores is not None
                else "longest",  # TPU hack
                return_tensors="pt",
                **self.dataset_kwargs,
            )

        return batch_encoding.data


#
# modified functions originally from utils.py
#


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""

    """
    We use this params for story completion, 
    because it seems summarization is more similar to story completion 
    than translation is.
    """

    task_specific_params = model.config.task_specific_params

    if "storycompletion" in task:
        if model.config.model_type != "pegasus":
            param_from_task = "summarization"
        else:
            param_from_task = "summarization_xsum"

        logger.info(
            f"To fine-tune for {task}, invoke the params for {param_from_task} with some exception."
        )
    else:
        param_from_task = task

    if task_specific_params is not None:
        pars = task_specific_params.get(param_from_task, {})

        # if model type is "t5", remove its summarization prefix
        if model.config.model_type == "t5":
            try:
                pars.pop("prefix")
            except:
                pass

        # if model type is "pegasus", don't change its max_position_embeddings
        if model.config.model_type == "pegasus":
            try:
                pars.pop("max_position_embeddings")
            except:
                pass

        logger.info(f"using task specific params for {param_from_task}: {pars}")
        model.config.update(pars)


def build_compute_metrics_fn(
    task_name: str, tokenizer: PreTrainedTokenizer
) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    def storycompletion_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    if "summarization" in task_name:
        compute_metrics_fn = summarization_metrics
    elif "translation" in task_name:
        compute_metrics_fn = translation_metrics
    elif "storycompletion" in task_name:
        compute_metrics_fn = storycompletion_metrics

    return compute_metrics_fn
