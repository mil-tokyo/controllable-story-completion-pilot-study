#! /usr/bin/env python
# coding=utf-8

# This is the modified version of run_pplm.py, originally written by Uber Technologies, Inc. (2019)
#
# Copyright (c) 2022 Machine Intelligence Laboratory (The University of Tokyo)
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

import datetime
import pytz

from example import example_story

from pplm_classification_head import ClassificationHead

from transformers.file_utils import cached_path

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from get_words_from_NRC_VAD_lexicon import (
    data_from_NRC_VAD_lexicon,
    NRC_VAD_lexicon_PATH,
    get_bag_of_words_indices,
)

from ROCStories_utils import ROCStories_fixed_missing_sentence_mask

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10


# PPLM in HuggingFace transformers library and the original repository (uber-research)
# have different BAG_OF_WORDS_ARCHIVE_MAP
#
# https://github.com/huggingface/transformers/blob/v4.1.1/examples/research_projects/pplm/run_pplm.py
# https://github.com/uber-research/PPLM/blob/5b262d6b625fae063e085a1f59aa40b7c7854fb5/run_pplm.py
#

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

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(
                logits < batch_mins, torch.ones_like(logits) * 0.0, logits
            )
        return torch.where(
            logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits
        )


def perturb_past(
    past_key_values,  # renamed from past to make it similar to transformers in v4.1.1
    model,
    last,
    output_so_far,
    encoder_hidden_states_using_context=None,  # for Seq2SeqLM
    encoder_attention_mask=None,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    one_hot_bows_vectors=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
):

    if model.config.is_encoder_decoder:  # Seq2SeqLM
        core_model = model.model if hasattr(model, "model") else model

    if not model.config.is_encoder_decoder:  # CausalLM
        grad_accumulator = [
            (
                (np.zeros(p[0].shape).astype("float32")),
                (np.zeros(p[1].shape).astype("float32")),
            )
            for p in past_key_values
        ]

        rep_past_t = past_key_values[0][0]
        past_shape = rep_past_t.shape

        curr_length = past_key_values[0][0].shape[2]

    elif model.config.is_encoder_decoder:  # Seq2SeqLM
        self_attn_past_key_values = tuple(
            [past_key_value[:2] for past_key_value in past_key_values]
        )
        cross_attn_past_key_values = tuple(
            [past_key_value[-2:] for past_key_value in past_key_values]
        )

        grad_accumulator = [
            (
                (np.zeros(p[0].shape).astype("float32")),
                (np.zeros(p[1].shape).astype("float32")),
            )
            for p in self_attn_past_key_values
        ]  # p.shape = (1, 16, length, 64) for bart-large

        rep_past_t = self_attn_past_key_values[0][0]
        past_shape = rep_past_t.shape

        curr_length = past_key_values[0][0].shape[2]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[
            1:
        ].unsqueeze(-1)
    else:
        decay_mask = 1.0

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
            tuple(past_shape[:-2]) + tuple([window_length]) + tuple(past_shape[-1:])
        )

        zeros_key_val_shape = (
            tuple(past_shape[:-2])
            + tuple([curr_length - window_length])
            + tuple(past_shape[-1:])
        )

        ones_mask = decay_mask * torch.ones(ones_key_val_shape)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(rep_past_t).to(device)

    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        print("Iteration ", i + 1)

        if not model.config.is_encoder_decoder:  # CausalLM

            curr_perturbation = tuple(
                [
                    tuple(
                        [
                            torch.from_numpy(p_).requires_grad_(True).to(device=device)
                            for p_ in g
                        ]
                    )
                    for g in grad_accumulator
                ]
            )

            for cp_ in curr_perturbation:
                for p_ in cp_:
                    p_.retain_grad()

            perturbed_past = tuple(
                [
                    tuple(map(add, past_key_value, cp_))
                    for past_key_value, cp_ in zip(past_key_values, curr_perturbation)
                ]
            )

            curr_length = curr_perturbation[0][0].shape[2]

            all_outputs = model(
                input_ids=last,
                past_key_values=perturbed_past,
                use_cache=True,
                return_dict=True,
            )
            all_logits = all_outputs.get("logits")
            all_hidden = all_outputs.get("hidden_states")
            hidden = all_hidden[-1]

        else:  # Seq2SeqLM
            curr_perturbation = tuple(
                [
                    tuple(
                        [
                            torch.from_numpy(p_).requires_grad_(True).to(device=device)
                            for p_ in g
                        ]
                    )
                    for g in grad_accumulator
                ]
            )
            for cp_ in curr_perturbation:
                for p_ in cp_:
                    p_.retain_grad()

            perturbed_self_attn_past_key_values = tuple(
                [
                    tuple(map(add, self_attn_past_key_value, cp_))
                    for self_attn_past_key_value, cp_ in zip(
                        self_attn_past_key_values, curr_perturbation
                    )
                ]
            )
            _, _, curr_length, _ = curr_perturbation[0][0].shape

            perturbed_past_key_values = tuple(
                [
                    self_attn + cross_attn
                    for self_attn, cross_attn in zip(
                        perturbed_self_attn_past_key_values, cross_attn_past_key_values
                    )
                ]
            )

            all_outputs = core_model.decoder(
                encoder_hidden_states=encoder_hidden_states_using_context,
                input_ids=output_so_far,
                past_key_values=perturbed_past_key_values,  ### Tuple[Tuple[torch.Tensor]]
                encoder_attention_mask=encoder_attention_mask,
                attention_mask=None,
                return_dict=True,
                use_cache=True,
            )
            last_hidden = all_outputs.get("last_hidden_state")

            if model.config.model_type == "t5":
                all_logits = model.lm_head(last_hidden)
            else:
                all_logits = F.linear(
                    last_hidden, core_model.shared.weight, bias=model.final_logits_bias
                )

            hidden = last_hidden

        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        # Original code's comment
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)

        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits) + 1e-7)
                loss += bow_loss
                loss_list.append(bow_loss)
            print(" pplm_bow_loss:", loss.data.cpu().numpy())

        # if loss_type == 2 or loss_type == 3:
        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()

            # Original code's comment
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)

            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)

            try:
                wte = model.resize_token_embeddings()
            except:
                wte = model.shared

            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)

                if not model.config.is_encoder_decoder:  # CausalLM
                    curr_outputs = model(
                        past_key_values=curr_unpert_past,
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        return_dict=True,
                    )
                    curr_unpert_past = curr_outputs.get("past_key_values")
                    curr_all_hidden = curr_outputs.get("hidden_states")
                    curr_hidden = curr_all_hidden[-1]

                else:  # Seq2SeqLM
                    curr_outputs = core_model.decoder(
                        encoder_hidden_states=encoder_hidden_states_using_context,
                        past_key_values=curr_unpert_past,
                        inputs_embeds=inputs_embeds,
                        encoder_attention_mask=encoder_padding_mask,
                        attention_mask=None,
                        return_dict=True,
                        use_cache=True,
                    )
                    curr_unpert_past = curr_outputs.get("past_key_values")
                    curr_last_hidden = curr_outputs.get("last_hidden_state")
                    curr_hidden = curr_last_hidden

                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1
                )

            prediction = classifier(
                new_accumulated_hidden / (curr_length + 1 + horizon_length)
            )

            label = torch.tensor(
                prediction.shape[0] * [class_label], device=device, dtype=torch.long
            )
            discrim_loss = ce_loss(prediction, label)
            print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                unpert_probs
                + SMALL_CONST
                * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = (
                SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            )
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            print(" kl_loss", kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        print(" pplm_loss", (loss - kl_loss).data.cpu().numpy())

        loss.backward(retain_graph=True)

        if not model.config.is_encoder_decoder:  # CausalLM

            if grad_norms is not None and loss_type == PPLM_BOW:
                grad_norms = [
                    [
                        torch.max(
                            grad_norms[index][index_2],
                            torch.norm(p_.grad * window_mask),
                        )
                        for index_2, p_ in enumerate(ps_)
                    ]
                    for index, ps_ in enumerate(curr_perturbation)
                ]
            else:
                grad_norms = [
                    [(torch.norm(p_.grad * window_mask) + SMALL_CONST) for p_ in ps_]
                    for ps_ in curr_perturbation
                ]

            grad = [
                [
                    -stepsize
                    * (p_.grad * window_mask / grad_norms[index][index_2] ** gamma)
                    .data.cpu()
                    .numpy()
                    for index_2, p_ in enumerate(ps_)
                ]
                for index, ps_ in enumerate(curr_perturbation)
            ]

            grad_accumulator = [
                list(map(add, g_, ga_)) for g_, ga_ in zip(grad, grad_accumulator)
            ]

            for cp_ in curr_perturbation:
                for p_ in cp_:
                    p_.grad.data.zero_()

            """
            The old implementation of gpt2 in HuggingFace Transformers wrote `past_key_value` in `past_key_values` using `torch.tensor` with each tensor of 
            shape (2, batch_size, num_heads, sequence_length, embed_size_per_head)).
            
            In new implementation of gpt2, as Bart, `past_key_value` in `past_key_values` is `Tuple[torch.Tensor]` and 
            self_atten part is not a tensor but a "2 tensors of shape (batch_size, num_heads, sequence_length - 1, embed_size_per_head))".
            """

            past_key_values = list(past_key_values)

            for i, past_key_value in enumerate(past_key_values):
                new_p_ = [p_.detach() for p_ in past_key_value]
                past_key_values[i] = new_p_

            past_key_values = tuple(past_key_values)

        else:  # Seq2SeqLM
            if grad_norms is not None and loss_type == PPLM_BOW:
                grad_norms = [
                    [
                        torch.max(
                            grad_norms[index][index_2],
                            torch.norm(p_.grad * window_mask),
                        )
                        for index_2, p_ in enumerate(ps_)
                    ]
                    for index, ps_ in enumerate(curr_perturbation)
                ]
            else:
                grad_norms = [
                    [(torch.norm(p_.grad * window_mask) + SMALL_CONST) for p_ in ps_]
                    for ps_ in curr_perturbation
                ]

            grad = [
                [
                    -stepsize
                    * (p_.grad * window_mask / grad_norms[index][index_2] ** gamma)
                    .data.cpu()
                    .numpy()
                    for index_2, p_ in enumerate(ps_)
                ]
                for index, ps_ in enumerate(curr_perturbation)
            ]

            grad_accumulator = [
                list(map(add, g_, ga_)) for g_, ga_ in zip(grad, grad_accumulator)
            ]

            for cp_ in curr_perturbation:
                for p_ in cp_:
                    p_.grad.data.zero_()

            self_attn_past_key_values = list(self_attn_past_key_values)

            for i, self_attn_past_key_value in enumerate(self_attn_past_key_values):
                new_p_ = [p_.detach() for p_ in self_attn_past_key_value]
                self_attn_past_key_values[i] = new_p_

            self_attn_past_key_values = tuple(self_attn_past_key_values)

            past_key_values = list(past_key_values)

            past_key_values[:2] = self_attn_past_key_values
            past_key_values[-2:] = cross_attn_past_key_values

            past_key_values = tuple(past_key_values)

    if not model.config.is_encoder_decoder:  # CausalLM

        grad_accumulator = [
            [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in ga_]
            for ga_ in grad_accumulator
        ]
        pert_past_key_values = tuple(
            [
                tuple(map(add, sa_p_, ga_))
                for sa_p_, ga_ in zip(past_key_values, grad_accumulator)
            ]
        )

    else:  # Seq2SeqLM
        grad_accumulator = [
            [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in ga_]
            for ga_ in grad_accumulator
        ]
        pert_self_attn_past_key_values = tuple(
            [
                tuple(map(add, sa_p_, ga_))
                for sa_p_, ga_ in zip(self_attn_past_key_values, grad_accumulator)
            ]
        )
        pert_past_key_values = tuple(
            [
                self_attn + cross_attn
                for self_attn, cross_attn in zip(
                    pert_self_attn_past_key_values, cross_attn_past_key_values
                )
            ]
        )

    return pert_past_key_values, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
    name: Optional[str], class_label: Union[str, int], device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params["class_size"], embed_size=params["embed_size"]
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError(
            "Either url or path have to be specified in the discriminator model parameters"
        )
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


# `get_bag_of_words_indices` is written in "get_words_from_NRC_VAD_lexicon.py"


def build_bows_one_hot_vectors(bow_indices, tokenizer, device="cuda"):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        # get_bag_of_word_indices の tokenizer.encode で special token が付与されないよう、
        # `get_bag_of_words_indices` において `add_special_tokens=False` とした。
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        single_bow.to(torch.int64)

        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, len(tokenizer)).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
    model,
    tokenizer,
    context=None,
    num_samples=1,
    device="cuda",
    bag_of_words=None,
    discrim=None,
    class_label=None,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
    use_NRC_VAD_lexicon=True,
    V_min=0.0,
    V_max=1.0,
    A_min=0.0,
    A_max=1.0,
    D_min=0.0,
    D_max=1.0,
    **kwargs,
):
    classifier, class_id = get_classifier(discrim, class_label, device)

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(
            bag_of_words.split(";"),
            tokenizer,
            use_NRC_VAD_lexicon=use_NRC_VAD_lexicon,
            NRC_VAD_lexicon_instance=None,
            V_min=V_min,
            V_max=V_max,
            A_min=A_min,
            A_max=A_max,
            D_min=D_min,
            D_max=D_max,
        )

    if bag_of_words and classifier:
        print("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")
        loss_type = PPLM_BOW_DISCRIM

    elif bag_of_words:
        loss_type = PPLM_BOW
        print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        repetition_penalty=repetition_penalty,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            repetition_penalty=repetition_penalty,
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == "cuda":
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
    model,
    tokenizer,
    context=None,
    past=None,
    device="cuda",
    perturb=True,
    bow_indices=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
    use_NRC_VAD_lexicon=True,
    V_min=0.0,
    V_max=1.0,
    A_min=0.0,
    A_max=1.0,
    D_min=0.0,
    D_max=1.0,
):
    output_so_far = None
    encoder_hidden_states_using_context = encoder_padding_mask = None

    if model.config.is_encoder_decoder:  # Seq2SeqLM
        core_model = model.model if hasattr(model, "model") else model

    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)

        if not model.config.is_encoder_decoder:  # CausalLM
            output_so_far = context_t

        elif model.config.is_encoder_decoder:  # Seq2SeqLM
            output_so_far = torch.full(
                (1, 1),
                model.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )

            encoder_hidden_states_using_context = core_model.encoder(context_t)[0]
            encoder_padding_mask = context_t.new_ones(
                context_t.shape
            )  ## encoderの出力はどれもマスクしないということ

    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    for i in trange(length, ascii=True):

        if not model.config.is_encoder_decoder:  # CausalLM

            if past is None and output_so_far is not None:
                last = output_so_far[:, -1:]
                if output_so_far.shape[1] > 1:
                    past = model(
                        output_so_far[:, :-1], return_dict=True, use_cache=True
                    ).get("past_key_values")

            unpert_outputs = model(output_so_far, return_dict=True, use_cache=True)

        elif model.config.is_encoder_decoder:  # Seq2SeqLM

            if past is None and output_so_far is not None:
                last = output_so_far[:, -1:]

                if output_so_far.shape[1] > 1:
                    past = core_model.decoder(
                        input_ids=output_so_far[:, :-1],
                        encoder_hidden_states=encoder_hidden_states_using_context,
                        return_dict=True,
                        use_cache=True,
                    ).get("past_key_values")

            if model.config.model_type == "t5":
                unpert_outputs = core_model.decoder(
                    input_ids=output_so_far,
                    encoder_hidden_states=encoder_hidden_states_using_context,
                    return_dict=True,
                    use_cache=True,
                )

            else:
                unpert_outputs = core_model.decoder(
                    input_ids=output_so_far,
                    encoder_hidden_states=encoder_hidden_states_using_context,
                    encoder_attention_mask=encoder_padding_mask,
                    attention_mask=None,  ## 生成時は常にNone
                    return_dict=True,
                    use_cache=True,
                )

        if not model.config.is_encoder_decoder:  # CausalLM
            unpert_logits = unpert_outputs.get("logits")
            unpert_past = unpert_outputs.get("past_key_values")
            unpert_all_hidden = unpert_outputs.get("hidden_states")

            unpert_last_hidden = unpert_all_hidden[-1]

        elif model.config.is_encoder_decoder:  # Seq2SeqLM
            unpert_past = unpert_outputs.get("past_key_values")
            unpert_last_hidden = unpert_outputs.get("last_hidden_state")

            if model.config.model_type == "t5":
                unpert_logits = model.lm_head(unpert_last_hidden)
            else:
                unpert_logits = F.linear(
                    unpert_last_hidden,
                    core_model.shared.weight,
                    bias=model.final_logits_bias,
                )

        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    output_so_far,
                    encoder_hidden_states_using_context=encoder_hidden_states_using_context,
                    encoder_attention_mask=encoder_padding_mask,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        if not model.config.is_encoder_decoder:  # CausalLM
            pert_outputs = model(
                last, past_key_values=pert_past, return_dict=True, use_cache=True
            )
            pert_logits = pert_outputs.get("logits")
            past = pert_outputs.get("past_key_values")
            _ = pert_outputs.get("hidden_states")

        elif model.config.is_encoder_decoder:  # Seq2SeqLM
            if model.config.model_type == "t5":
                pert_outputs = core_model.decoder(
                    # input_ids=last,
                    input_ids=output_so_far,
                    encoder_hidden_states=encoder_hidden_states_using_context,
                    past_key_values=pert_past,
                    return_dict=True,
                    use_cache=True,
                )
            else:
                pert_outputs = core_model.decoder(
                    # input_ids=last,
                    input_ids=output_so_far,
                    encoder_hidden_states=encoder_hidden_states_using_context,
                    past_key_values=pert_past,
                    encoder_attention_mask=encoder_padding_mask,
                    attention_mask=None,
                    return_dict=True,
                    use_cache=True,
                )

            past = pert_outputs.get("past_key_values")
            pert_last_hidden = pert_outputs.get("last_hidden_state")

            if model.config.model_type == "t5":
                pert_logits = model.lm_head(pert_last_hidden)
            else:
                pert_logits = F.linear(
                    pert_last_hidden,
                    core_model.shared.weight,
                    bias=model.final_logits_bias,
                )

        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST

        for token_idx in set(output_so_far[0].tolist()):
            if pert_logits[0, token_idx] < 0:
                pert_logits[0, token_idx] *= repetition_penalty
            else:
                pert_logits[0, token_idx] /= repetition_penalty

        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device, dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            print("unperturbed discrim loss", unpert_discrim_loss.data.cpu().numpy())
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs ** gm_scale) * (
                unpert_probs ** (1 - gm_scale)
            )  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            try:
                last = torch.multinomial(pert_probs, num_samples=1)
            except:
                # `nan` in t5
                pert_probs[pert_probs != pert_probs] = 0
                last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        )

        print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError(
            "When using a generic discriminator, discrim_weights need to be specified"
        )
    if discrim_meta is None:
        raise ValueError(
            "When using a generic discriminator, discrim_meta need to be specified"
        )

    with open(discrim_meta, "r") as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta["path"] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS["generic"] = meta


def run_pplm_example(
    pretrained_model=None,
    cond_text="",
    uncond=False,
    num_samples=1,
    bag_of_words=None,
    discrim=None,
    discrim_weights=None,
    discrim_meta=None,
    class_label=-1,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    seed=0,
    no_cuda=False,
    colorama=False,
    repetition_penalty=1.0,
    use_NRC_VAD_lexicon=True,
    V_min=0.0,
    V_max=1.0,
    A_min=0.0,
    A_max=1.0,
    D_min=0.0,
    D_max=1.0,
    output_dir="./tmp/",
    **kwargs,
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == "generic":
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim]["pretrained_model"]
        print(
            "discrim = {}, pretrained_model set to discriminator's = {}".format(
                discrim, pretrained_model
            )
        )

    # load pretrained model

    # `PreTrainedModel` class has `self.config.is_encoder_decoder`.

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model, output_hidden_states=True
        )
        assert model.config.is_encoder_decoder == True
        print(f"Seq2SeqLM mode, model type:{model.config.model_type}")
    except AssertionError:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model, output_hidden_states=True
        )
        assert model.config.is_encoder_decoder == False
        print(f"CLM mode, model type:{model.config.model_type}")

    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # update `self.add_prefix_space` in the tokenizer
    tokenizer.add_prefix_space = False

    print(f"tokenizer: {model.config.model_type}")

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode([tokenizer.bos_token])
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        if model.config.is_encoder_decoder == False:  # CausalLM
            tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + raw_text)
        elif model.config.is_encoder_decoder == True:  # Seq2SeqLM
            tokenized_cond_text = tokenizer.encode(raw_text)

    if model.config.is_encoder_decoder == False:  # CausalLM
        print("= Prefix of sentence =")
        print(tokenizer.decode(tokenized_cond_text))
        print()

    if model.config.is_encoder_decoder == True:  # Seq2SeqLM
        print("= Inputs to the encoder =")
        print(tokenizer.decode(tokenized_cond_text))
        print()

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        repetition_penalty=repetition_penalty,
        use_NRC_VAD_lexicon=use_NRC_VAD_lexicon,
        V_min=V_min,
        V_max=V_max,
        A_min=A_min,
        A_max=A_max,
        D_min=D_min,
        D_max=D_max,
    )

    results_dict = {}

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    unpert_gen_text_skip_special = tokenizer.decode(
        unpert_gen_tok_text.tolist()[0], skip_special_tokens=True
    )

    print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    results_dict["Unperturbed"] = unpert_gen_text_skip_special

    generated_texts = []

    bow_word_ids = set()
    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices(
            bag_of_words.split(";"),
            tokenizer,
            use_NRC_VAD_lexicon=use_NRC_VAD_lexicon,
            NRC_VAD_lexicon_instance=None,
            V_min=V_min,
            V_max=V_max,
            A_min=A_min,
            A_max=A_max,
            D_min=D_min,
            D_max=D_max,
        )
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts

    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if colorama:
                import colorama

                pert_gen_text = ""
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += "{}{}{}".format(
                            colorama.Fore.RED,
                            tokenizer.decode([word_id]),
                            colorama.Style.RESET_ALL,
                        )
                    else:
                        pert_gen_text += tokenizer.decode([word_id])
            else:
                pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

            print("= Perturbed generated text {} =".format(i + 1))
            print(pert_gen_text)
            print()
        except Exception as exc:
            print("Ignoring error while generating perturbed text:", exc)

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

        if output_dir is not None:
            pert_gen_text = tokenizer.decode(
                pert_gen_tok_text.tolist()[0], skip_special_tokens=True
            )

            results_dict[f"Perturbed_{i}"] = pert_gen_text

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="facebook/bart-base",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default=example_story["context"], help="Input Story",
    )
    parser.add_argument(
        "--uncond", action="store_true", help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help=(
            "Bags of words used for PPLM-BoW. "
            "Either a BOW id (see list in code) or a filepath. "
            "Multiple BoWs separated by ;"
        ),
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument(
        "--discrim_weights",
        type=str,
        default=None,
        help="Weights for the generic discriminator",
    )
    parser.add_argument(
        "--discrim_meta",
        type=str,
        default=None,
        help="Meta information for the generic discriminator",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate by random sampling (instead of greedy)",
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; 0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true", help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true", help="colors keywords")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalize repetition. More than 1.0 -> less repetition",
    )

    parser.add_argument("--use_NRC_VAD_lexicon", action="store_true"),
    parser.add_argument("--V_min", type=float, default=0.0)
    parser.add_argument("--V_max", type=float, default=1.0)
    parser.add_argument("--A_min", type=float, default=0.0)
    parser.add_argument("--A_max", type=float, default=1.0)
    parser.add_argument("--D_min", type=float, default=0.0)
    parser.add_argument("--D_max", type=float, default=1.0)

    parser.add_argument(
        "--output_dir", type=str, default="./tmp/"
    )  # to save the output

    parser.add_argument(
        "--multiple_trial", type=int, default=1
    )  # to run multiple times

    parser.add_argument("--eval", action="store_true", help="Use validation set")
    parser.add_argument("--test", action="store_true", help="Use test set")

    args = parser.parse_args()

    if not args.bag_of_words and args.use_NRC_VAD_lexicon:
        args.bag_of_words = "nrcvad"

    if args.multiple_trial == 1:
        save_dir = os.path.join(
            args.output_dir,
            os.path.basename(args.pretrained_model)
            + "_"
            + datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime(
                "%Y%m%d-%H%M%S"
            ),
        )
        print(f"save_dir: {save_dir}")
        args.output_dir = save_dir

        # mkdir for save
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, "params.json"), mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

        results_dict = run_pplm_example(**vars(args))

        print("=" * 80)
        print("= Input Context = ")
        print(args.cond_text)

        results_dict["context"] = args.cond_text
        results_dict["target_sentence"] = example_story["target_sentence"]

        if save_dir is not None:
            with open(os.path.join(save_dir, "pplm_result.json"), "w") as f:
                json.dump(results_dict, f, indent=4)

    elif args.multiple_trial >= 2:

        assert not (
            args.eval == True and args.test == True
        ), "Please use either --eval or --test"
        assert (
            args.eval == True or args.test == True
        ), "Please use either --eval or --test"

        if args.eval:
            print("Use validation dataset")
            dataset = ROCStories_fixed_missing_sentence_mask(
                data_path="../data/rocstories_completion_dev.csv"
            )
        elif args.test:
            print("Use test dataset")
            dataset = ROCStories_fixed_missing_sentence_mask(
                data_path="../data/rocstories_completion_test.csv"
            )
        else:
            dataset = None

        save_dir_base = os.path.join(
            args.output_dir,
            os.path.basename(args.pretrained_model)
            + "_"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        if not os.path.exists(save_dir_base):
            os.makedirs(save_dir_base)

        with open(os.path.join(save_dir_base, "params.json"), mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

        np.random.seed(args.seed)
        random_indices = np.random.randint(low=0, high=len(dataset), size=5)

        for trial_id, rnd_idx in zip(range(args.multiple_trial), random_indices):
            rnd_idx = rnd_idx.item()
            if dataset is not None:
                dataset_instance = dataset[rnd_idx]
                args.cond_text = dataset_instance["context"]
                story_id = dataset_instance["storyid"]

                print(f"story_id: {story_id}")
                print(f"cond_text: {args.cond_text}")

            save_dir = os.path.join(
                save_dir_base, f"trial-{trial_id}_" + f"storyid-{story_id}"
            )
            print(f"save_dir: {save_dir}")
            args.output_dir = save_dir

            # mkdir for save
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            results_dict = run_pplm_example(**vars(args))
            results_dict["context"] = dataset_instance["context"]
            results_dict["target_sentence"] = dataset_instance["target_sentence"]

            if save_dir is not None:
                with open(os.path.join(save_dir, "pplm_result.json"), "w") as f:
                    json.dump(results_dict, f, indent=4)

            print("=" * 80)
            print("= Input Context = ")
            print(args.cond_text)
