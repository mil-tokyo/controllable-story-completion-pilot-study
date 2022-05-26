import os
import sys
import json
import argparse

from collections import OrderedDict

from datasets import load_metric

import bert_score
from bleurt import score as bleurt_score

sys.path.append("../seq2seqlm_storycompletion/")
from ROCStories_utils import ROCStories_fixed_missing_sentence_mask

# hide the loading messages for bert_score
import logging
import transformers

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

import tensorflow as tf

try:
    tf.compat.v1.flags.DEFINE_string("ip", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("stdin", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("control", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("hb", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("Session.signature_scheme", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("Session.key", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("shell", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("transport", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("iopub", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("f", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("path", "", "kernel")
except:
    pass
try:
    tf.compat.v1.flags.DEFINE_string("out_dir", "", "kernel")
except:
    pass


def run_eval(path):
    ## DATASET
    test_dataset = ROCStories_fixed_missing_sentence_mask(
        data_path="../data/rocstories_completion_test.csv"
    )

    with open(path, "r") as f:
        generated_sentences = f.read().strip().split("\n")

    references_for_bleu = [
        [test_dataset[i].get("target_sentence")] for i in range(len(test_dataset))
    ]
    references = [
        test_dataset[i].get("target_sentence") for i in range(len(test_dataset))
    ]

    results = OrderedDict()

    ## BLEU
    metric = load_metric("sacrebleu")
    print("Calc sacrebleu")

    """
    Returns:
    'score': BLEU score,
    'counts': Counts,
    'totals': Totals,
    'precisions': Precisions,
    'bp': Brevity penalty,
    'sys_len': predictions length,
    'ref_len': reference length,
    """

    metric.add_batch(predictions=generated_sentences, references=references_for_bleu)
    score = metric.compute()
    results[metric.name] = score

    ## ROUGE
    metric = load_metric("rouge")
    print("Calc rouge")

    """
    Returns:
    rouge1: rouge_1 (precision, recall, f1),
    rouge2: rouge_2 (precision, recall, f1),
    rougeL: rouge_l (precision, recall, f1),
    rougeLsum: rouge_lsum (precision, recall, f1)
    """

    metric.add_batch(predictions=generated_sentences, references=references)
    score = metric.compute()
    results[metric.name] = score

    ## METEOR
    metric = load_metric("meteor")
    print("Calc meteor")

    metric.add_batch(predictions=generated_sentences, references=references)
    score = metric.compute()
    results[metric.name] = score

    ## BERTScore
    print("Calc BERTScore")

    P, R, F1 = bert_score.score(
        generated_sentences, references, lang="en", verbose=True
    )

    #
    # BERTScore uses average for model evaluation
    #
    # https://github.com/Tiiiger/bert_score/blob/3dc3e3d43af3c7ccc068d3a0ace744cee4c38a26/example/Demo.ipynb
    #

    bert_score_dict = OrderedDict()
    bert_score_dict["P"] = P.mean().item()
    bert_score_dict["R"] = R.mean().item()
    bert_score_dict["F1"] = F1.mean().item()

    results["bert_score"] = bert_score_dict

    ## BLEURT
    bleurt_checkpoint_base_dir = "/path/to/metric/BLEURT"
    bleurt_checkpoint_name = "bleurt-tiny-128"  # default

    bleurt_checkpoint = os.path.join(bleurt_checkpoint_base_dir, bleurt_checkpoint_name)

    bleurt_scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)
    bleurt_scores = bleurt_scorer.score(
        references=references, candidates=generated_sentences
    )

    results["BLEURT"] = bleurt_scores

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/path/to/result/predict/test_generations.txt",
        help="generated_sentences (txt)",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="where to save json")
    args = parser.parse_args()

    print(f"generated data path: {args.path}")
    results = run_eval(args.path)

    print(results)

    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.path)

    with open(os.path.join(args.out_dir, "multiple_metrics_results.json"), "w") as f:
        json.dump(results, f, indent=4)
