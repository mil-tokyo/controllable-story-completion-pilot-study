# Plug-and-Play Controller for Story Completion (Pilot Study)

This repository is for our work presented at [In2Writing](https://in2writing.glitch.me/) (The First Workshop on Intelligent and Interactive Writing Assistants).

## Overview

This folder contains our modified version of the Plug and Play Language Model (PPLM) codes and document (this "README.md").
Our implementation is based on the PPLM codes distributed in <https://github.com/huggingface/transformers> and in <https://github.com/uber-research/PPLM>, under Apache License 2.0.

Note: in the Transformers library, the PPLM codes were placed in <https://github.com/huggingface/transformers/tree/master/examples/text-generation/pplm> previously. Now it is replaced in <https://github.com/huggingface/transformers/tree/main/examples/research_projects>.

## How to use our modified PPLM

### Seq2SeqLM Story Completion

In `seq2seqlm_storycompletion` directory, 

```
python storycompletion_finetune_trainer.py \
    --data_dir ../data/ \
    --learning_rate=3e-5 \
    --do_train --do_eval \
    --predict_with_generate \
    --logging_steps 3000 \
    --save_steps 3000 \
    --model_name_or_path facebook/bart-base \
    --save_total_limit 3 \
    --use_task_specific_params \
    --output_dir /path/to/output/seq2seqlm/
```

### BART + PPLM-BoW for controlling emotions

In `modified_pplm` directory, 

```sh
python run_pplm_various_lm.py \
     -M /path/to/output/seq2seqlm/ \
     --length 40 \
     --gamma 1.0 \
     --num_iterations 3 \
     --num_samples 3 \
     --stepsize 0.015 \
     --window_length 5 \
     --kl_scale 0.30 \
     --gm_scale 0.95 \
     --colorama \
     --sample \
     --use_NRC_VAD_lexicon \
     --V_max 0.3 --A_min 0.7 \
     --output_dir /path/to/output/pplm/
```

#### Note

The results presented in our paper were obtained using version 4.8.1 of HuggingFace Transformers in Seq2SeqLM finetuning.
We use one NVIDIA V100 GPU for each training and each inference.

## Citation

If you find this repository helpful for your work, please consider citing the related paper.

> Yusuke Mori, Hiroaki Yamane, Ryohei Shimizu, Tatsuya Harada, “Plug-and-Play Controller for Story Completion: A Pilot Study toward Emotion-aware Story Writing Assistance,” The First Workshop on Intelligent and Interactive Writing Assistants (In2Writing) (ACL 2022, Workshop), 2022.

Please visit the ACL Anthology to get [Bibtex file](https://aclanthology.org/2022.in2writing-1.6/).

### Original PPLM

We also recommend you to cite (Dathathri et al., 2020), because our work is based on it.

> Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. (2020). _Plug and Play Language Models: a Simple Approach to Controlled Text Generation._ International Conference on Learning Representations (ICLR) 2020. [link to ICLR page](https://iclr.cc/virtual_2020/poster_H1edEyBKDS.html)
