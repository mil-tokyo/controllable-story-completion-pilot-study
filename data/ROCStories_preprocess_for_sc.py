#
# Preprocess "ROC Stories" for Story Completion, for using "special token"
#

import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATAPATH = "/path/to/ROCStories"
OUT_TRAIN_CSV = "./rocstories_completion_train.csv"
OUT_DEV_CSV = "./rocstories_completion_dev.csv"
OUT_TEST_CSV = "./rocstories_completion_test.csv"

# logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(f"load ROCStories from {DATAPATH}")

# random_seeds
split_random_state = 42
np_random_seed = 1234

logger.info(f"random_state for train/dev/test split: {split_random_state}")
logger.info(f"numpy random seed: {np_random_seed}")

np.random.seed(np_random_seed)

# load ROCStories

ROCstory_spring2016 = pd.read_csv(
    os.path.join(DATAPATH, "ROCStories__spring2016 - ROCStories_spring2016.csv")
)
ROCstory_winter2017 = pd.read_csv(
    os.path.join(DATAPATH, "ROCStories_winter2017 - ROCStories_winter2017.csv")
)

ROCstory_train = pd.concat([ROCstory_spring2016, ROCstory_winter2017])

logger.info(f"length of ROCStories: {len(ROCstory_train)}")

stories = ROCstory_train.loc[:, "sentence1":"sentence5"].values
stories_with_id = ROCstory_train.loc[
    :, ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5", "storyid"]
].values

# Split: Train, Dev, Test
train_and_dev, test_stories = train_test_split(
    ROCstory_train, test_size=0.1, random_state=split_random_state
)
train_stories, dev_stories = train_test_split(
    train_and_dev, test_size=1 / 9, random_state=split_random_state
)

logger.info("length of train, dev (valid), test sets")
logger.info([len(train_stories), len(dev_stories), len(test_stories)])

# dev
dev_missing_indexes = np.random.randint(low=0, high=5, size=len(dev_stories))

dev_stories_with_missing = []

for (_, st), mi in zip(dev_stories.iterrows(), dev_missing_indexes):
    sentences = st["sentence1":"sentence5"].values

    missing_sentence = sentences[mi]
    remain_sentences = np.delete(sentences, mi)

    dev_stories_with_missing.append(
        [
            remain_sentences[0],
            remain_sentences[1],
            remain_sentences[2],
            remain_sentences[3],
            mi,
            missing_sentence,
            st["storyid"],
            st["storytitle"],
        ]
    )


dev_df = pd.DataFrame(
    dev_stories_with_missing,
    columns=[
        "stories_with_missing_sentence1",
        "stories_with_missing_sentence2",
        "stories_with_missing_sentence3",
        "stories_with_missing_sentence4",
        "missing_id",
        "missing_sentence",
        "storyid",
        "storytitle",
    ],
)

dev_df.to_csv(OUT_DEV_CSV, index=False)
logger.info(f"dev (valid) csv is saved in {OUT_DEV_CSV}")

# test
test_missing_indexes = np.random.randint(low=0, high=5, size=len(test_stories))

test_stories_with_missing = []

for (_, st), mi in zip(test_stories.iterrows(), test_missing_indexes):
    sentences = st["sentence1":"sentence5"].values

    missing_sentence = sentences[mi]
    remain_sentences = np.delete(sentences, mi)

    test_stories_with_missing.append(
        [
            remain_sentences[0],
            remain_sentences[1],
            remain_sentences[2],
            remain_sentences[3],
            mi,
            missing_sentence,
            st["storyid"],
            st["storytitle"],
        ]
    )

test_df = pd.DataFrame(
    test_stories_with_missing,
    columns=[
        "stories_with_missing_sentence1",
        "stories_with_missing_sentence2",
        "stories_with_missing_sentence3",
        "stories_with_missing_sentence4",
        "missing_id",
        "missing_sentence",
        "storyid",
        "storytitle",
    ],
)

test_df.to_csv(OUT_TEST_CSV, index=False)
logger.info(f"test csv is saved in {OUT_TEST_CSV}")


# train
train_df = pd.DataFrame(
    train_stories,
    columns=[
        "sentence1",
        "sentence2",
        "sentence3",
        "sentence4",
        "sentence5",
        "storyid",
        "storytitle",
    ],
)

train_df.to_csv(OUT_TRAIN_CSV, index=False)
logger.info(f"train csv is saved in {OUT_TRAIN_CSV}")

# missing_id value counts
dev_missing_ids_count = dev_df.missing_id.value_counts()
test_missing_ids_count = test_df.missing_id.value_counts()

logger.info("missing_id value counts")
logger.info("train")
print("Randomly managed when training")
logger.info("dev (valid)")
print(dev_missing_ids_count)
logger.info("train")
print(test_missing_ids_count)
