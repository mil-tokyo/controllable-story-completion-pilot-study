import numpy as np

# example story for story completion (from our dev set)
example_story = {
    "context": "I got a call from the hospital. My doctor told me to stop everything I'm doing and come to her. Although I was nervous, I tried to drive calmly. <missing_sentence> The doctor diagnosed me with leukemia.",
    "target_sentence": "The front desk worker sent me to an office.",
    "missing_id": 3,
    "remain_sentences_list": np.array(
        [
            "I got a call from the hospital.",
            "My doctor told me to stop everything I'm doing and come to her.",
            "Although I was nervous, I tried to drive calmly.",
            "The doctor diagnosed me with leukemia.",
        ],
        dtype=object,
    ),
    "storyid": "703901fa-61ec-4a94-8118-aa3c840f3bda",
    "dataset_index": 0,
}
