import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import numpy as np
from scipy.special import softmax


VOCAB_PATH = (
    "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt"
)


def init_bem_model():
    os.environ["TFHUB_CACHE_DIR"] = "/home/wallat/.cache/tfhub_modules"
    vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=VOCAB_PATH,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        ),
        num_oov_buckets=1,
    )
    cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(["[CLS]", "[SEP]"]))
    tokenizer = text.BertTokenizer(
        vocab_lookup_table=vocab_table,
        token_out_type=tf.int64,
        preserve_unused_token=True,
        lower_case=True,
    )

    bem = hub.load("https://tfhub.dev/google/answer_equivalence/bem/1")

    return bem, tokenizer, cls_id, sep_id


def predict(bem, question, reference, candidate, tokenizer, cls_id, sep_id):
    inputs = _bertify_examples(
        [{"question": question, "reference": reference, "candidate": candidate}],
        tokenizer,
        cls_id,
        sep_id,
    )

    # The outputs are raw logits.
    raw_outputs = bem(inputs)

    # They can be transformed into a classification 'probability' like so:
    bem_score = float(softmax(np.squeeze(raw_outputs))[1])

    # print(f"BEM score: {bem_score}")
    return bem_score


def _bertify_example(example, tokenizer, cls_id, sep_id):
    question = tokenizer.tokenize(example["question"]).merge_dims(1, 2)
    reference = tokenizer.tokenize(example["reference"]).merge_dims(1, 2)
    candidate = tokenizer.tokenize(example["candidate"]).merge_dims(1, 2)

    input_ids, segment_ids = text.combine_segments(
        (candidate, reference, question), cls_id, sep_id
    )

    return {"input_ids": input_ids.numpy(), "segment_ids": segment_ids.numpy()}


def _pad(a, length=512):
    return np.append(a, np.zeros(length - a.shape[-1], np.int32))


def _bertify_examples(examples, tokenizer, cls_id, sep_id):
    input_ids = []
    segment_ids = []
    for example in examples:
        example_inputs = _bertify_example(example, tokenizer, cls_id, sep_id)
        input_ids.append(_pad(example_inputs["input_ids"]))
        segment_ids.append(_pad(example_inputs["segment_ids"]))

    return {"input_ids": np.stack(input_ids), "segment_ids": np.stack(segment_ids)}
