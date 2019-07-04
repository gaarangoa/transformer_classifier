import argparse
import tensorflow_datasets as tfds
import tensorflow as tf

from utensor.optimizer import CustomSchedule, loss_function
from utensor.dataset import Dataset
from utensor.model import Transformer
from utensor.dataset import load_dataset
from utensor.masking import create_masks
import pickle
from sklearn.metrics import classification_report
import time
import os
import json
import pickle
import numpy as np

from tqdm import tqdm

tf.keras.backend.clear_session()


def restore(params):
    # loading tokenizers for future predictions
    tokenizer_source = pickle.load(
        open(params["checkpoint_path"] + "tokenizer_source.pickle", "rb")
    )
    tokenizer_target = pickle.load(
        open(params["checkpoint_path"] + "tokenizer_target.pickle", "rb")
    )

    input_vocab_size = tokenizer_source.vocab_size + 2
    target_vocab_size = tokenizer_target.num_classes

    learning_rate = CustomSchedule(params["d_model"])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer = Transformer(
        params["num_layers"],
        params["d_model"],
        params["num_heads"],
        params["dff"],
        input_vocab_size,
        target_vocab_size,
        params["dropout_rate"],
    )

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, params["checkpoint_path"], max_to_keep=1
    )

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    else:
        print("Initializing from scratch.")

    return transformer, tokenizer_source, tokenizer_target


def evaluate(inp_sentence, bert_input, params):
    start_token = [tokenizer_source.vocab_size]
    end_token = [tokenizer_source.vocab_size + 1]

    inp = [start_token + tokenizer_source.encode(inp_sentence) + end_token]
    inp = tf.keras.preprocessing.sequence.pad_sequences(
        inp, maxlen=params["MAX_LENGTH"], padding="post"
    )
    # inp = tf.expand_dims(inp, 0)
    enc_padding_mask = create_masks(inp, None)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, _, attention_weights = transformer(
        inp, None, bert_input, False, enc_padding_mask, None, None
    )

    predictions = tf.squeeze(predictions, axis=0)
    predictions_index = tf.cast(
        tf.argsort(predictions, axis=-1, direction="DESCENDING"), tf.int32
    )

    predictions = predictions.numpy()[predictions_index.numpy()]

    _pred = [
        {"score": i, "label": tokenizer_target.int2str(j.numpy())}
        for i, j in zip(predictions, predictions_index)
    ][: params["max_predictions"]]

    return _pred, attention_weights


def translate(sentence, bert_input, params):
    predictions, w = evaluate(sentence, bert_input, params)
    # print("Input: {}".format(sentence))
    # print("predictions: {}".format(predictions))

    return {"Input": sentence, "pred": predictions, 'weight': w}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--max_predictions", type=int,
                        default=5, required=False)
    parser.add_argument("--inline", type=bool, default=False, required=False)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    params = json.load(open(args.checkpoint_path + "/params.json"))
    params["checkpoint_path"] = args.checkpoint_path
    params["input_file"] = args.input_file
    params["max_predictions"] = args.max_predictions
    transformer, tokenizer_source, tokenizer_target = restore(params)

    if args.inline:
        while 1:
            sentence = input()
            translate(sentence, [],  params)

    results = []

    for i in tqdm(open(args.input_file)):
        text, bert_input, label = i.strip().split('\t')
        bert_input = tf.Variable([[float(i) for i in bert_input.split(',')]])
        res = translate(text, bert_input, params)
        res['label'] = label
        results.append(res)

    pickle.dump(results, open(args.input_file+'.pred', 'wb'))
