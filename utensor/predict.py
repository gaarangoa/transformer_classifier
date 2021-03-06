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


def evaluate(inp_sentence, params, tokenizer_source, tokenizer_target, transformer):
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
        inp, None, False, enc_padding_mask, None, None
    )

    predictions = tf.squeeze(predictions, axis=0)
    predictions_index = tf.cast(
        tf.argsort(predictions, axis=-1, direction="DESCENDING"), tf.int32
    )

    predictions = predictions.numpy()[predictions_index.numpy()]

    _pred = [
        {"score": float(i), "label": tokenizer_target.int2str(j.numpy())}
        for i, j in zip(predictions, predictions_index)
    ][: params["max_predictions"]]

    return _pred, attention_weights


def translate(sentence, params, tokenizer_source, tokenizer_target, transformer):
    predictions, attention = evaluate(
        sentence, params, tokenizer_source, tokenizer_target, transformer
    )

    return predictions, attention
