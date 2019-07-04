import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import unicodedata
import re
import random
import argparse
import pickle

# Converts the unicode file to ascii


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w):

    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w


class Dataset:
    def __init__(
        self,
        filename="",
        vocab_dim=10000,
        max_length=40,
        buffer_size=20000,
        batch_size=64,
        num_classes=5,
        classes=[],
    ):
        self.input_filename = filename
        self.classes_filename = filename + ".classes"
        self.train_filename = filename + ".train"
        self.test_filename = filename + ".test"
        self.vocabulary_size = vocab_dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        self.classes = classes

    def build_train_test(self, test=0.2):
        """ This function takes the input file and splits into training and testing """

        labels = {}

        file_lines = []
        for ix, i in enumerate(open(self.input_filename)):
            i = i.strip().split("\t")
            labels[i[2]] = True
            file_lines.append(ix)

        random.shuffle(file_lines)
        test_limit = int(len(file_lines) * test)

        test_dict = {i: True for i in file_lines[0:test_limit]}

        fo_test = open(self.test_filename, "w")
        fo_train = open(self.train_filename, "w")
        for ix, i in enumerate(open(self.input_filename)):
            try:
                assert test_dict[ix]
                fo_test.write(i)
            except:
                fo_train.write(i)

        fo = open(self.classes_filename, "w")
        for i in labels:
            fo.write(i + "\n")
        fo.close()

    def format_train_test(self):
        filenames = [self.train_filename]
        train_dataset_tf = tf.data.Dataset.from_tensor_slices(filenames)
        train_dataset_tf = train_dataset_tf.flat_map(
            lambda filename: (tf.data.TextLineDataset(filename))
        )

        filenames = [self.test_filename]
        test_dataset_tf = tf.data.Dataset.from_tensor_slices(filenames)
        test_dataset_tf = test_dataset_tf.flat_map(
            lambda filename: (tf.data.TextLineDataset(filename))
        )

        return train_dataset_tf, test_dataset_tf

    def format_dataset(self):
        filenames = [self.train_filename, self.test_filename]
        test_dataset_tf = tf.data.Dataset.from_tensor_slices(filenames)
        test_dataset_tf = test_dataset_tf.flat_map(
            lambda filename: (tf.data.TextLineDataset(filename))
        )
        return test_dataset_tf

    def tokenizer(self, train_examples):
        self.tokenizer_source = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (
                preprocess_sentence(pt.numpy().decode(
                    "UTF-8").split("\t")[0]).encode()
                for pt in train_examples
            ),
            target_vocab_size=self.vocabulary_size,
        )

        self.tokenizer_target = tfds.features.ClassLabel(
            names_file=self.classes_filename
        )

        return self.tokenizer_source, self.tokenizer_target

    def encode(self, string):
        source = (
            [self.tokenizer_source.vocab_size]
            + self.tokenizer_source.encode(
                preprocess_sentence(
                    string.numpy().decode("UTF-8").split("\t")[0]
                ).encode()
            )
            + [self.tokenizer_source.vocab_size + 1]
        )

        target = self.tokenizer_target.encode_example(
            string.numpy().decode("UTF-8").split("\t")[2]
        )

        bert_embedding = [float(i) for i in string.numpy().decode(
            "UTF-8").split('\t')[1].split(',')]

        return source, target, bert_embedding

    def tf_encode(self, string):
        return tf.py_function(self.encode, [string], [tf.int64, tf.int64, tf.float32])

    def filter_max_length(self, x, y, z):
        return tf.logical_and(
            tf.size(x) <= self.max_length, tf.size(x) <= self.max_length
        )


def load_dataset(params={}):

    # setup parameters
    MAX_LENGTH = params["MAX_LENGTH"]
    BUFFER_SIZE = params["BUFFER_SIZE"]
    BATCH_SIZE = params["BATCH_SIZE"]
    vocab_dim = params["vocab_dim"]
    test_partition = params["test_partition"]
    dataset_file = params["dataset_file"]
    checkpoint_path = params["checkpoint_path"]
    retrain = params["retrain"]

    # Build the dataset for training validation
    dataset = Dataset(filename=dataset_file,
                      vocab_dim=vocab_dim, max_length=MAX_LENGTH)
    dataset.build_train_test(test=test_partition)
    train_examples, val_examples = dataset.format_train_test()
    full_dataset = dataset.format_dataset()

    if retrain:

        # loading tokenizers for future predictions
        with open(checkpoint_path + "/tokenizer_source.pickle", "rb") as handle:
            tokenizer_source = pickle.load(handle)

        with open(checkpoint_path + "/tokenizer_target.pickle", "rb") as handle:
            tokenizer_target = pickle.load(handle)

        # update dataset class with previous data
        dataset.tokenizer_source = tokenizer_source
        dataset.tokenizer_target = tokenizer_target

    else:
        tokenizer_source, tokenizer_target = dataset.tokenizer(full_dataset)

    train_dataset = train_examples.map(dataset.tf_encode)
    train_dataset = train_dataset.filter(dataset.filter_max_length)
    # train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=([MAX_LENGTH], [], [768])
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(dataset.tf_encode)
    val_dataset = val_dataset.filter(dataset.filter_max_length).padded_batch(
        BATCH_SIZE, padded_shapes=([MAX_LENGTH], [], [768])
    )
    # val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # saving tokenizers
    with open(checkpoint_path + "/tokenizer_source.pickle", "wb") as handle:
        pickle.dump(tokenizer_source, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(checkpoint_path + "/tokenizer_target.pickle", "wb") as handle:
        pickle.dump(tokenizer_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, val_dataset, tokenizer_source, tokenizer_target
