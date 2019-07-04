from sklearn.model_selection import train_test_split
import json
import os
import time
from sklearn.metrics import classification_report
import pickle
from utensor.masking import create_masks, create_masks_train
from utensor.dataset import load_dataset
from utensor.model import Transformer
from utensor.dataset import Dataset
from utensor.optimizer import CustomSchedule, loss_function
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import logging
import sys
import numpy as np
from sklearn.metrics import f1_score

# file_handler = logging.FileHandler(filename="tmp.log")
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.ERROR,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger()
logger.info("start")


tf.keras.backend.clear_session()


def test_acc(batch=32, test_dataset=[], transformer=[], test_accuracy=[], test_loss=[]):
    real = []
    pred = []
    for (batch, (inp, tar, bert_inp)) in enumerate(test_dataset):
        logger.debug("input: {}".format(inp.shape))
        logger.debug("target: {}".format(tar.shape))

        enc_padding_mask = create_masks(inp, tar)

        predictions, _, _ = transformer(
            inp, tar, bert_inp, False, enc_padding_mask, None, None)
        logger.debug("predictions: {}".format(predictions.shape))
        logger.debug("tar_real: {}".format(tar.shape))

        rand_samples = np.random.choice(
            len(tar),
            int(len(tar) * 0.9),
            replace=False
        )

        tar = tf.Variable([tar[i] for i in rand_samples])
        predictions = tf.Variable([predictions[i] for i in rand_samples])

        real += tar.numpy().tolist()
        pred += [i for i in np.argmax(predictions.numpy(), axis=1)]

        test_accuracy(tar, predictions)
        test_loss(loss_function(tar, predictions))

    print(classification_report(real, pred))
    F1 = f1_score(real, pred, average='macro')

    return F1


def train(args):

    params = dict(
        MAX_LENGTH=args.MAX_LENGTH,
        BUFFER_SIZE=args.BUFFER_SIZE,
        BATCH_SIZE=args.BATCH_SIZE,
        EPOCHS=args.EPOCHS,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_model=args.d_model,
        dff=args.dff,
        vocab_dim=args.vocab_dim,
        dropout_rate=args.dropout_rate,
        test_partition=args.test_partition,
        dataset_file=args.dataset_file,
        checkpoint_path=args.checkpoint_path,
        retrain=args.retrain,
    )

    # save parameters
    logger.info("saving parameters to {}".format(params["checkpoint_path"]))
    json.dump(params, open(params["checkpoint_path"] + "/params.json", "w"))

    # load the dataset
    logger.info("loading dataset")
    train_dataset, val_dataset, tokenizer_source, tokenizer_target = load_dataset(
        params=params
    )

    logger.debug(
        "FORMAT DATASET: the dataset consists of a query text and a target class (numerical value)"
    )

    example_entry = [i for i in train_dataset][0]
    logger.debug("example entry: {}".format(example_entry[0][0]))
    logger.debug("example class: {}".format(example_entry[1][0]))

    sample_string = [
        tokenizer_source.decode([i]) for i in example_entry[0].numpy()[0][1:3]
    ]
    sample_string = "".join(sample_string)

    tokenized_string = tokenizer_source.encode(sample_string)
    sample_class = tokenizer_target.int2str(example_entry[1][0].numpy())
    tokenized_class = tokenizer_target.encode_example(sample_class)

    logger.debug("sample string: {}".format(sample_string))
    logger.debug("tokenized string: {}".format(tokenized_string))
    for ts in tokenized_string:
        logger.debug("{} ----> {}".format(ts, tokenizer_source.decode([ts])))
    logger.debug(
        "original class: {} ===> tokenized class: {}".format(
            sample_class, tokenized_class
        )
    )
    logger.debug("Number of classes: {}".format(tokenizer_target.num_classes))

    input_vocab_size = tokenizer_source.vocab_size + 2
    target_vocab_size = tokenizer_target.num_classes

    logger.info("setup learning rate and optimizer")
    # Setup the learning rate and optimizer
    learning_rate = CustomSchedule(params["d_model"])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    logger.info("setup loss function")
    # setup loss
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")

    logger.info("Setup transformer model")
    # setup Transformer Model
    transformer = Transformer(
        params["num_layers"],
        params["d_model"],
        params["num_heads"],
        params["dff"],
        input_vocab_size,
        target_vocab_size,
        params["dropout_rate"],
    )

    logger.info(
        "input_vocab_size: {} classes: {}".format(
            input_vocab_size, target_vocab_size)
    )

    # setup checkpoints
    logger.info("Setup Checkpoints: {}".format(params["checkpoint_path"]))
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, params["checkpoint_path"], max_to_keep=1
    )

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint and params["retrain"]:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logger.info("Latest checkpoint restored!!")
    else:
        logger.info("Initializing from scratch.")

    # define training function step
    # @tf.function
    def train_step(inp, tar, bert_inp):

        logger.debug("input: {}".format(inp.shape))
        logger.debug("target: {}".format(tar.shape))

        enc_padding_mask = create_masks_train(inp, tar)

        logger.debug("enc_padding_mask: {}".format(enc_padding_mask.shape))

        with tf.GradientTape() as tape:
            predictions, enc_output, _ = transformer(
                inp, tar, bert_inp, True, enc_padding_mask, None, None
            )

            logger.debug("predictions: {}".format(predictions.shape))
            logger.debug("tar_real: {}".format(tar.shape))
            logger.debug("enc_output: {}".format(enc_output.shape))
            # logger.debug("enc_output: {}".format(enc_output.shape))

            loss = loss_function(tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar, predictions)

    # training loop
    best_test_acc = 0
    for epoch in range(params["EPOCHS"]):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (_, (inp, tar, bert_inp)) in enumerate(train_dataset):
            train_step(inp, tar, bert_inp)

        print(
            "Epoch {} Train Loss {:.4f} Accuracy {:.4f}".format(
                epoch + 1, train_loss.result(), train_accuracy.result()
            )
        )

        # Perform accuracy over the test dataset
        test_accuracy.reset_states()
        test_loss.reset_states()
        test_f1_score = test_acc(
            batch=32,
            test_dataset=val_dataset,
            transformer=transformer,
            test_accuracy=test_accuracy,
            test_loss=test_loss,
        )

        print(
            "Epoch {} Test Loss {:.4f} Accuracy {:.4f} F1 {:.4f}".format(
                epoch + 1, test_loss.result(), test_accuracy.result(), test_f1_score
            )
        )

        if best_test_acc < test_f1_score:
            ckpt_save_path = ckpt_manager.save()
            print(
                "Saving checkpoint for epoch {} at {}".format(
                    epoch + 1, ckpt_save_path)
            )
            best_test_acc = test_f1_score

        print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MAX_LENGTH",
        type=int,
        default=40,
        help="maximum length of the input text sentences",
    )
    parser.add_argument(
        "--BUFFER_SIZE", type=int, default=5000, help="buffer size for training"
    )
    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=32,
        help="batch size to use for training (min batch)",
    )
    parser.add_argument(
        "--EPOCHS", type=int, default=100, help="number of epochs to train the model"
    )
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="transformer: Number of attention heads",
    )
    parser.add_argument(
        "--num_layers", default=4, type=int, help="transformer: number of encoder units"
    )
    parser.add_argument(
        "--d_model",
        default=64,
        type=int,
        help="transformer: embedding dimension of the model",
    )
    parser.add_argument(
        "--dff",
        default=264,
        type=int,
        help="transformer: placeholder dimension for normalization",
    )
    parser.add_argument(
        "--vocab_dim",
        default=10000,
        type=int,
        help="transformer: maximum vocabulary size",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="transformer: dropout rate for training",
    )
    parser.add_argument(
        "--test_partition", default=0.2, type=float, help="slice of data for testing"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="input tsv dataset with two columns: sentence -> class",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="checkpoint where to save the model",
    )
    parser.add_argument(
        "--retrain",
        default=False,
        action="store_true",
        help="if retraining the model, start from previous stored model",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    os.makedirs(args.checkpoint_path, exist_ok=True)
    train(args)
