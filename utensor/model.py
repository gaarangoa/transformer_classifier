import tensorflow as tf
from .layers import Encoder


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        rate=0.1,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, rate
        )

        self.flatten = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.dense_1 = tf.keras.layers.Dense(target_vocab_size, activation="softmax")
        # self.out = tf.nn.softmax(target_vocab_size)

    def call(
        self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask
    ):

        enc_output, attention_weights = self.encoder(
            inp, training, enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        predictions = self.final_layer(
            enc_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        predictions = self.flatten(predictions)
        predictions = self.dense_1(predictions)

        return predictions, enc_output, attention_weights
