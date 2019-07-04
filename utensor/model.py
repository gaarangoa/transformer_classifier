import tensorflow as tf
from .layers import Encoder, SupportPreLayer


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
        bert_embeddings_size=768
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, rate
        )

        self.flatten = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.dense_1 = tf.keras.layers.Dense(
            target_vocab_size, activation="softmax")

        self.bert_embeddings_layer = SupportPreLayer(bert_embeddings_size)

        # self.out = tf.nn.softmax(target_vocab_size)

    def call(
        self, inp, tar, inp_bert_embeddings, training, enc_padding_mask, look_ahead_mask, dec_padding_mask
    ):

        enc_output, attention_weights = self.encoder(
            inp, training, enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        bert_embeddings_out = self.bert_embeddings_layer(inp_bert_embeddings)

        flat_encoder_output = self.flatten(enc_output)

        merge_output = tf.concat(
            (bert_embeddings_out, flat_encoder_output), 1
        )

        predictions = self.final_layer(
            merge_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        predictions = self.dense_1(predictions)

        return predictions, enc_output, attention_weights
