import tensorflow as tf
from .attention import MultiHeadAttention
from .positional_encoding import positional_encoding


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class SupportPreLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SupportPreLayer, self).__init__()

        self.ffn_1 = tf.keras.layers.Dense(units)
        self.ffn_2 = tf.keras.layers.Dense(128)
        self.drop_1 = tf.keras.layers.Dropout(0.1)

    def call(self, x):

        out = self.ffn_1(x)
        out = self.drop_1(out)
        out = self.ffn_2(out)

        return out


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, attention_weights = self.mha(
            x, x, x, mask
        )  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attention_weights = {}

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, training, mask)

        attention_weights["decoder_layer{}_block1".format(i + 1)] = block1

        return x, attention_weights  # (batch_size, input_seq_len, d_model)
