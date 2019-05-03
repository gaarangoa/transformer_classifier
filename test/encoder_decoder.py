from utensor.layers import Encoder, Decoder
import tensorflow as tf
import random

# ENCODER
sample_encoder = Encoder(
    num_layers=1, d_model=512, num_heads=1, dff=2048, input_vocab_size=20
)

labels = [random.randint(0, 2) for i in range(0, 50)]

inp = tf.random.uniform((50, 20))  # 5 sentences of 20 tokens each
tar = tf.Variable(labels)  #  5 target lables of 1 token (class)

sample_encoder_output = sample_encoder(inp, training=False, mask=None)

print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

# DECODER
# target_vocab_size  =  num of classes
sample_decoder = Decoder(
    num_layers=1,
    d_model=512,
    num_heads=1,
    dff=2048,
    target_vocab_size=len(list(set(labels))),
)

output, attn = sample_decoder(
    tar,
    enc_output=sample_encoder_output,
    training=False,
    look_ahead_mask=None,
    padding_mask=None,
)

print(output.shape, attn["decoder_layer1_block1"].shape)
