## Transformer

This architecture follows the transformer multihead attention module. However, it does not have the decoder as it is used to predict scalar outputs directly, without using the decoder layer. 


docker run --rm -it -v $PWD:/src/  simple_transformer_transformer:latest /bin/bash