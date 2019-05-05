## Transformer classifier

This architecture follows the transformer multihead attention module. However, it does not have the decoder as it is used to predict scalar outputs directly, without using the decoder layer. 

This model is useful when classifying particular classes from a tex input data. For instance, sentiment analysis or categorization of text input. 

The source is developed using Tensorflow 2.0 and an API is released with the repository using flask. Everything is wrapped up into a docker container. Therefore, this model can be released as a rest API

        docker-compose build
        docker-compose up

To run the container and make predictions in the command line use: 

        docker run --rm -it -v $PWD:/src/  transformer_classifier_transformer:latest /bin/bash


        usage: train.py [-h] [--MAX_LENGTH MAX_LENGTH] [--BUFFER_SIZE BUFFER_SIZE]
                        [--BATCH_SIZE BATCH_SIZE] [--EPOCHS EPOCHS]
                        [--num_heads NUM_HEADS] [--num_layers NUM_LAYERS]
                        [--d_model D_MODEL] [--dff DFF] [--vocab_dim VOCAB_DIM]
                        [--dropout_rate DROPOUT_RATE]
                        [--test_partition TEST_PARTITION] --dataset_file DATASET_FILE
                        --checkpoint_path CHECKPOINT_PATH [--retrain]

                        optional arguments:
                        -h, --help            show this help message and exit
                        --MAX_LENGTH MAX_LENGTH
                        --BUFFER_SIZE BUFFER_SIZE
                        --BATCH_SIZE BATCH_SIZE
                        --EPOCHS EPOCHS
                        --num_heads NUM_HEADS
                        --num_layers NUM_LAYERS
                        --d_model D_MODEL
                        --dff DFF
                        --vocab_dim VOCAB_DIM
                        --dropout_rate DROPOUT_RATE
                        --test_partition TEST_PARTITION
                        --dataset_file DATASET_FILE
                        --checkpoint_path CHECKPOINT_PATH
                        --retrain

### REST API USAGE

This repository has a simple flask api wrapped into a docker container that can be used to make predictions. After training and after running docker-compse up you will get an API exposed on the port 65431

You can run the next CURL request to check it: 

        curl -X POST http://localhost:65431/predict/ \
                -d'{"sentence": "hola, buenos dias", "max_predictions": 5, "attention": false, "layer": 4, "block": 1}' \
                -H "Content-Type: application/json"


### TRAINING on COLAB
This repository includes a colab notebook for training: 
* You need to create a ./data/ directory under the repository directory
* The input data has to be a tab (\t) delimited file where the first column is the text input and the second column is the class label. 
* Follow the instructions on the colab notebook to perform the training and testing. 
* Enjoy!

### Contact
www.umayuxlabs.com
info@umayuxlabs.com