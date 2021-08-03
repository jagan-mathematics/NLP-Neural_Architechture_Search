'''
Bellow configurations will be used by main file to create tokenizer and to define the
input data defentions

There are some default values were defined if you want to change those value uncomment 
the variable and pass the value to it.

Tokenizer default value:
  -> pattern

Data provider default value:
  -> dim
  -> field_delim
  -> header
'''

import os

class Config:
  # Tokenizer configuration
  stagging_folder = 'stagging'
  tokenizer_file = 'tokenizer_config.tfrecord'
  vocab_size = 10001 # your desire vocab_size + 1
  max_sequence_length = 250
  # pattern # regex pattern which applies to preprocess input data

  # Data Configuration
  dataset_config = {
    'input_file' : 'dataset/train.tsv', # input dataset file path
    'default_records' : ['', 1.0],  # default value for the selected column
    'feature_idx' : [2, 3], # columns to be selected as input for model search first idex must be text 
                          # second index must be label
    'batch_size' : 32, # batch size
    'label' : 5, # no of output labels
    # 'dim' : int # no of embedding dim default : 128
    # 'field_delim': str # delimator used for given fie default : \t
    # 'header' : bool # is header is present in file or not default : True 
  }

  # spec ptxt file
  spec_path = 'model_search/configs/rnn_last_config.pbtxt'
  model_staging_foler = "Model searching_1"
  experiment_name = 'example' 
  number_models = 5
  train_steps = 350
  eval_steps = 100
  search_staging_dir = os.path.join(stagging_folder, model_staging_foler)