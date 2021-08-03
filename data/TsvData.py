from typing import List, Union
import tensorflow as tf
import json
from model_search.data import data
from utils.text import TextProcessing
# from glove_initializer.initializer import embedding_initialize


class provider(data.Provider):
  """A tsv data provider."""

  def __init__(self, input_file, default_records, feature_idx:List[Union['idx']], batch_size, label, dim = 128,
    tokenizer_path=None, field_delim='\t', header=True):
    self._input_file = input_file
    self._default_records = default_records
    self._feature_idx = feature_idx
    self._field_delim = field_delim
    self._header = header
    self._batch_size = batch_size
    self._label = label
    self.tokenizer_path = tokenizer_path
    self.dim = dim
    self.processor = TextProcessing()
    if tokenizer_path is not None:
      self.processor.from_config(tokenizer_path)

  def get_input_fn(self, hparams, mode, batch_size):
    del hparams
    def input_fn(params=None):
      del params
 
      def encode_pyfn(text, label):
        text_encoded, target = tf.py_function(self.processor.int_vectorize_text,
                                              inp=[text, label],
                                              Tout=(tf.int32, tf.float32))
        return {'text':text_encoded}, target
      
      dataset = tf.data.experimental.CsvDataset(self._input_file,
                                             record_defaults= self._default_records,
                                             header=self._header,
                                             field_delim=self._field_delim,
                                             select_cols=self._feature_idx)
      

      if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(100 * batch_size)
      
      if mode is not None:
        assert self.tokenizer_path is not None, 'tokenizer configuration is not found. create tokenizer using "TextProcessing"'
        dataset = dataset.map(encode_pyfn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      
      if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
      return dataset

    return input_fn

  def get_serving_input_fn(self, hparams):
    """Returns an `input_fn` for serving in an exported SavedModel.
    Args:
      hparams: tf.HParams object.
    Returns:
      Returns an `input_fn` that takes no arguments and returns a
        `ServingInputReceiver`.
    """
    tf.compat.v1.disable_eager_execution()
    features = {
        'text':
            tf.compat.v1.placeholder(
                tf.int32, [None, self.processor._max_sequence_length, 1],
                'text')
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        features=features)

  def number_of_classes(self):
    return self._label

  def get_input_layer_fn(self, problem_type):
    # initialize = embedding_initialize(self.processor.word_to_idx.export()[0], (50, ), '/content/drive/MyDrive/experiment_nlp/glove_initializer/glove.6B/glove.6B.50d.tfrecord-00000-of-00001')
    def input_layer_fn(features,
                         is_training,
                         scope_name="Phoenix/Input",
                         lengths_feature_name=None):
      

      input_feature = tf.reshape(features['text'], [-1, self.processor._max_sequence_length, 1])
      features = {
          'text': tf.sparse.from_dense(input_feature)
          }
      one_hot_layer = tf.feature_column.sequence_categorical_column_with_identity('text', num_buckets=self.processor.vocab_size)
      text_embedding = tf.feature_column.embedding_column(one_hot_layer,
                                            dimension=self.dim)
      #                                       initializer = initialize)
      columns = [text_embedding]
      sequence_input_layer = tf.keras.experimental.SequenceFeatures(columns, name = scope_name)
      sequence_input, sequence_length = sequence_input_layer(features, training=is_training)
      
      return sequence_input, sequence_length
    return input_layer_fn