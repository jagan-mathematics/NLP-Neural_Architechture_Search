import tensorflow as tf
import numpy as np
from functools import partial


class GloveEmbedding:
    def __init__(self, vocab, glove_file, size):
        self._vocab = vocab
        self.glove_file = glove_file
        self.size = size
    
    def __load_glove_model(self, glove_file):
        dataset = tf.data.TFRecordDataset(self.glove_file)
        raw_data = tf.data.experimental.get_single_element(dataset)
        keypair = tf.io.parse_example(raw_data, {'key' : tf.io.VarLenFeature(dtype = tf.string), 
                               'value': tf.io.VarLenFeature(dtype = tf.string)})
        keypair = {key : tf.sparse.to_dense(value) for key, value in keypair.items()}
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys = keypair['key'], values = keypair['value'])
        
        self.embedding_lookup = tf.lookup.StaticHashTable(initializer=initializer, default_value=tf.constant(''))
        # self.embedding_lookup = keypair['key']
        # self.embedding_lookup_value = keypair['value']
        del keypair

    def is_in(self, x, k, v):
      if tf.where(x == k).shape[0]:
        return v[tf.where(x == k)[0][0]]
      else:
        return b''

    def get_embeddings(self):
      def convert_to_tensor(x):
        if x ==  b'':
          return tf.random.normal(shape = self.size,dtype=tf.float32)
        else:
          return tf.strings.to_number(tf.strings.split(tf.strings.strip(x), ' '))

      self.__load_glove_model(self.glove_file)
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.tables_initializer())
      embeddings = self.embedding_lookup.lookup(tf.constant(self._vocab))
      # function = partial(self.is_in, k = self.embedding_lookup, v=self.embedding_lookup_value)
      # embeddings = tf.map_fn(function, self._vocab)
      
      # embeddings = tf.gather(self.embedding_lookup_value, embeddings)
      embeddings = tf.map_fn(convert_to_tensor, embeddings, tf.float32)
      embeddings = tf.reshape(embeddings, [-1, 50])
      return tf.cast(embeddings, tf.float32)