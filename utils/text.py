import tensorflow as tf
import json

class TextProcessing:
  def __init__(self, vocab_size=1, max_sequence_length=1, pattern = '[!"#$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n]'):
    self._max_sequence_length = max_sequence_length
    self.word_to_idx = None
    self.idx_to_word = None
    self.pattern = pattern
    self.vocab_size = vocab_size
    self.__types = {'max_sequence_length': tf.io.FixedLenFeature([], tf.int64),
         'vocab_size': tf.io.FixedLenFeature([], tf.int64),
         'pattern': tf.io.FixedLenFeature([], tf.string),
         'words': tf.io.VarLenFeature(tf.string),
         'count': tf.io.VarLenFeature(tf.int64),
         'words_to_idx_w': tf.io.VarLenFeature(tf.string),
         'words_to_idx_i': tf.io.VarLenFeature(tf.int64),
         'idx_to_word_w': tf.io.VarLenFeature(tf.string),
         'idx_to_word_i': tf.io.VarLenFeature(tf.int64)
         }

  def __pad_sequence(self, sequence, max_sequence):
    sequence = tf.squeeze(sequence)
    if tf.size(sequence) > max_sequence:
      sequence = tf.slice(sequence, [0], [max_sequence])
    axis_1 = tf.zeros_like(sequence, dtype = tf.int32)
    idx = tf.stack((axis_1, tf.range(tf.size(axis_1))), axis = -1)
    idx = tf.cast(idx, dtype=tf.int64)
    return tf.sparse.to_dense(tf.sparse.SparseTensor(indices=idx, values=sequence, dense_shape=[1,max_sequence]))

  def __parse_example(self, record):
    return tf.io.parse_single_example(record, self.__types)

  def from_config(self, filename):
    raw_data = tf.data.TFRecordDataset(filename)
    raw_record = next(iter(raw_data))
    dic = self.__parse_example(raw_record)
    for item, value in dic.items():
      if isinstance(value, tf.SparseTensor):
        dic[item] = tf.sparse.to_dense(value)
      elif isinstance(value, tf.Tensor):
        dic[item] = value.numpy()

    self.words = dic['words']
    self.count = dic['count']
    self.word_to_idx = tf.lookup.StaticHashTable(
                    initializer=tf.lookup.KeyValueTensorInitializer(
                        keys=dic['words_to_idx_w'],
                        values=tf.cast(dic['words_to_idx_i'], tf.int32),
                    ),
                    default_value=tf.constant(-1),
                    name="word_to_idx"
                )
    self.idx_to_word = tf.lookup.StaticHashTable(
                  initializer=tf.lookup.KeyValueTensorInitializer(
                      keys=tf.cast(dic['idx_to_word_i'], tf.int32),
                      values=dic['idx_to_word_w'],
                  ),
                  default_value=tf.constant(''),
                  name="idx_to_word"
              )
    self._max_sequence_length = dic['max_sequence_length']
    self.pattern = dic['pattern'].decode()
    self.vocab_size = dic['vocab_size']

  def export(self, filename):
    words_to_idx = self.word_to_idx.export()
    idx_to_word = self.idx_to_word.export()
    features = {
      'max_sequence_length' : tf.train.Feature(int64_list=tf.train.Int64List(value=[self._max_sequence_length])),
      'vocab_size' : tf.train.Feature(int64_list=tf.train.Int64List(value=[self.vocab_size])),
      'pattern': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.pattern.encode()])),
      'words': tf.train.Feature(bytes_list=tf.train.BytesList(value=self.words.numpy().tolist())),
      'count': tf.train.Feature(int64_list=tf.train.Int64List(value=self.count.numpy().tolist())),

      'words_to_idx_w': tf.train.Feature(bytes_list=tf.train.BytesList(value=words_to_idx[0].numpy().tolist())),
      'words_to_idx_i': tf.train.Feature(int64_list=tf.train.Int64List(value=words_to_idx[1].numpy().tolist())),

      'idx_to_word_w': tf.train.Feature(bytes_list=tf.train.BytesList(value=idx_to_word[1].numpy().tolist())),
      'idx_to_word_i': tf.train.Feature(int64_list=tf.train.Int64List(value=idx_to_word[0].numpy().tolist())),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    writer = tf.io.TFRecordWriter(filename)
    writer.write(example_proto.SerializeToString())

  def adapt(self, dataset):
    dataset = dataset = dataset.flat_map(lambda x , y : tf.data.Dataset.from_tensor_slices(self._split(x)))
    dataset = dataset.flat_map(lambda x :tf.data.Dataset.from_tensor_slices(x))
    words_list = list(map(lambda x : x.decode(), list(dataset.as_numpy_iterator())))
    words, idx, count  = tf.unique_with_counts(words_list)
    self.words = words
    self.count = count
    top_k_count = tf.math.top_k(count, k = self.vocab_size-1)
    sorted_words = tf.gather(words,top_k_count.indices)
    word_idx = tf.range(1, len(sorted_words)+1)
    
    self.word_to_idx = tf.lookup.StaticHashTable(
                    initializer=tf.lookup.KeyValueTensorInitializer(
                        keys=sorted_words,
                        values=word_idx,
                    ),
                    default_value=tf.constant(-1),
                    name="word_to_idx"
                )
      
    self.idx_to_word = tf.lookup.StaticHashTable(
                  initializer=tf.lookup.KeyValueTensorInitializer(
                      keys=word_idx,
                      values=sorted_words,
                  ),
                  default_value=tf.constant(''),
                  name="idx_to_word"
              )


  def _split(self, text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, self.pattern,'')
    text = tf.strings.regex_replace(text, ' +',' ')
    text = tf.strings.strip(text)
    text = tf.sparse.to_dense(tf.compat.v1.strings.split(text, ' '))
    return text

  def _text_to_sequence(self, text):
    text = self._split(text)
    text = self.word_to_idx.lookup(text)
    text = tf.squeeze(text)
    text = tf.gather(text, tf.where(tf.not_equal(text, -1)))
    text = tf.squeeze(text)
    return text

  def int_vectorize_text(self, text, label):
    text = self._text_to_sequence(text)
    text = self.__pad_sequence(text, self._max_sequence_length)
    # text = tf.cast(text, dtype = tf.dtypes.float32)
    label = tf.cast(label, dtype = tf.dtypes.float32)
    text = tf.reshape(text, [self._max_sequence_length, 1])
    return text, label
