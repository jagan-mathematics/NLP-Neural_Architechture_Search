from glove_initializer.utils.glove import GloveEmbedding
from tensorflow.keras.initializers import Initializer
import tensorflow as tf

class embedding_initialize(Initializer):
  def __init__(self, vocab, vocab_size, file_path):
    self.glove = GloveEmbedding(vocab, file_path, vocab_size)


  def __call__(self, shape, dtype=None, **kwargs):
    return self.glove.get_embeddings()

