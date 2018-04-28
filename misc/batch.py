import lmdb
import tensorflow as tf
import numpy as np

class Batch(object):

    def __init__(self, _db_path, _batch_size):

        self.batch_size = _batch_size
        self.db_path = _db_path




