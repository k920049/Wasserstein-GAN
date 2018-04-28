import tensorflow as tf
import numpy as np

class Discriminator(object):

    def __init__(self, _num_units=49152, _num_layers=3, _scope="discriminator"):
        # [batch_size, 128, 128, 3]
        self.num_units = _num_units
        self.num_layer = _num_layers
        self.scope = _scope
        # parameters in layers
        self.kernel_sizes = [[11, 11], [5, 5], [3, 3], [3, 3], [3, 3]]
        self.strides = [[4, 4], [2, 2], [1, 1], [1, 1], [1, 1]]
        self.filters = [96, 256, 384, 384, 256]
        self.units = [4096, 4096, 1]


    def build_network(self, _input_plc, _training):
        self.input_place = _input_plc
        self.training = _training
        with tf.variable_scope(name_or_scope=self.scope, reuse=tf.AUTO_REUSE):
            # placeholder including input values
            X = self.input_place
            # use AlexNet
            layer = 0
            for kernel_size, stride, filter in zip(self.kernel_sizes, self.strides, self.filters):
                # convolution layers
                X = tf.layers.conv2d(inputs=X,
                                     filters=filter,
                                     kernel_size=kernel_size,
                                     strides=stride,
                                     padding='same',
                                     name="conv" + str(layer))
                # batch normalize every layer
                X = tf.layers.batch_normalization(inputs=X,
                                                  training=self.training,
                                                  momentum=0.9,
                                                  name="bn" + str(layer))
                # use leaky relu
                X = tf.nn.leaky_relu(X)
                layer = layer + 1
            # fully connected layers for logit
            for unit in self.units:
                # dense layers
                X = tf.layers.dense(inputs=X,
                                    units=unit,
                                    name="dense" + str(layer))
                # use leaky relu
                X = tf.nn.leaky_relu(X)
                layer = layer + 1
        self.logit = X
        return X

    def get_logit(self):
        return self.logit

    def get_trainable(self):
        return tf.trainable_variables(scope=self.scope)
