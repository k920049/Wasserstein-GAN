import tensorflow as tf
import numpy as np

class Generator(object):

    def __init__(self, _num_units,  _scope):

        self.num_units = _num_units
        self.scope = _scope
        # Gaussian prior
        self.mean = np.zeros(shape=[1, self.num_units], dtype=np.float64)
        self.sigma = np.eye(N=self.num_units, dtype=np.float64)

        self.kernel_sizes = [[5, 5], [5, 5], [5, 5]]
        self.filters = [512, 256, 128]
        self.strides = [[2, 2], [2, 2], [2, 2]]

    def build_network(self, _prior, _training):
        self.prior = _prior
        self.training = _training

        with tf.variable_scope(name_or_scope=self.scope, reuse=tf.AUTO_REUSE):
            # whether we're training or not

            # Sample from gaussian prior
            z = self.prior
            # project and reshape
            z = tf.layers.dense(inputs=z, units=65536, activation=None, name="projection")
            print(z)
            z = tf.reshape(tensor=z, shape=(-1, 8, 8, 1024))
            layer = 0
            # deconvolution layers
            for kernel_size, filter, stride in zip(self.kernel_sizes, self.filters, self.strides):
                z = tf.layers.conv2d_transpose(inputs=z,
                                               filters=filter,
                                               kernel_size=kernel_size,
                                               strides=stride,
                                               padding='same',
                                               name="deconv" + str(layer))
                # batch normalize every layer
                z = tf.layers.batch_normalization(inputs=z,
                                                  training=self.training,
                                                  momentum=0.9,
                                                  name="bn" + str(layer))
                # activation function
                z = tf.nn.relu(z)
                layer = layer + 1

            # last output layer
            z = tf.layers.conv2d_transpose(inputs=z,
                                           filters=3,
                                           kernel_size=[5, 5],
                                           strides=[2, 2],
                                           padding='same',
                                           name="deconv" + str(layer))
            z = tf.layers.batch_normalization(inputs=z,
                                              training=self.training,
                                              momentum=0.9,
                                              name="bn" + str(layer))
            z = tf.nn.tanh(z)
        # return the last logit
        self.logit = z
        return self.logit

    def get_logit(self):
        return self.logit

    def get_trainable(self):
        return tf.trainable_variables(scope=self.scope)






