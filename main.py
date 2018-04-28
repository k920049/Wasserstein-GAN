from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lmdb

from models.discriminator import Discriminator
from models.generator import Generator

lmdb_paths = "./resource/lsun/church_outdoor_train_lmdb"
n_epoch = 200000
n_batch = 64
n_critic = 5
c = 0.01
learning_rate = 0.001

def get_batch(db_path, batch_size):
    # Open the lmdb environment
    env = lmdb.open(path=db_path, map_size=1099511627776, max_readers=100, readonly=True)
    # list of batches
    batch = []
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for key, val in cur:
            batch.append(val)
            if len(batch) == batch_size:
                yield batch
                batch.clear()
        if len(batch) > 0:
            yield batch
            batch.clear()
    # close the environment
    env.close()

def main():

    with tf.name_scope("placeholders"):
        input_plc = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3))
        prior_plc = tf.placeholder(dtype=tf.float32, shape=(None, 128))
        training = tf.placeholder_with_default(False, shape=())

    with tf.name_scope("networks"):
        flag = False
        generator = Generator(_num_units=128, _scope="generator")
        discriminator = Discriminator(_scope="discriminator")
        G = generator.build_network(_prior=prior_plc, _training=training)
        gen_logit = discriminator.build_network(_input_plc=G, _training=training)
        disc_logit = discriminator.build_network(_input_plc=input_plc, _training=training)

    with tf.name_scope("loss"):
        loss_disc = tf.reduce_mean(disc_logit) - tf.reduce_mean(gen_logit)
        loss_gen = tf.reduce_mean(gen_logit)

    with tf.name_scope("optimization"):
        # get the trainable variables from both networks
        var_disc = discriminator.get_trainable()
        var_gen = generator.get_trainable()
        # compute gradient for discriminator network
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        disc_grad = optimizer.compute_gradients(loss=loss_disc, var_list=var_disc)
        disc_training_op = optimizer.apply_gradients(grads_and_vars=disc_grad)
        # clip the weights
        assign_ops = []
        for var in var_disc:
            clipped = tf.clip_by_value(var, -c, c)
            assign_ops.append(tf.assign(var, clipped))
        # compute gradient for generative network
        gen_grad = optimizer.compute_gradients(loss=loss_gen, var_list=var_gen)
        gen_training_op = optimizer.apply_gradients(grads_and_vars=gen_grad)

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        shown = False

        for ep in range(n_epoch):
            batch_generator = get_batch(lmdb_paths, n_batch)
            for critic in range(n_critic):
                batch = []
                try:
                    b = next(batch_generator)
                except StopIteration:
                    break

                for image in b:
                    img = tf.image.decode_jpeg(contents=image, channels=3)
                    img = tf.image.resize_images(img, [128, 128])

                    if not shown:
                        shown = True
                        img_t = tf.cast(x=img, dtype=tf.uint8)
                        decoded = img_t.eval()
                        plot, axes = plt.subplots(figsize=(10, 6))
                        axes.imshow(decoded,
                                    aspect="equal",
                                    interpolation="none",
                                    vmin=0.0,
                                    vmax=255.0)
                        plt.show()
                    img = img.eval()
                    batch.append(img)
                discriminator_batch = np.stack(batch)
                # generate random priors from gaussian distribution
                generator_batch = np.random.normal(0, 1, size=(n_batch, 128)).astype(np.float32)
                # get all generated samples
                loss, train, assign = sess.run([loss_disc, disc_training_op, assign_ops], feed_dict={input_plc: discriminator_batch, prior_plc: generator_batch, training: True})
                print("At epoch: {}, running {}-th critic. duality loss: {}".format(ep, critic, loss))
            # generate random priors from gaussian distribution
            generator_batch = np.random.normal(0, 1, size=(n_batch, 128)).astype(np.float32)
            images, loss, _ = sess.run([G, loss_gen, gen_training_op], feed_dict={prior_plc: generator_batch, training: True})
            print("At epoch: {}, loss: {}".format(ep, loss))
            img, axes = plt.subplots(figsize=(10, 6))
            axes.imshow(images[0],
                        aspect="equal",
                        interpolation="none",
                        vmin=-1.0,
                        vmax=1.0)
            plt.show()

            saver.save(sess=sess, save_path="./resource/snapshot.ckpt")

if __name__ == '__main__':
    main()
