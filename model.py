import os
import tensorflow as tf
from layers import stacked_hourglass

from utils.dirs import ensure_dir


class StackedHourglassModel:
    def __init__(self, config):
        self.n_channels = config.n_channels
        self.img_size = config.img_size
        self.out_size = config.out_size

        self.out_dim = len(config.categories)
        self.n_stacks = config.n_stacks
        self.learn_rate = config.learn_rate
        # self.decay = config.decay
        # self.decay_step = config.decay_step
        self.is_training = config.is_training
        self.cp_dir = os.path.join(config.cp_dir, config.name)

        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.step = tf.Variable(0, trainable=False, name='step')
        self.increment_epoch = tf.assign(self.epoch, self.epoch + 1)
        self.increment_step = tf.assign(self.step, self.step + 1)

        self.min_validation_loss = tf.Variable(1., trainable=False, name='min_validation_loss', dtype=tf.float64)

        self.build_model()
        self.saver = tf.train.Saver()

    def save(self, sess):
        ensure_dir(self.cp_dir)
        print('Saving model...')
        self.saver.save(sess, self.cp_dir + "/cp", self.step)
        print('Model saved')

    def load(self, sess):
        cp = tf.train.latest_checkpoint(self.cp_dir)
        if cp:
            print(f'Loading model checkpoint {cp}...')
            self.saver.restore(sess, cp)
            print('Model loaded')

    def build_model(self):
        with tf.variable_scope('inputs'):
            self.img = tf.placeholder(dtype=tf.float32, shape=(None, self.img_size, self.img_size, self.n_channels), name='input_img')
            self.gtmap = tf.placeholder(dtype=tf.float32, shape=(None, self.n_stacks, self.out_size, self.out_size, self.out_dim))
            self.avg_lengths = tf.placeholder(dtype=tf.float32, shape=None)
            self.is_training = tf.placeholder(tf.bool, name='training_flag')

        self.output = stacked_hourglass(inputs=self.img, nstacks=self.n_stacks, nlow=4, nfeats=256, outdim=self.out_dim,
                                        dropout_rate=0.9, is_training=self.is_training)
        self.prob_output = tf.sigmoid(self.output)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=self.output, targets=self.gtmap, pos_weight=0.5),
                                       name='cross_entropy_loss')

        with tf.variable_scope('minimizer'):
            # self.lr = tf.train.exponential_decay(self.learn_rate, self.step, self.decay_step, self.decay, staircase=True, name='learning_rate')
            self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate)
            self.train_op = self.rmsprop.minimize(self.loss)

        self.init = tf.global_variables_initializer()
