import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
import numpy as np


# one convolution
def conv(inputs, num_out, kernel_size, strides, padding=0, name='conv'):
    with tf.name_scope(name):
        kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0001)
        if padding > 0:
            paddings = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])
            inputs = tf.pad(inputs, paddings, "CONSTANT")
        outputs = Conv2D(filters=num_out, kernel_size=kernel_size, strides=strides, padding='valid',
                         kernel_initializer=kernel_initializer)(inputs)
    return outputs


# one convolution + batch normalization + ReLu activation
def conv_bn_relu(inputs, num_out, kernel_size, strides, padding, is_training, name='conv_bn_relu'):
    with tf.name_scope(name):
        cnv = conv(inputs, num_out, kernel_size, strides, padding)
        cnv_bn = BatchNormalization()(cnv, training=is_training)
        outputs = Activation('relu')(cnv_bn)
    return outputs


# batch normalization + ReLu activation
def bn_relu(inputs, is_training):
    bn = BatchNormalization()(inputs, training=is_training)
    outputs = Activation('relu')(bn)
    return outputs


# (batch normalization + convolution) x 3
def conv_block(inputs, num_out, is_training, name='conv_block'):
    with tf.name_scope(name):
        with tf.name_scope('norm_1'):
            conv_1 = conv(inputs, int(num_out / 2), kernel_size=1, strides=1, padding=0, name='conv')
            norm_1 = BatchNormalization()(conv_1, training=is_training)
            outp_1 = Activation('relu')(norm_1)
        with tf.name_scope('norm_2'):
            conv_2 = conv(outp_1, int(num_out / 2), kernel_size=3, strides=1, padding=1, name='conv')
            norm_2 = BatchNormalization()(conv_2, training=is_training)
            outp_2 = Activation('relu')(norm_2)
        with tf.name_scope('norm_3'):
            conv_3 = conv(outp_2, int(num_out), kernel_size=1, strides=1, padding=0, name="conv")
            outp_3 = BatchNormalization()(conv_3, training=is_training)

        return outp_3


# one-by-one convolution layer
def skip_layer(inputs, num_out, name='skip_layer'):
    with tf.name_scope(name):
        if inputs.get_shape().as_list()[3] == num_out:
            return inputs
        else:
            cnv = conv(inputs, num_out, kernel_size=1, strides=1, padding=0, name='conv')
            return cnv


def pool_layer(inputs, num_out, is_training, name='pool_layer'):
    with tf.name_scope(name):
        bnr_1 = bn_relu(inputs, is_training)
        pool = MaxPooling2D((2, 2))(bnr_1)
        cnv_1 = conv(pool, num_out, kernel_size=3, strides=1, padding=1, name='conv')
        bnr_2 = bn_relu(cnv_1, is_training)
        cnv_2 = conv(bnr_2, num_out, kernel_size=3, strides=1, padding=1, name='conv')
        upsample = tf.image.resize_nearest_neighbor(cnv_2, tf.shape(cnv_2)[1:3]*2, name='upsampling')
    return upsample


# (batch normalization + convolution) x 3 + skip_layer + ReLu activation
def residual(inputs, num_out, is_training, name='residual_block'):
    with tf.name_scope(name):
        convb = conv_block(inputs, num_out, is_training)
        skipl = skip_layer(inputs, num_out)
        resb = tf.add_n([convb, skipl], name='res_block')
        outputs = Activation('relu')(resb)
    return outputs


def residual_pool(inputs, num_out, is_training, name='residual_pool'):
    with tf.name_scope(name):
        convb = conv_block(inputs, num_out, is_training)
        skipl = skip_layer(inputs, num_out)
        pooll = pool_layer(inputs, num_out, is_training)
        outputs = tf.add_n([convb, skipl, pooll])
    return outputs


def rep_residual(inputs, num_out, n_rep, is_training, name='rep_residual'):
    with tf.name_scope(name):
        out = [None] * n_rep
        for i in range(n_rep):
            if i == 0:
                tmpout = residual(inputs, num_out, is_training)
            else:
                tmpout = residual_pool(out[i-1], num_out, is_training)
            out[i] = tmpout
    return out[n_rep-1]


def hourglass(inputs, n, num_out, is_training, name):     # if input size: (b, 64, 64, 256)
    with tf.name_scope(name):
        # upper branch
        up_1 = residual(inputs, num_out=num_out, is_training=is_training, name='up_1')
        # lower branch
        low_ = MaxPooling2D((2, 2))(inputs)
        low_1 = residual(low_, num_out=num_out, is_training=is_training, name='low_1')

        if n > 0:
            low_2 = hourglass(low_1, n=n-1, num_out=num_out, is_training=is_training, name='low_2')
        else:
            low_2 = residual(low_1, num_out=num_out, is_training=is_training, name='low_2')

        low_3 = residual(low_2, num_out=num_out, name='low_3', is_training=is_training)
        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name='upsampling')
        outputs = tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
    return outputs


def stacked_hourglass(inputs, nstacks=1, nlow=4, nfeats=256, outdim=1, dropout_rate=0.9, is_training=True):
    with tf.name_scope('preprocessing'):
        cnv1 = conv_bn_relu(inputs, num_out=64, kernel_size=7, strides=2, padding=3,
                            is_training=is_training)  # output size: (b, 128, 128, 64)
        r1 = residual(cnv1, num_out=64, is_training=is_training)  # output size: (b, 128, 128, 128)
        pool1 = MaxPooling2D((2, 2))(r1)  # output size: (b, 64, 64, 128)
        r2 = residual(pool1, int(nfeats / 2), is_training=is_training)  # output size: (b, 64, 64, 128)
        r3 = residual(r2, nfeats, is_training=is_training)  # output size: (b, 64, 64, 256)

    hg = [None] * nstacks
    ll = [None] * nstacks
    ll_ = [None] * nstacks
    drop = [None] * nstacks
    out = [None] * nstacks
    out_ = [None] * nstacks
    sum_ = [None] * nstacks

    with tf.name_scope('stacks'):
        with tf.name_scope('hourglass_0'):
            hg[0] = hourglass(r3, n=nlow, num_out=nfeats, is_training=is_training, name='hourglass')
            drop[0] = tf.layers.dropout(hg[0], rate=dropout_rate, training=is_training, name='dropout')
            ll[0] = conv_bn_relu(drop[0], num_out=nfeats, kernel_size=1, strides=1, padding=0,
                                 is_training=is_training, name='conv')
            out[0] = conv(ll[0], num_out=outdim, kernel_size=1, strides=1, padding=0, name='out')

            ll_[0] = conv(ll[0], num_out=nfeats, kernel_size=1, strides=1, padding=0, name='ll_')
            out_[0] = conv(out[0], num_out=nfeats, kernel_size=1, strides=1, padding=0, name='out_')
            sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')

        for i in range(1, nstacks - 1):
            with tf.name_scope('hourglass_' + str(i)):
                hg[i] = hourglass(sum_[i-1], n=nlow, num_out=nfeats, is_training=is_training, name='hourglass')
                drop[i] = tf.layers.dropout(hg[i], rate=dropout_rate, training=is_training, name='dropout')
                ll[i] = conv_bn_relu(drop[i], num_out=nfeats, kernel_size=1, strides=1, padding=0,
                                     is_training=is_training, name='conv')
                out[i] = conv(ll[i], num_out=outdim, kernel_size=1, strides=1, padding=0, name='out')

                ll_[i] = conv(ll[i], num_out=nfeats, kernel_size=1, strides=1, padding=0, name='ll_')
                out_[i] = conv(out[i], num_out=nfeats, kernel_size=1, strides=1, padding=0, name='out_')
                sum_[i] = tf.add_n([out_[i], sum_[i-1], ll_[0]], name='merge')

        with tf.name_scope('hourglass_' + str(nstacks - 1)):
            hg[nstacks-1] = hourglass(sum_[nstacks-2], n=nlow, num_out=nfeats, is_training=is_training, name='hourglass')
            drop[nstacks-1] = tf.layers.dropout(hg[nstacks-1], rate=dropout_rate, training=is_training, name='dropout')
            ll[nstacks-1] = conv_bn_relu(drop[nstacks-1], num_out=nfeats, kernel_size=1, strides=1, padding=0,
                                         is_training=is_training, name='conv')
            out[nstacks-1] = conv(ll[nstacks-1], num_out=outdim, kernel_size=1, strides=1, padding=0, name='out')

    return tf.stack(out, axis=1, name='final_output')
