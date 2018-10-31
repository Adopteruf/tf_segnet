# Author: Jingwei Guo
# Date: 10-30-2018
# Define layers in building net

from util_segnet import _variable_with_weight_decay, _variable_on_gpu_0
import tensorflow as tf
import numpy as np
import math
from math import ceil


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl ** 2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)


def orthogonal_initializer(scale=1.1):
    """
    From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu_0('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out


def get_deconv_filter(f_shape):
    """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    sess_temp = tf.global_variables_initializer()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv


def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                        center=False, updates_collections=None, scope=scope + "_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                        updates_collections=None, center=False, scope=scope + "_bn",
                                                        reuse=True))


def inference(images, img_h, img_w, nb_class, batch_size, phase_train):
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                      name='norm1')
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool1')
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, img_h // 8, img_w // 8, 64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3 = deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, img_h // 4, img_w // 4, 64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2 = deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, img_h // 2, img_w // 2, 64], 2, "up2")
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1 = deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, img_h, img_w, 64], 2, "up1")
    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
    """ end of decode """
    """ start classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, nb_class],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.0005)
        conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu_0('biases', [nb_class], tf.constant_initializer(0.0))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logits = conv_classifier
    # loss = cal_loss(conv_classifier, labels)
    variables = tf.global_variables()
    return logits, variables


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keep_dims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keep_dims=True)
        return exponential_map / normalize


def cross_entropy(y_, output_map):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=output_map)
    # return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")
