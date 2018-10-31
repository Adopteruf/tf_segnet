# Author: Jingwei Guo
# Date: 10-31-2018
# model module
# the same structure to U-net in coding
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import dtypes
# from tensorflow.python.ops import gen_nn_ops

from layers_segnet import inference, pixel_wise_softmax
import tensorflow as tf
import numpy as np
import logging


class Segnet(object):
    """
    A classic segnet implementation with input size 184x184 --> the same output size
    :param model_name: the name of the current constructed model
    :param channels: channels' number of the input image
    :param nb_class: the number of classified class
    :param img_h: height of the input image
    :param img_w: the width of the input image
    :param batch_size: the size of one training batch
    :param cost_name: the name of the designed cost function
    :param regularizer: the contributed weights of regularizer to the loss and None if no regularizer
    :param log_net: true if the information of current model should be logged
    :param is_bw: true if weighted loss is considered
    :param x: input batch-image
    :param y: ground-truth label of x in indexes instead of one-hot
    :param batch_weights: loss weight for each input in the current batch
    :param keep_prob: the dropout probability
    :param phase_train: true if layers should be updated
    :param variables: all the variables set in the model
    :param cost: the tensor of the final loss of the current model
    :param predictor: the predicted result
    """

    def __init__(self, model_name, log_net, **kwargs):
        tf.reset_default_graph()

        # load params
        self.model_name = model_name
        self.log_net = log_net
        self.channels = kwargs.pop("channels")
        self.nb_class = kwargs.pop("nb_class")
        self.img_h = kwargs.pop("img_h", 184)
        self.img_w = kwargs.pop("img_w", 184)
        self.batch_size = kwargs.pop("batch_size", 20)
        self.cost_name = kwargs.pop("cost_name", "cross_entropy")
        self.regularizer = kwargs.pop("regularizer", None)
        self.is_bw_loss = kwargs.pop("is_bw_loss", False)
        # set place-holder
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[self.batch_size, self.img_h, self.img_w, self.channels],
                                name="x")
        self.y = tf.placeholder(dtype=tf.int32,
                                shape=[self.batch_size, self.img_h, self.img_w, 1],
                                name="y")
        self.batch_weights = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, ], name="batch_weights")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_prob")
        self.phase_train = tf.placeholder(dtype=tf.bool, name="is_training")
        # log all the settings
        if self.log_net:
            logging.info("Seg-Net information")
            logging.info("model name: " + self.model_name + "\r\n" +
                         "nb_channel: " + str(self.channels) + "\r\n" +
                         "nb_class: " + str(self.nb_class) + "\r\n" +
                         "image height: " + str(self.img_h) + "\r\n" +
                         "image width: " + str(self.img_w) + "\r\n" +
                         "batch size: " + str(self.batch_size) + "\r\n" +
                         "cost name: " + str(self.cost_name) + "\r\n" +
                         "regularizer: " + str(self.regularizer) + "\r\n" +
                         "log_net: " + str(self.log_net) + "\r\n" +
                         "is_bw_loss: " + str(self.is_bw_loss) + "\r\n")
        # build model
        logits, self.variables = inference(images=self.x,
                                           img_h=self.img_h,
                                           img_w=self.img_w,
                                           nb_class=self.nb_class,
                                           batch_size=self.batch_size,
                                           phase_train=self.phase_train)
        self.cost = self._get_cost(logits=logits,
                                   cost_name=self.cost_name,
                                   batch_weights=self.batch_weights,
                                   regularizer=self.regularizer)
        # build result
        with tf.name_scope("results"):
            self.predictor = pixel_wise_softmax(logits)

    def _get_cost(self, logits, cost_name, batch_weights, regularizer):
        """
        Constructs the cost function: either cross_entropy, weighted cross_entropy or dice_coefficient
        """
        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, self.nb_class])
            flat_labels = tf.reshape(self.y, [-1, ])
            # build loss-function
            if cost_name == "cross_entropy":
                # batch-weights
                if self.is_bw_loss:
                    batch_weights_map = tf.constant(np.array(batch_weights), dtype=tf.float32)
                    loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels)
                    weighted_loss_map = tf.multiply(loss_map, batch_weights_map)
                    loss = tf.reduce_mean(weighted_loss_map)
                else:
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                         labels=flat_labels))
            elif cost_name == "dice_coefficient":
                eps = 1e-5
                prediction = pixel_wise_softmax(logits)
                intersection = tf.reduce_sum(prediction * tf.one_hot(tf.squeeze(self.y), depth=self.nb_class))
                union = eps + tf.reduce_sum(prediction) + tf.size(self.y)
                loss = -(2 * intersection / union)
            else:
                raise ValueError("Unknown cost name: " + cost_name)
            # add regularizer
            if regularizer is not None:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
                loss += (regularizer * regularizers)
            return loss

    def save(self, sess, model_path):
        """
        Save model
        :param sess: current session
        :param model_path: path to save the current model
        """
        saver = tf.train.Saver()
        saver.save(sess, model_path)
        logging.info("Model saved into file: " + model_path)

    def restore(self, sess, model_path):
        """
        Restore model
        :param sess: current session
        :param model_path: path to restore the model
        """
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: " + model_path)
