# Author: Jingwei Guo
# Date: 10-31-2018
# training module
# the same structure to U-net in coding

import tensorflow as tf
import numpy as np
import util_segnet
import logging
import time
import os


def precision_rate(predictions, labels, nb_input):
    """
    Return the precision rate of each prediction based on dense predictions and 1-hot labels
    """
    predictions = np.argmax(predictions, 3).reshape([nb_input, -1])
    labels = labels.reshape([nb_input, -1])
    return np.mean(np.mean(predictions == labels, axis=1, dtype=np.float), axis=0)


def img_saver(images, file_paths):
    """
    Save images through iterations
    """
    for idx, img in enumerate(images):
        util_segnet.save_image(img, file_paths[idx])


class Solver(object):
    """
    Train the classic segnet
    # model and data
    :param net: designed segnet network
    :param log_solver: true if the training params should be logged
    :param pre_trained_ckpt: pre-trained model in ckpt file
    :param test_ckpts: file-paths of the test-ckpts
    :param test_names: names of the test-ckpts
    :param train_data_dic: dict storing training dataSet with keys: "x" denotes images and "y" denotes labels
    :param verification_data_dic: dict storing testing dataSet with keys: "x" denotes images and "y" denotes labels
    # learning params
    :param nb_epoch: number of training epoch
    :param nb_training: number of training samples
    :param nb_verification: number of verification samples
    :param train_batch_size: size of training batch
    :param ver_batch_size: size of verification batch
    :param lr: learning rate
    :param dr: decay rate
    :param momentum: momentum
    :param dropout_prob: probability on dropout
    # name and path
    :param op_name: name of the optimizer to use (momentum or adam)
    :param ckpt_folder_path: folder path for saving trained model and evaluation result
    :param pred_folder_path: folder path for saving prediction result
    # flag
    :param norm_grads: true if normalized gradients should be added to the summaries
    :param restore: true if previous model should be restored
    :param write_graph: true if the computation graph should be written as protobuf file to the output path
    :param store_test: true if the testing result should be saved
    """

    def __init__(self, net, log_solver, restore, store_test, **opt_kwargs):
        # model
        self.net = net
        self.model_name = self.net.model_name
        self.log_solver = log_solver
        self.restore = restore
        self.store_test = store_test
        self.pre_trained_ckpt = opt_kwargs.pop("pre_trained_ckpt")
        self.test_ckpts = opt_kwargs.pop("test_ckpts")
        self.test_names = opt_kwargs.pop("test_names")
        # data
        self.train_data_dict = opt_kwargs.pop("train_data_dict")
        self.verification_data_dict = opt_kwargs.pop("verification_data_dict")
        self.org_x_shape = opt_kwargs.pop("org_x_shape")
        self.org_y_shape = opt_kwargs.pop("org_y_shape")
        # params
        self.nb_epoch = opt_kwargs.pop("nb_epoch")
        self.nb_training = opt_kwargs.pop("nb_training")
        self.nb_verification = opt_kwargs.pop("nb_verification")
        self.batch_size = self.net.batch_size
        self.lr = opt_kwargs.pop("learning_rate", 0.001)
        self.dr = opt_kwargs.pop("decay_rate", 0.95)
        self.momentum = opt_kwargs.pop("momentum", 0.2)
        self.dropout_prob = opt_kwargs.pop("dropout_prob", 0.5)
        # optimizer
        self.op_name = opt_kwargs.pop("op_name")
        self.lr_node, self.train_op = self._optimizer()
        # key paths
        self.ckpt_folder_path = opt_kwargs.pop("ckpt_folder_path")
        self.pred_folder_path = opt_kwargs.pop("pred_folder_path")
        # more
        self.top, self.bottom, self.left, self.right = util_segnet.border_extend_params(org_shape=self.org_x_shape,
                                                                                        target_shape=self.net.x.shape)
        self.training_iters = np.int(np.ceil(np.float(self.nb_training) / np.float(self.batch_size)))
        self.verification_iters = np.int(np.ceil(np.float(self.nb_verification) / np.float(self.batch_size)))
        # build related folders and log some info
        self._initializer()

    def _optimizer(self):
        """
        Build the training-optimizer
        """
        # create placeholder
        lr_node = None
        train_op = None
        # create training optimizer
        global_step = tf.Variable(0, name="global_step")
        if self.op_name == "momentum":
            lr_node = tf.train.exponential_decay(learning_rate=self.lr,
                                                 global_step=global_step,
                                                 decay_steps=self.batch_size,
                                                 decay_rate=self.dr,
                                                 staircase=True)
            train_op = tf.train.MomentumOptimizer(learning_rate=lr_node, momentum=self.momentum
                                                  ).minimize(self.net.cost, global_step=global_step)
        elif self.op_name == "adam":
            lr_node = tf.Variable(self.lr, name="learning_rate")
            train_op = tf.train.AdamOptimizer(learning_rate=lr_node).minimize(self.net.cost, global_step=global_step)
        return lr_node, train_op

    def _initializer(self):
        """
        Initialize the training process
        """
        # create folders
        # model.ckpt path
        if not os.path.exists(self.ckpt_folder_path):
            logging.info("Allocating '{:}'".format(self.ckpt_folder_path))
            os.makedirs(self.ckpt_folder_path)
        # prediction path
        if not os.path.exists(self.pred_folder_path):
            logging.info("Allocating '{:}'".format(self.pred_folder_path))
            os.makedirs(self.pred_folder_path)
        # log training-info
        if self.log_solver:
            logging.info("Solver Information")
            logging.info("\r\n" +
                         "restore: " + str(self.restore) + "\r\n" +
                         "store_test: " + str(self.store_test) + "\r\n" +
                         "original shape of train-x-data: " + str(self.org_x_shape) + "\r\n" +
                         "original shape of train-y-data: " + str(self.org_y_shape) + "\r\n" +
                         "nb_training: " + str(self.nb_training) + "\r\n" +
                         "nb_verification: " + str(self.nb_verification) + "\r\n" +
                         "nb_epoch: " + str(self.nb_epoch) + "\r\n" +
                         "batch size: " + str(self.batch_size) + "\r\n" +
                         "learning rate: " + str(self.lr) + "\r\n" +
                         "decay rate: " + str(self.dr) + "\r\n" +
                         "momentum: " + str(self.momentum) + "\r\n" +
                         "dropout prob: " + str(self.dropout_prob) + "\r\n" +
                         "optimizer name: " + str(self.op_name) + "\r\n" +
                         "ckpt folder path: " + str(self.ckpt_folder_path) + "\r\n" +
                         "pred folder path: " + str(self.pred_folder_path) + "\r\n" +
                         "extend-top: " + str(self.top) + "\r\n" +
                         "extend-bottom: " + str(self.bottom) + "\r\n" +
                         "extend-left: " + str(self.left) + "\r\n" +
                         "extend-right: " + str(self.right) + "\r\n" +
                         "nb_training_iters: " + str(self.training_iters) + "\r\n" +
                         "nb_verification_iters: " + str(self.verification_iters) + "\r\n")

    def train(self):
        """
        Launch the training process
        """
        # load training data
        train_x_filepaths = self.train_data_dict["x_filepaths"]
        train_y_filepaths = self.train_data_dict["y_filepaths"]
        # set config
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # session start
        with tf.Session() as sess:
            logging.info("Start training session")
            sess.run(tf.global_variables_initializer())
            if self.restore:
                self.net.restore(sess, self.pre_trained_ckpt)
                logging.info("Restore pre-trained model: " + self.pre_trained_ckpt)
            logging.info("Start optimization")
            Loss = np.ndarray(shape=[self.nb_epoch, ], dtype=np.float32)
            # start training
            for cur_epoch in np.arange(self.nb_epoch):
                # shuffle the training data indexes per training epoch
                np.random.seed()
                np.random.shuffle(train_x_filepaths)
                np.random.seed()
                np.random.shuffle(train_y_filepaths)
                # initialize params
                start_time = time.time()
                start_idx = 0
                total_loss = 0
                # batch iteration
                for cur_step in np.arange(self.training_iters):
                    print("Batch training: " + str(cur_step + 1) + "/" + str(self.training_iters)
                          + " in epoch " + str(cur_epoch + 1))
                    end_idx = start_idx + self.batch_size
                    if start_idx >= self.nb_training:
                        break
                    if end_idx > self.nb_training:
                        end_idx = self.nb_training
                    cur_idxs = np.arange(start_idx, end_idx)
                    # load cur_batch_data
                    batch_x, batch_y = util_segnet.load_np_xy(x_filepaths=train_x_filepaths[cur_idxs],
                                                              y_filepaths=train_y_filepaths[cur_idxs],
                                                              x_shape=self.org_x_shape,
                                                              y_shape=self.org_y_shape,
                                                              load_size=self.batch_size)
                    # build feed_dict
                    feed_dict = {self.net.x: util_segnet.extend_border(batch=batch_x,
                                                                       top=self.top,
                                                                       bottom=self.bottom,
                                                                       left=self.left,
                                                                       right=self.right),
                                 self.net.y: util_segnet.extend_border(batch=batch_y,
                                                                       top=self.top,
                                                                       bottom=self.bottom,
                                                                       left=self.left,
                                                                       right=self.right),
                                 self.net.keep_prob: self.dropout_prob,
                                 self.net.phase_train: True}
                    # train
                    _, loss, cur_lr = sess.run([self.train_op, self.net.cost, self.lr_node], feed_dict=feed_dict)
                    # update loss and batch-idx
                    total_loss += loss
                    start_idx = end_idx
                # result monitor
                Loss[cur_epoch] = np.float(total_loss) / np.float(self.nb_epoch)
                logging.info("Epoch " + str(cur_epoch + 1) +
                             ", Average Loss: " + str(Loss[cur_epoch]) + ", learning rate: " + str(cur_lr))
                print("epoch " + str(cur_epoch + 1) + ": " + str(time.time() - start_time) + "s")
                # save the current model
                self.net.save(sess, self.ckpt_folder_path + self.model_name + "_epoch_" + str(cur_epoch + 1) + ".ckpt")
            # save loss
            util_segnet.save_pickle(Loss, self.ckpt_folder_path + self.model_name + "_loss.pkl")
            logging.info("Optimization Finished")

    def test(self):
        """
        Launch the training process
        """
        logging.info("Start Testing")
        # build folder
        test_folder_paths = []
        if self.store_test:
            for j in np.arange(len(self.test_ckpts)):
                cur_folder_path = self.pred_folder_path + self.test_names[j] + "/"
                test_folder_paths += cur_folder_path
                if not os.path.exists(cur_folder_path):
                    os.mkdir(cur_folder_path)
                    logging.info("Allocating '{:}'".format(cur_folder_path))
        # load test data
        veri_x_filepaths = self.verification_data_dict["x_filepaths"]
        veri_y_filepaths = self.verification_data_dict["y_filepaths"]
        # config setting
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # session start
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            APs = np.ndarray(shape=[len(self.test_ckpts), ], dtype=np.float)
            # iters on test models
            for i, each_ckpt in enumerate(self.test_ckpts):
                self.net.restore(sess, each_ckpt)
                Precisions = []
                Imgs = []
                Predictions = []
                start_time = 0
                start_idx = 0
                for cur_step in np.arange(self.verification_iters):
                    print("Batch testing: " + str(cur_step + 1) + "/" + str(self.verification_iters)
                          + " in test-ckpt-" + self.test_names[cur_step])
                    end_idx = start_idx + self.batch_size
                    if start_idx >= self.nb_verification:
                        break
                    if end_idx > self.nb_verification:
                        end_idx = self.nb_verification
                    cur_idxs = np.arange(start_idx, end_idx)
                    batch_x, batch_y = util_segnet.load_np_xy(x_filepaths=veri_x_filepaths[cur_idxs],
                                                              y_filepaths=veri_y_filepaths[cur_idxs],
                                                              x_shape=self.org_x_shape,
                                                              y_shape=self.org_y_shape,
                                                              load_size=self.batch_size)
                    # prediction
                    _, batch_precision, batch_img, batch_prediction = self.prediction_generator(sess,
                                                                                                batch_x,
                                                                                                batch_y)
                    Precisions += batch_precision
                    Imgs += batch_img
                    Predictions += batch_prediction
                    # update batch-idx
                    start_idx = end_idx
                    # store current AP and pred_imgs into array
                APs[i] = np.mean(np.array(Precisions), dtype=np.float)
                # save pred_imgs
                if self.store_test:
                    Img_file_paths = [test_folder_paths[i] + self.model_name + "_img_" + str(j + 1) + ".jpg"
                                      for j in np.arange(len(Imgs))]
                    img_saver(np.concatenate(Imgs, axis=0), Img_file_paths)
                    util_segnet.save_hickle(Predictions, test_folder_paths[i] + self.test_names[i] + "_predictions.hkl")
                    print("store result-imgs and predictions")
                logging.info("Verification precision of " + each_ckpt + ": " + str(100.0 * APs[id]) + "%")
                # process-monitor
                print("test-model " + str(i + 1) + ": " + str(time.time() - start_time) + "s")
            # store APs
            if self.store_test:
                util_segnet.save_pickle(APs, self.pred_folder_path + self.model_name + "_APs.pkl")
                logging.info("Store the APs of model: " + self.model_name)
            logging.info("Testing Finished")

    def prediction_generator(self, sess, batch_x, batch_y):
        """
        Generate predicted results
        :return: shape of the prediction, predicted precisions, predicted segmented images, and the predictions
        """
        batch_prediction = sess.run(self.net.predictor,
                                    feed_dict={self.net.x: util_segnet.extend_border(batch=batch_x,
                                                                                     top=self.top,
                                                                                     bottom=self.bottom,
                                                                                     left=self.left,
                                                                                     right=self.right),
                                               self.net.y: util_segnet.extend_border(batch=batch_y,
                                                                                     top=self.top,
                                                                                     bottom=self.bottom,
                                                                                     left=self.left,
                                                                                     right=self.right),
                                               self.net.keep_prob: 1.,
                                               self.net.phase_train: False})
        # crop bach the scale of output
        batch_prediction = util_segnet.border_crop(batch=batch_prediction,
                                                   top=self.top,
                                                   bottom=self.bottom,
                                                   left=self.left,
                                                   right=self.right)
        batch_precision = precision_rate(batch_prediction, batch_y, len(batch_y))
        batch_img_array = util_segnet.prediction_img_generator(batch_x, batch_y, batch_prediction, self.net.nb_class)
        return batch_prediction.shape, batch_precision, batch_img_array, batch_prediction
