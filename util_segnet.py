# Author: Jingwei Guo
# Date: 10-31-2018
# function module

from __future__ import print_function, division, absolute_import
from keras.utils import to_categorical
from PIL import Image
import tensorflow as tf
import cPickle as pickle
import numpy as np
import hickle
import cv2


def _variable_on_gpu_0(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, initializer, wd):
    """
    Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = _variable_on_gpu_0(
        name,
        shape,
        initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def load_np_xy(x_filepaths, y_filepaths, x_shape, y_shape, load_size):
    """
    Load x-y data in .npy by batch size
    """
    batch_x = np.ndarray(shape=[load_size, x_shape[1], x_shape[2], x_shape[3]])
    batch_y = np.ndarray(shape=[load_size, y_shape[1], y_shape[2], y_shape[3]])
    for i in np.arange(load_size):
        batch_x[i] = np.load(x_filepaths[i])
        batch_y[i] = np.load(y_filepaths[i])
    # use label-index instead of one-hot in batch-y
    batch_y = np.argmax(batch_y, axis=3)
    batch_y = np.expand_dims(batch_y, axis=3)
    return batch_x, batch_y


def border_crop(batch, top, bottom, left, right):
    """
    Crops the border of the input with shape [nb, nx, ny, channel].
    """
    return batch[:, top:-bottom, left:-right, :]


def border_extend_params(org_shape, target_shape):
    """
    Return the params for border extending
    """
    top = (target_shape[1] - org_shape[1]) // 2
    bottom = target_shape[1] - org_shape[1] - top
    left = (target_shape[2] - org_shape[2]) // 2
    right = target_shape[2] - org_shape[2] - left
    return np.int(top), np.int(bottom), np.int(left), np.int(right)


def extend_border(batch, top, bottom, left, right):
    """
    Extend the border of input with shape [nb, nx, ny, channel]
    """
    # left-right extending
    batch = np.array(
        [np.array([np.concatenate((np.tile(row[0], (left, 1)), row, np.tile(row[-1], (right, 1))), axis=0)
                   for row in per]) for per in batch])
    # top-bottom extending
    batch = np.array([np.concatenate((np.tile(per[0], (top, 1, 1)), per, np.tile(per[-1], (bottom, 1, 1))), axis=0)
                      for per in batch])
    return batch


def concatenate_segment(batch, nb_class):
    """
    Combine the segments in batch-sample
    :param batch: batch-sample
    :param nb_class: number of the segmented class
    """
    return np.array([np.concatenate((per[..., i] for i in np.arange(nb_class)), axis=0) for per in batch])


def prediction_img_generator(batch_x, batch_y, batch_pred, nb_class):
    """
    Generate the integral result-img
    :param batch_x: batch-image
    :param batch_y: batch-gt-label
    :param batch_pred: batch-prediction
    :param nb_class: number of the segmented class
    :return: integral result-img in array
    """
    # prediction to one-hot
    batch_pred = to_categorical(np.argmax(batch_pred, axis=3))
    # one-hot to RGB
    batch_y *= 255
    batch_pred *= 255
    # reshape
    batch_shape = batch_x.shape
    batch_y = concatenate_segment(batch_y, nb_class)
    batch_pred = concatenate_segment(batch_pred, nb_class)
    batch_x = np.array([cv2.copyMakeBorder(per,
                                           top=batch_shape[1] // 2,
                                           bottom=batch_shape[1] - batch_shape[1] // 2,
                                           left=0,
                                           right=0,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=0) for per in batch_x])
    return np.array([np.concatenate((x, y, z), axis=1) for x, y, z in zip(batch_x, batch_y, batch_pred)])


def save_image(img_array, path):
    """
    Writes the image-array to disk
    :param img_array: img in array format
    :param path: the target path
    # # , 'JPEG', dpi=[300, 300], quality=90
    """
    Image.fromarray(img_array.round().astype(np.uint8)).save(path)


# load hickle
def load_hickle(path):
    print('Loaded ' + path + '..')
    return hickle.load(path)


# save data in hickle
def save_hickle(data, path):
    print('Saved ' + path + '..')
    hickle.dump(data, path)


# load pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file


# save data in pickle
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path)
