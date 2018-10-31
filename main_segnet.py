# Author: Jingwei Guo
# Date: 10-31-2018
# the same structure to U-net in coding

from model_segnet import Segnet
from solver_segnet import Solver
import numpy as np
import util_segnet
import logging
import os


# global name
model_name = "basic_segnet_1"

# global path
data_folder_path = "/DATA2/Jingwei/RA_XJTLU/chromosome_segmentation/cs_data/"
cur_net_folder_path = "/DATA2/Jingwei/RA_XJTLU/chromosome_segmentation/cs_result/Seg_net/"
cur_model_folder_path = cur_net_folder_path + model_name + "/"
ckpt_folder_path = cur_model_folder_path + "ckpt_" + model_name + "/"
pred_folder_path = cur_model_folder_path + "pred_" + model_name + "/"


# data-loader
def load_data():
    # set cur_data path
    cur_data_indexes_folder_path = data_folder_path + ""

    # load data indexes
    x_filepaths = util_segnet.load_pickle(cur_data_indexes_folder_path + "x_filepaths.pkl")
    y_filepaths = util_segnet.load_pickle(cur_data_indexes_folder_path + "y_filepaths.pkl")

    # set dataset separation-params
    nb_data = len(x_filepaths)

    # x-train, val, test
    train_x_filepaths = x_filepaths[:np.int(0.7 * nb_data)]
    val_x_filepaths = x_filepaths[np.int(0.7 * nb_data):np.int(0.85 * nb_data)]
    test_x_filepaths = x_filepaths[np.int(0.85 * nb_data):]

    # y-train, val, test
    train_y_filepaths = y_filepaths[:np.int(0.7 * nb_data)]
    val_y_filepaths = y_filepaths[np.int(0.7 * nb_data):np.int(0.85 * nb_data)]
    test_y_filepaths = y_filepaths[np.int(0.85 * nb_data):]

    # construct data-indexes-dict
    train_data_dict = {"x_filepaths": train_x_filepaths,
                       "y_filepaths": train_y_filepaths}
    val_data_dict = {"x_filepaths": val_x_filepaths,
                     "y_filepaths": val_y_filepaths}
    test_data_dict = {"x_filepaths": test_x_filepaths,
                      "y_filepaths": test_y_filepaths}
    verification_data_dict = {
        "x_filepaths": np.array(list(val_data_dict["x_filepaths"]) + list(test_data_dict["x_filepaths"])),
        "y_filepaths": np.array(list(val_data_dict["y_filepaths"]) + list(test_data_dict["y_filepaths"]))}

    return train_data_dict, val_data_dict, test_data_dict, verification_data_dict


def initializer():
    # create net-folder
    if not os.path.exists(cur_net_folder_path):
        os.mkdir(cur_net_folder_path)
        print("Allocating net-folder: " + cur_net_folder_path)
    # create model-folder
    if not os.path.exists(cur_model_folder_path):
        os.mkdir(cur_model_folder_path)
        print("Allocating model-folder: " + cur_model_folder_path)


def main():
    # initialize the current model-folder
    initializer()

    # set log-config for printing into .log
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        filename=cur_model_folder_path + "U_net_" + model_name + ".log",
                        filemode="w")
    # set StreamHandler for printing into screen
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # load data
    pre_trained_ckpt = None
    test_ckpts = None
    test_names = None
    train_data_dict, val_data_dict, test_data_dict, verification_data_dict = load_data()

    # data-params
    nb_training = len(train_data_dict["x_filepaths"])
    nb_verification = len(verification_data_dict["x_filepaths"])
    nb_channel = 1
    nb_class = 4

    # build model
    segnet = Segnet(model_name=model_name,
                    log_net=True,
                    channels=nb_channel,
                    nb_class=nb_class,
                    img_h=184,
                    img_w=184,
                    batch_size=20,
                    cost_name="cross_entropy",
                    regularizer=None,
                    is_bw_loss=False)

    # train model
    segnet_solver = Solver(net=segnet,
                           log_solver=True,
                           restore=False,
                           store_test=True,
                           pre_trained_ckpt=pre_trained_ckpt,
                           test_ckpts=test_ckpts,
                           test_names=test_names,
                           train_data_dict=train_data_dict,
                           verification_data_dict=verification_data_dict,
                           org_x_shape=(nb_training, 175, 169, nb_channel),
                           org_y_shape=(nb_training, 175, 169, nb_class),
                           nb_epoch=150,
                           nb_training=nb_training,
                           nb_verification=nb_verification,
                           batch_size=20,
                           lr=0.001,
                           dr=0.95,
                           momentum=0.2,
                           dropout_prob=0.5,
                           op_name="momentum",
                           ckpt_folder_path=ckpt_folder_path,
                           pred_folder_path=pred_folder_path)

    # train and test
    segnet_solver.train()
    segnet_solver.test()


main()
