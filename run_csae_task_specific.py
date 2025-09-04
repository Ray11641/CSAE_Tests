"""
This file supposedly runs the CSAE model. Please reference to
csae_model.py or https://github.com/JohnNellas/CSAE for more
details.
"""

import os                              #builtin library imports here
import sys
import logging
import argparse
import typing
from typing import Dict, Tuple
from collections.abc import Callable
from datetime import datetime as dt
import pandas as pd                    #external imports from here
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from csae_model import *              #custom imports from here
from utils_csae import *


__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2024"
__credits__ = ["Donald Wunsch, Leonardo Enzo Brito Da Silva, Sasha Petrenko"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Leonardo Enzo Brito Da Silva"
__email__ = "lb284@mst.edu"
__status__ = "Development"
__date__ = "2024.05.25"


#Constants
CHUNK_SIZE = 1000
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = [16, 25, 32, 50, 64, 75, 128, 256, 512, 1024]
EPOCHS = 100
NUM_CLASSES = {
    'OPTDIGITS': 10,
    'USPS': 10,
    'MNIST': 10,
    'OLIVETTI-FACES': 40,
    'CIFAR-10': 10,
    'FASHION-MNIST': 10,
    'COVERTYPE': 7
}

ENCODER_DENSE_TOPOLOGIES = [
    [128, 128],
    [256, 128, 64],
    [256, 256, 128],
    [256, 256, 128, 64]]

ENCODER_TOPOLOGY_INDICES = list(range(len(ENCODER_DENSE_TOPOLOGIES)))

#logger setup
current_time = dt.now().strftime("D%Y-%m-%d_T%H-%M-%S")
logger_csae = logging.getLogger("csaelogger")
log_dir = os.path.join(os.getcwd(), 'logs')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
handler = logging.FileHandler(os.path.join(log_dir, f"csae_logger_{current_time}.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                             datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger_csae.addHandler(handler)
logger_csae.setLevel(logging.DEBUG)
logger_csae.info("Running Convolutional Supervised Auto Encoder")



def run_task_convolutional_supervised_auto_encoder(dataset_name: str,
                                                  train_data: DataObject,
                                                  latent_dimensions: int,
                                                  config: Dict,
                                                  tasks: List[int],
                                                  current_run: int,
                                                  test_data: DataObject = None,
                                                  batch_size: int = 32,
                                                  logger: logging.Logger = logger_csae) -> Callable:
    """
    This function should run CSAE for given dataset

    Arguments
    :param dataset_name: name of the dataset to be loaded
    :param train_data: data for training the csae_model
    :param test_data: data for testing the model
    :param latent_dimensions: the number of latent dimensions for the data
    :param config: the config for running the current set of experiments
                   config should include the following.
                   [1] Kernel sizes
                   [2] Filters
                   [3] Strides
                   [4] Learning rate
                   [5] Classifier Units (dense)
                   [6] Autoencoder Units (dense)
    :param tasks:
    :param current_run: current run step
    :param batch_size: the batch size to be used for the data

    Return
    :unknown: I have no idea what to output at this point
    """

    try:
        if latent_dimensions <= 0:
            raise Exception("Latent space dimensions should be greater than 0")
    except Exception as e:
        logger.exception(e)
        logger.info("Terminating the run")
        sys.exit()

    try:
        config_contents = ["kernels",
                           "filters",
                           "strides",
                           "learning_rate",
                           "classifer_units",
                           "autoencoder_units"]
        if len(config.keys()) == 0:
            raise ValueError(f"Empty config provided. \n config needs to be dictionary with following values {config_contents}")
        elif not any([(x in config.keys()) for x in config_contents]):
            raise ValueError(f"Missing value in config. \n config needs to be dictionary with following values {config_contents}")
    except Exception as e:
        logger.exception(e)
        logger.info("Terminating run")
        sys.exit()


    logger.info("Building autoencoder for CSAE")
    #Build model
    ae, clsfr = build_fixed_ae_model(input_shape = train_data.X[0].shape,
                                    latent_dims = latent_dimensions,
                                    n_classes = NUM_CLASSES[dataset_name],
                                    filters = config["filters"],
                                    kernels = config["kernels"],
                                    strides = config["strides"],
                                    ae_dense_units = config["autoencoder_units"],
                                    classifier_dense_units = config["classifer_units"])

    logger.debug("Chosing Adam optimiser with learning rate {}".format(config["learning_rate"]))
    optimizer = tf.keras.optimizer.Adam(learning_rate = config["learning_rate"])
    logger.debug(f"AE loss set to MSE and Classifier loss set to Sparse Categorical Crossentropy")
    autoencoder_loss_fcn = tf.keras.losses.MeanSquaredError()
    classifier_loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
    ae_metric = tf.keras.metrics.MeanSquaredError(name="ae_loss")
    classifier_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False,
                                                            name="classifier_loss")
    logger.info("Build successful")
    logger.debug(f"Following is model summary: \n {ae.summary()} \n {clsfr.summary()}")
    logger.info("Building CSAE")
    model_csae = CSAE(ae, clsfr)
    model_csae.build((None, train_data.shape[0],train_data.shape[1],train_data.shape[2]))
    model_csae.compile(optimizer,
                       optimizer,
                       classifier_loss_fcn,
                       autoencoder_loss_fcn,
                       classifier_metric,
                       ae_metric)
    logger.debug(f"Build successful and summary \n {model_csae.summary()}")
    logger.info(f"Training the CSAE model")

    for i, current_task in enumerate(tasks):
        logger.info(f"Fitting class (es), {current_task}")
        train_X, train_y, val_X,val_y = get_task_train_and_val_data_csae(train_data = train_data,
                                                                        current_task = current_task,
                                                                        current_run = current_run)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y))

        if i == 0:
            pretraining_flag = True
        else:
            pretraining_flag = False

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
        val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_classifier_loss',
                                                      mode = "min",
                                                      patience = 5),
                    tf.keras.callbacks.ModelCheckpoint(filepath = "checkpoint.model.keras",
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       save_freq='epoch',
                                                       initial_value_threshold=None)]
        history = model_csae.fit(train_dataset,
                                epochs = EPOCHS,
                                validation_data = val_dataset,
                                callbacks = callbacks,
                                verbose = 1)

    return model_csae, history, test_results


if __name__ == "__main__":
    """
    Commence the run of the simulations
    """
    print("Running CSAE")
