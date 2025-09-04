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
import json
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
OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Tensorflow setup
if tf.executing_eagerly():
    print("Eager execution is enabled.")
else:
    print("Eager execution is disabled.")
    tf.compat.v1.enable_eager_execution() # if needed, enable it.

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

def get_config(kernels: List[int],
               filters: List[int],
               strides: List[int],
               learning_rate: float,
               classifer_units: List[int],
               autoencoder_units: List[int],
               logger: logging.Logger):
    """
    Arguments
    :param kernels: number of kernels for the 
    :param filters: 
    :param strides: 
    :param learning_rate: 
    :param classifer_units: 
    :param autoencoder_units:
    :param logger:
    
    Return
    :config: config for the running experiments
    """
    try:
        if len(kernels) != len(filters) or len(filters) != len(strides):
            raise Excpetion("Length of kernels, filters, and strides do not match")
    except Exception as e:
        logger.exception(e)
    
    config = {}
    config["kernels"] = kernels
    config["filters"] = filters
    config["strides"] = strides
    config["learning_rate"] = learning_rate
    config["classifer_units"] = classifer_units
    config["autoencoder_units"] = autoencoder_units
    return config

def run_convolutional_supervised_auto_encoder(dataset_name: str,
                                              train_data: DataObject,
                                              test_data: DataObject,
                                              latent_dimensions: int,
                                              config: Dict,
                                              current_run: int,
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
    optimizer = tf.keras.optimizers.Adam(learning_rate = config["learning_rate"])
    logger.debug(f"AE loss set to MSE and Classifier loss set to Sparse Categorical Crossentropy")
    autoencoder_loss_fcn = tf.keras.losses.MeanSquaredError()
    classifier_loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
    ae_metric = tf.keras.metrics.MeanSquaredError(name="ae_loss")
    classifier_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False,
                                                            name="classifier_loss")    
    logger.info("Build successful") 
    #logger.debug(f"Following is model summary: \n {ae.summary()} \n {clsfr.summary()}")
    logger.info("Building CSAE")
    model_csae = CSAE(ae, clsfr,ae_metric, classifier_metric)
    model_csae.build((None, train_data.shape[0],train_data.shape[1],train_data.shape[2]))
    model_csae.compile(optimizer,
                       optimizer,
                       classifier_loss_fcn,
                       autoencoder_loss_fcn)
    
    logger.debug(f"Build successful and summary")
    logger.debug(model_csae.summary())
    logger.info(f"Training the CSAE model")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data.X, train_data.y))
    val_dataset = tf.data.Dataset.from_tensor_slices((test_data.X, test_data.y))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_classifier_loss',
                                                  mode = "min",
                                                  patience = 5)]

#                tf.keras.callbacks.ModelCheckpoint(filepath = "checkpoint.model.keras",
#                                                   monitor='val_classifier_loss',
#                                                   verbose=0,
#                                                   save_best_only=True,
#                                                   save_weights_only=False,
#                                                   mode='auto',
#                                                   save_freq='epoch',
#                                                   initial_value_threshold=None)]

    history = model_csae.fit(train_dataset,
                        batch_size = batch_size,
                        epochs = EPOCHS,
                        validation_data = val_dataset,
                        callbacks = callbacks,
                        verbose = 1)
        
    return model_csae, history


def grid_search(dataset_name: str,
                logger: logging.Logger,
                n_folds: int = 10) -> Callable:
    """
    Arguments
    :param dataset_name: dataset to be used for the grid search
    :param logger: logging object to record the logs
    :param n_folds: number of folds for k-fold validation

    Returns
    : : Need to think
    """
    
    topology_indices, mini_batch_sizes = np.meshgrid(ENCODER_TOPOLOGY_INDICES, BATCH_SIZE)
    topology_indices = topology_indices.flatten()
    mini_batch_sizes = mini_batch_sizes.flatten()
    logger.info(f"Dataset being used: {dataset_name}")
    data = get_subset_train_and_val_data(dataset_name, logger, 0.3)
    logger.info(f"Number of hyper-parameter combinations: {len(topology_indices)}")
    logger.info(f"Number of dataset folds for crossvalidation = {n_folds}")
    for ind, (topology_index, mini_batch_size) in enumerate(zip(topology_indices, mini_batch_sizes)):
        topology = ENCODER_DENSE_TOPOLOGIES[topology_index]
        config = get_config([3,3],
                            [32, 64],
                            [2,2],
                            0.001,
                            [128, 128],
                            topology,
                            logger)
        logger.debug("Config for current run")
        for key, value in config.items():
            logger.debug(f"{key} : {value}")
        skf = StratifiedKFold(n_splits = n_folds,
                              shuffle = False,
                              random_state = None)
        train_error = np.zeros(n_folds)
        val_error = np.zeros(n_folds)
        train_accuracy = np.zeros(n_folds)
        val_accuracy = np.zeros(n_folds)
        for i, (train_index, val_index) in enumerate(skf.split(data.X, data.y)):
            logger.info(f"Commencing the {i} fold of training")
            train_data = DataObject(data.X[train_index], data.y[train_index])
            test_data = DataObject(data.X[val_index], data.y[val_index])
            model, history = run_convolutional_supervised_auto_encoder(dataset_name = "MNIST",
                                                                      train_data = train_data,
                                                                      test_data = test_data,
                                                                      latent_dimensions = 2,
                                                                      config = config,
                                                                      current_run = i,
                                                                      batch_size = mini_batch_size,
                                                                      logger = logger)
            train_error[i] = max(history.history['classifier_loss'])
            val_error[i] = max(history.history['val_classifier_loss'])
            train_accuracy[i] = max(history.history['accuracy'])
            val_accuracy[i] = max(history.history['val_accuracy'])
            logger.info(f"The {i} fold of training ended with best values as following:")
            logger.info(f"The best train error = {train_error[i]}")
            logger.info(f"The best val error = {val_error[i]}")
            logger.info(f"The best train accuracy = {train_accuracy[i]}")
            logger.info(f"The best train error = {val_accuracy[i]}")
            
        logger.info("Results for all folds (as averages):")
        logger.info(f"The avg train error = {np.mean(train_error)}")
        logger.info(f"The avg val error = {np.mean(val_error)}")
        logger.info(f"The avg train accuracy = {np.mean(train_accuracy)}")
        logger.info(f"The avg train error = {np.mean(val_accuracy)}")
        config["batch_size"] = int(mini_batch_size)
        config["train_error"] = np.mean(train_error)
        config["val_error"] = np.mean(val_error)
        config["train_accuracy"] = np.mean(train_accuracy)
        config["val_accuracy"] = np.mean(val_accuracy)
        results_dir = os.path.join(OUTPUT_DIR, dataset_name)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        with open(os.path.join(results_dir, f'combination_{ind}.json'), 'a', encoding = "utf-8") as file:
            json.dump(config, file)
        


if __name__ == "__main__":
    """
    Commence the run of the simulations
    """
    print("Commencing grid search for CSAE")
    datasets = ['MNIST-Train','Fashion-MNIST-Train', 'USPS']
    for dataset in datasets:
        grid_search(dataset_name = dataset,
                        logger = logger_csae,
                        n_folds = 10)
