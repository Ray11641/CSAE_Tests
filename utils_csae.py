"""
This file provides utilities for CSAE and is based on csae_model.py
from the source. Defaults in build_ae_model, build_ae_incremental_model
are set to the source defaults.

Source:
https://github.com/JohnNellas/CSAE

Reference: 
"Nellas, I.A., Tasoulis, S.K., Plagianakos, V.P. and Georgakopoulos, S.V., 
2022. Supervised Dimensionality Reduction and Image Classification 
Utilizing Convolutional Autoencoders. arXiv preprint arXiv:2208.12152."
"""

import os
import typing
from typing import Tuple, List, Any
from collections.abc import Callable
import logging
import numpy as np
from numpy import save
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
#Datasets
from sklearn.datasets import fetch_olivetti_faces, fetch_covtype, make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, PILToTensor, Grayscale, Compose
from ucimlrepo import fetch_ucirepo


__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2024"
__credits__ = ["Donald Wunsch, Leonardo Enzo Brito Da Silva, Sasha Petrenko"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Leonardo Enzo Brito Da Silva"
__email__ = "lb284@mst.edu"
__status__ = "Development"
__date__ = "2024.05.25"


def build_fixed_ae_model(input_shape: Tuple[int],
                        latent_dims: int,
                        n_classes: int,
                        filters: Tuple[int] = (32, 64),
                        kernels: Tuple[int] | Tuple[Tuple[int]] = (3,3),
                        strides: Tuple[int] | Tuple[Tuple[int]] = (2,2),
                        ae_dense_units: Tuple[int] = (128,128),
                        classifier_dense_units: Tuple[int] = (128,128)) -> Tuple[tf.keras.Model]:
    """
    Arguments
    :param input_shape: shape of the data
    :param latent_dims: number of latent dimensions for encoder
    :param n_classes: number of classifier classes
    :param filters: n filters for the convolutional layer, default
                    is set to (32, 64)
    :param filters: n filters for the convolutional layer, default
                    is set to (32, 64)
    :param stride: n strides for the convolutional layer, default
                    is set to (2, 2)                
    :param ae_dense_units: encoder/decoder dense units, default set
                    to (128, 128)
    :param classifier_dense_units: classifier head dense units,
                    default set to (128, 128)

    Returns
    :object autoencoder: tf.keras.Model of autoencoder
    :object classifier: tf.keras.Model of classifier
    """

    input_layer = tf.keras.Input(shape = input_shape, name = "InputLayer")
    first_filter = filters[0]
    last_filter = filters[-1]
    conv_depth = len(filters)
    
    # Encoder-conv
    for i, filter in enumerate(filters):
        if i == 0:
            x = tf.keras.layers.Conv2D(filters = filter,
                                       kernel_size = kernels[i],
                                       strides = strides[i],
                                       activation = "relu",
                                       padding = "same",
                                       name = f"Conv{i+1}")(input_layer)
        else:
            x = tf.keras.layers.Conv2D(filters = filter,
                                       kernel_size = kernels[i],
                                       strides = strides[i],
                                       activation = "relu",
                                       padding = "same",
                                       name = f"Conv{i+1}")(x)

    ignored, conv_w, conv_b, conv_c = x.shape
    print(f"conv_w = {conv_w}, conv_b = {conv_b}, conv_c = {conv_c}")
    
    x = tf.keras.layers.Flatten()(x)
    
    # Encoder-Dense
    for i, units in enumerate(ae_dense_units):
        x = tf.keras.layers.Dense(units = units,
                                  activation = "relu",
                                  name = f"Encoder{i+1}")(x)
    
    enc = tf.keras.layers.Dense(units = latent_dims,
                                activation = "linear",
                                name = "LatentLayer")(x)

    # Classifier
    for i, units in enumerate(classifier_dense_units):
        if i == 0:
            clsfr = tf.keras.layers.Dense(units = units,
                                        activation = "relu",
                                        name = f"Cls{i+1}")(enc)
        else:
            clsfr = tf.keras.layers.Dense(units = units,
                                        activation = "relu",
                                        name = f"Cls{i+1}")(clsfr)

    classification_layer = tf.keras.layers.Dense(units = n_classes,
                                                 activation = "softmax",
                                                 name = "ClassificationLayer")(clsfr)

    # Decoder Dense
    for i, units in enumerate(ae_dense_units[::-1]):
        if i == 0:
            x = tf.keras.layers.Dense(units = units,
                                      activation = "relu",
                                      name = f"Decoder{len(ae_dense_units)-i}")(enc)
        else:
            x = tf.keras.layers.Dense(units = units,
                                      activation = "relu",
                                      name = f"Decoder{len(ae_dense_units)-i}")(x)
    x = tf.keras.layers.Dense(units = conv_w*conv_b*conv_c,
                              activation = "relu",
                              name = "FlatToSpatial")(x)
    # conv_c == filters[-1]
    kernels = kernels[::-1]
    strides = strides[::-1]
    for  i, filter in enumerate(filters[::-1]):
        if i == 0:
            x = tf.keras.layers.Reshape((conv_w, conv_b, filter))(x)
        else:
            x = tf.keras.layers.Conv2DTranspose(filters = filter,
                                                kernel_size = kernels[i],
                                                strides = strides[i],
                                                activation = "relu",
                                                padding = "same",
                                                name = f"Deconv{len(filters) - i + 1}")(x)

    output = tf.keras.layers.Conv2DTranspose(filters = input_shape[-1],
                                             kernel_size = kernels[i],
                                             strides = strides[i],
                                             activation = "sigmoid",
                                             padding = "same",
                                             name = "Deconv1")(x)

    autoencoder = tf.keras.Model(inputs = input_layer,
                                 outputs = output,
                                 name = "AutoEncoder")
    classifier = tf.keras.Model(inputs = input_layer,
                                outputs = classification_layer,
                                name = "Classifier")

    autoencoder.summary()
    classifier.summary()

    return autoencoder, classifier

def build_incremental_ae_model(input_shape: Tuple[int],
                        latent_dims: int,
                        n_classes: int,
                        filters: Tuple[int] = (32, 64),
                        ae_dense_units: Tuple[int] = (128,128),
                        classifier_dense_units: Tuple[int] = (128,128)) -> Tuple[tf.keras.Model]:
    """
    Arguments
    :param input_shape: shape of the data
    :param latent_dims: number of latent dimensions for encoder
    :param n_classes: number of classifier classes
    :param filters: n filters for the convolutional layer, default
                    is set to (32, 64)
    :param ae_dense_units: encoder/decoder dense units, default set
                    to (128, 128)
    :param classifier_dense_units: classifier head dense units,
                    default set to (128, 128)

    Returns
    :object autoencoder: tf.keras.Model of autoencoder
    :object classifier: tf.keras.Model of classifier
    """

    pass

class DataObject():
    def __init__(self,
                 features_train: np.ndarray,
                 labels_train: np.ndarray,
                 name: str = None,
                 features_test: np.ndarray = None,
                 labels_test: np.ndarray = None,
                 transform: Callable = ToTensor(),
                 target_transform: Callable = None) -> None:
        """
        Constructor for storing the data

        Arguments
        :param features_train: data for training
        :param features_test: data for testing
        :param labels_train: labels for training
        :param labels_test: labels for testing
        """
        self.name = name
        self.X = features_train
        self.y = labels_train
        self.X_val = features_test
        self.y_val = labels_test
        self.transform = transform
        self.target_transform = target_transform
        self.shape = [features_train.shape[1],
                      features_train.shape[2],
                      features_train.shape[3]]
        
    
    def scale_min_max(self,
                      features: np.ndarray,
                      train: bool = False) -> None:
        """
        Takes the features to determine dataset max and min for scaling

        Arguments
        :param features: comprehensive dataset
        :param train: boolean to ensure that the validation set is scaled
        """
        scaler = MinMaxScaler()
        scaler.fit(features)
        self.X = scaler.transform(self.X)
        if train:
            self.X_val = scaler.transform(self.X_val)
    
    def split_validation(self,
                         indices: List[int]) -> None:
        """
        Splits the self.X to have self.X and self.X_val

        Arguments
        :param indices: a list consisting indices to be treated as validation
                        data
        """
        self.X_val = self.X[indices]
        self.y_val = self.y[indices]
        self.X = np.delete(self.X, indices, axis = 0)
        self.y = np.delete(self.y, indices, axis = 0)
    
    def normalize(self, train: bool = False) -> None:
        """
        Normalise data depending on the name of the dataset
        """
        if 'optdigits' in self.name:
            self.X /= 16.0
            if train:
                self.X_val /= 16.0
        elif self.name in ['usps', 'mnist', 'fashion-mnist', 'cifar-10']:
            self.X /= 255.0
            if train:
                self.X_val /= 255.0
        elif 'covertype' in self.name:
            features, labels = fetch_covtype(data_home='data',
                                            shuffle=True,
                                            random_state=0,
                                            download_if_missing=True,
                                            return_X_y=True)    
            self.scale_min_max(features, train)
        elif self.name in ['letter-recognition', 'statlog-landsat-satellite', 'isolet']:
            data = fetch_ucirepo(id=get_uci_id[dataset_name[:position]])
            features = data.data.features.to_numpy()            
            self.scale_min_max(features, train)


def GetDataset(dataset_name: str,
               logger: logging.Logger,
               train: bool = False) -> DataObject:
    """
    Download dataset of specified name from different sources

    Arguments
    :param dataset_name: Name of the dataset to be downloaded
    :param logger: logging object to record the logs
    :param train: flag to split data between test and train,
                  default set to False
    Returns
    :dataobject: DataObject class container
    """
    dataset_name = dataset_name.lower()
    transform = None
    position = dataset_name.rfind('-')
    get_pytorch_data = {
        'usps': datasets.USPS,
        'mnist': datasets.MNIST,
        'fashion-mnist': datasets.FashionMNIST,
        'cifar-10': datasets.CIFAR10
    }
    n_features = {
        'usps': 256,
        'mnist': 784,
        'fashion-mnist': 784,
        'cifar-10': 1024
    }
    get_uci_id = {
        'letter-recognition': 59,
        'statlog-landsat-satellite': 146,
        'isolet': 54
    }

    if dataset_name[:position] in ['letter-recognition', 'statlog-landsat-satellite', 'isolet']:
        logger.info(f"fetching {dataset_name[:position]} from UCI repository")
        data = fetch_ucirepo(id=get_uci_id[dataset_name[:position]])
        features = data.data.features.to_numpy()
        labels = data.data.targets.to_numpy().squeeze()
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        logger.debug(f"{dataset_name}: features = {features.shape}, labels = {labels.shape}")
        
    elif 'olivetti-faces' in dataset_name:
        logger.info(f"fetching {dataset_name[:position]} from SKLearn repository")
        features, labels = fetch_olivetti_faces(data_home = 'data',
                                                shuffle = True,
                                                random_state = 0,
                                                download_if_missing = True,
                                                return_X_y = True)
        
    elif 'covertype' in dataset_name:
        logger.info(f"fetching {dataset_name[:position]} from SKLearn repository")
        features, labels = fetch_covtype(data_home='data',
                                        shuffle=True,
                                        random_state=0,
                                        download_if_missing=True,
                                        return_X_y=True)
        labels -= 1 # start class from 0
        
    elif dataset_name[:position] in ['usps', 'mnist', 'fashion-mnist', 'cifar-10']:
        logger.info(f"fetching {dataset_name[:position]} from Torch repository")
        if 'cifar-10' in dataset_name:
            transform = Compose([Grayscale(), PILToTensor()])
        else:
            transform = Compose([PILToTensor()])

        data = get_pytorch_data[dataset_name[:position]](root="data",
                                                         train=True,
                                                         transform=transform,
                                                         download=True,)
        print(data)
        logger.info(f"#samples in retrieved data from Torch: {len(data)}")
        features = np.zeros((len(data),
                             data[0][0].shape[1],
                             data[0][0].shape[2],
                             data[0][0].shape[0]))
        labels = np.zeros((len(data), ), dtype=int)
        for i, (img, label) in enumerate(data):
            #features[i, :] = img.squeeze().reshape(1, n_features[dataset_name[:position]]).numpy()
            features[i, : ] = img.numpy().transpose()
            labels[i] = label
        
    else:
        error = ValueError('Data not available: {}'.format(dataset_name[:position]))
        logger.error(error)
        raise error
    if train:
        features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                    labels,
                                                                                    test_size = 0.2,
                                                                                    random_state = 0,
                                                                                    shuffle = True,
                                                                                    stratify = labels)
        
        dataset = DataObject(name = dataset_name[:position],
                             features_train = features_train,
                             labels_train = labels_train,
                             features_test = features_test,
                             labels_test = labels_test,
                             transform = transform)
        logger.info(f"Dataset shape:")
        logger.info(f"features train: {dataset.X.shape}")
        logger.info(f"labels train: {dataset.y.shape}")
        logger.info(f"features validation: {dataset.X_val.shape}")
        logger.info(f"labels validation: {dataset.y_val.shape}")
    else:
        dataset = DataObject(name = dataset_name[:position],
                             features_train = features,
                             labels_train = labels,
                             features_test = None,
                             labels_test = None,
                             transform = transform)
        logger.info(f"Dataset shape:")
        logger.info(f"features train: {dataset.X.shape}")
        logger.info(f"labels train: {dataset.y.shape}")
        logger.info(f"features validation: None")
        logger.info(f"labels validation: None")
        
    return dataset


def get_task_train_and_val_data_csae(train_data: DataObject,
                                     current_task: int,
                                     current_run: int) -> Tuple[np.ndarray]:
    """
    determines the train data appropriate for the current task
    and returns it.

    Arguments
    :param train_data: custom container, refer to 
    :param current_task: the current task details
    :param current_run: int for setting seed

    Returns
    :subset_train_X: subset of train_data corresponding to label 
                  current_task
    :subset_train_y: subset of train_labels corresponding to label 
                    current_task
    :subset_val_X: subset of train_data corresponding to label 
                  current_task used for validation
    :subset_val_y: subset of train_labels corresponding to label 
                    current_task used for validation
    """
    retain_indices = np.isin(train_data.y, current_task)
    subset_train_X = train_data.X[retain_indices, :]
    subset_train_y = train_data.y[retain_indices]
    
    retain_indices = np.isin(train_data.y_val, current_task)
    subset_val_X = train_data.X_val[retain_indices, :]
    subset_val_y = train_data.y_val[retain_indices]
    
    return subset_train_X, subset_train_y, subset_val_X, subset_val_y


def get_subset_train_and_val_data(dataset_name: str,
                                  logger: logging.Logger,
                                  subset_percent: float | int = 0.3) -> DataObject:
    """
    get the subset of train and validation data

    Arguments
    :param dataset_name: dataset to be fetched
    :param subset_percent: has to be float between (0,1) or int between (0, 100)
    :param logger: logger to record the data recovery
    """
    try:
        if isinstance(subset_percent, float):
            if subset_percent < 0 or subset_percent > 1:
                raise ValueError(f"subset_percent needs to be between (0,1) or (0,100), recieved {subset_percent}")
        elif isinstance(subset_percent, int):
            if subset_percent < 0 or subset_percent > 100:
                raise ValueError(f"subset_percent needs to be between (0,1) or (0,100), recieved {subset_percent}")
            else:
                subset_percent /= 100.0
        else:
            raise TypeError(f"subset_percent needs to be either float or int, got {type(subset_percent)}")
    except Exception as e:
        logger.error('Error at retrieving subset of data', exc_info = e)
                
    dataset = GetDataset(dataset_name, logger)
    logger.info(f"Splitting the retrieved dataset to {subset_percent*100}%")
    _, X_subset, _, y_subset = train_test_split(dataset.X,
                                                   dataset.y,
                                                   test_size = subset_percent,
                                                   random_state = 0,
                                                   shuffle = True,
                                                   stratify = dataset.y)
    logger.debug(f'train data subset shape: X: {X_subset.shape}, y: {y_subset.shape}')
    dataset.X = X_subset
    dataset.y = y_subset
    return dataset
