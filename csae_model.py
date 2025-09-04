"""
This file is sourced from JohnNellas CSAE and is modified to retain the
Convolutional Supervised Autoencoder (CSAE). 

Modifications:
removed 'build_test_model', 'decode_grid_of_latent_representations',
'decision_boundary_on_latent_space'

Source: 
https://github.com/JohnNellas/CSAE

Reference: 
"Nellas, I.A., Tasoulis, S.K., Plagianakos, V.P. and Georgakopoulos, S.V., 
2022. Supervised Dimensionality Reduction and Image Classification 
Utilizing Convolutional Autoencoders. arXiv preprint arXiv:2208.12152."
"""

import os
import typing
import numpy as np 
from numpy import save
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import tensorflow as tf

# ensure eager execution
if tf.executing_eagerly():
    print("Eager execution is enabled.")
else:
    print("Eager execution is disabled.")
    tf.compat.v1.enable_eager_execution() # if needed, enable it.

#debugging tool
import ipdb


class CSAE(tf.keras.Model):
    def __init__(self, autoencoder, class_model,ae_metric, class_metric, **kwargs):
        """
        Args:
        autoencoder: The autoencoder model
        class_model: The classifier model
        ae_metric: The autoencoder loss metric
        class_metric: The classifier loss metric
        """
        super(CSAE,self).__init__(**kwargs)
        self.autoencoder = autoencoder
        self.class_model = class_model
        self.ae_metric = ae_metric
        self.class_metric = class_metric
        self.acc_metric = tf.keras.metrics.Accuracy()

    def compile(self,
                class_optimizer,
                ae_optimizer,
                class_loss_fn,
                ae_fn):
        super(CSAE, self).compile()
        """
            class_optimizer: The optimizer for the classifier Network
            ae_optimizer: The optimizer for the Convolutional Autoencoder network
            class_loss_fn: The Loss Functions of the Classifier Network
            ae_fn: The Loss Functions of the Convolutional Autoencoder Network
            class_metric: The Metric for the classifier Network
            ae_metric: The Metric for the Convolutional Autoencoder network
        """
        self.class_optimizer = class_optimizer
        self.ae_optimizer = ae_optimizer
        self.class_loss_fn = class_loss_fn
        self.ae_loss_fn = ae_fn

    @property
    def metrics(self):
        return [self.ae_metric, self.class_metric, self.acc_metric]
    
    def train_step(self, data):
        x_data, y_data = data

        # AUTOENCODER PART
        with tf.GradientTape() as tape1:
            reconstructions = self.autoencoder(x_data, training=True)
            loss_value = self.ae_loss_fn(x_data, reconstructions)

        grads = tape1.gradient(loss_value, self.autoencoder.trainable_weights)
        self.ae_optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

        # CLASSIFIER PART
        with tf.GradientTape() as tape2:
            predictions = self.class_model(x_data, training=True)
            loss_value_class = self.class_loss_fn(y_data, predictions)

        grads_class = tape2.gradient(loss_value_class, self.class_model.trainable_weights)
        # ipdb.set_trace()
        self.class_optimizer.apply_gradients(zip(grads_class,
                                                self.class_model.trainable_weights))

        # update states
        self.ae_metric.update_state(x_data, reconstructions)
        self.class_metric.update_state(y_data, predictions)
        self.acc_metric.update_state(y_data, tf.math.argmax(predictions,axis=1))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_data, y_data = data
        
        # compute the reconstructions and the predictions
        reconstructions = self.autoencoder(x_data, training=False)
        predictions = self.class_model(x_data, training=False)

        # update states
        self.ae_metric.update_state(x_data, reconstructions)
        self.class_metric.update_state(y_data, predictions)
        self.acc_metric.update_state(y_data, tf.math.argmax(predictions,axis=1))


        return {m.name: m.result() for m in self.metrics}

    def call(self, data):
        return self.autoencoder(data), self.class_model(data)
