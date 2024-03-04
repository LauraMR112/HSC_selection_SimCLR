#!/usr/bin/env python
# coding: utf-8

# # Semi-supervised image classification using contrastive pretraining with SimCLR
# 
# **Author:** [András Béres](https://www.linkedin.com/in/andras-beres-789190210)<br>
# **Date created:** 2021/04/24<br>

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os, glob
from itertools import compress
from tqdm import tqdm
import time
import pandas as pd


# ## Hyperparameter setup

# Dataset hyperparameters
unlabeled_dataset_size = 124952 #100000
#labeled_dataset_size = 5000
image_channels = 5

# Algorithm hyperparameters
num_epochs = 20
batch_size = 500 # Corresponds to 250 steps per epoch
width = 128
temperature = 0.1
# Stronger augmentations for contrastive, weaker ones for supervised training
#contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2, 'stddev': 0.05}
contrastive_augmentation = {"rot_factor": 0.50}
#classification_augmentation = {"min_area": 0.75,"brightness": 0.3,"jitter": 0.1, 'stddev': 0.02}
classification_augmentation = {"rot_factor": 0.25}


def prepare_dataset(train_dataset):
    steps_per_epoch = unlabeled_dataset_size // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
    print("batch size is {unlabeled_batch_size} (unlabeled)".format(unlabeled_batch_size))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    unlabeled_train_dataset = (train_dataset.shuffle(buffer_size=10 * unlabeled_batch_size)
                               .batch(unlabeled_batch_size))

    train_dataset = unlabeled_train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset

#train_dataset = prepare_dataset(train_set)

# ## Image augmentations
# Distorts the color distibutions of images
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform((batch_size, 1, 1, 1),
                minval=-self.brightness,maxval=self.brightness,)
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 5, 5), minval=-self.jitter, maxval=self.jitter)

            color_transforms = (tf.eye(5, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices)
            # Matrix multiplication in tensorflow works as (a,b,c,d)x(e,f,d,h) --> (a,b,c,h)
            # images.shape(6,84,84,5) and transformations.shape(6,1,5,5)
            # result.shape(6,84,84,5)
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images

class Gaussian_noise(layers.Layer):
    def __init__(self, stddev = 0, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config
    def call(self, images, training=True):
        if training:
            funct = layers.GaussianNoise(self.stddev, seed=None)
            images = funct(images, training = True)
        return images

# Image augmentation module
def get_augmenter(rot_factor): 
 
    return keras.Sequential(
        [
            layers.Input(shape=(60,60,5)),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomFlip("vertical"),
            layers.RandomRotation(rot_factor, fill_mode='nearest', seed=None, fill_value=0.0),
            ## rot_factor = 0.5 means 50% * 2pi --> random value in range [-pi, pi]
            ## Gaussian_noise(stddev)
            #RandomColorAffine(brightness, jitter),
        ] 
    )


# Define the encoder architecture
def get_encoder():
    return keras.Sequential(
        [
            layers.Conv2D(width, input_shape=(60,60,5), kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu") 
        ],
        name="encoder",
    )

@keras.saving.register_keras_serializable('Contrastive_model_package')
class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)
        self.encoder = get_encoder()
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential([
                layers.Dense(width, input_shape=(width,)), 
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
                layers.Dense(10),], )

        self.encoder.summary()
        self.projection_head.summary()
        
    def compile(self, contrastive_optimizer,  **kwargs): 
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
        ]
    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )
        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )
        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2
        
    def train_step(self, data):
        unlabeled_images = data
        images = unlabeled_images
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )

        self.contrastive_loss_tracker.update_state(contrastive_loss)
        
        return {m.name: m.result() for m in self.metrics}

assert keras.saving.get_registered_object('Contrastive_model_package>ContrastiveModel') == ContrastiveModel
assert keras.saving.get_registered_name(ContrastiveModel) == 'Contrastive_model_package>ContrastiveModel'

# Trained and labeled set
print('Loading tensor with g,r,i,z images of the candidate sources.................')
train_set = np.load('hsc_tensors/train_set.npy')

# Contrastive pretraining
print('Compiling and training the model.........................')
pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
)

pretraining_history = pretraining_model.fit(
    train_set, epochs=num_epochs) 


print('Getting and saving representation in latent space for the sources.....')
encoder_preds = pretraining_model.encoder.predict(train_set, batch_size=batch_size)
np.save('encoder_predict_train.npy', encoder_preds)

print('Saving the model with name: CL_model.keras in the current path')
pretraining_model.save('CL_model.keras')


