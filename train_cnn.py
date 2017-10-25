#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from datasets.mnist import MNIST
from datasets.cifar import CIFAR
from models.cnn import cnn_model_fn

tf.logging.set_verbosity(tf.logging.INFO)


def main1(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/logs/mnist_convnet_model1_1")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for i in range(10):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=6000)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


def main2(unused_argv):
    # Load training and eval data
    mnist = MNIST('/backups/work/mnist', shuffle=True, normalize=True, augment=False, one_hot=False)

    def train_input_fn():
        dataset = mnist.train_set
        dataset = dataset.repeat(10)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'x': features}, labels

    def eval_input_fn():
        dataset = mnist.test_set
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'x': features}, labels

    # Create the Estimator
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig().replace(session_config=sess_config)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/logs/mnist_convnet_model_2_1", config=config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"accuracy": "train_accuracy"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=100)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    for i in range(10):
        # Train the model
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=6000)

        # Evaluate the model and print results
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


def main3(unused_argv):
    # Load training and eval data
    cifar = CIFAR('/backups/work/CIFAR10', shuffle=True, normalize=True, augment=True, one_hot=False, batch_size=100)

    def train_input_fn():
        dataset = cifar.train_set
        # dataset = dataset.skip(16)
        dataset = dataset.repeat(10)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'x': features}, labels

    def eval_input_fn():
        dataset = cifar.test_set
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'x': features}, labels

    # Create the Estimator
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig().replace(session_config=sess_config)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/logs/cifar_cnn", config=config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"accuracy": "train_accuracy"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=100)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    for i in range(30):
        # Train the model
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=5000)

        # Evaluate the model and print results
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run(main=main3)
