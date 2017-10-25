#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Alvin Zhu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train DenseNet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg

from datasets.cifar import CIFAR, export_cifar
from models.dense_net import DenseNet

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    # export_cifar('/backups/datasets/cifar-10-python.tar.gz', '/backups/work/CIFAR10')
    cifar = CIFAR('/backups/work/CIFAR10', shuffle=True, normalize=True, augment=True, one_hot=False, batch_size=100)

    densenet = DenseNet(num_classes=10, growth_rate=12, depth=100, bc_mode=True,
                        total_blocks=3, dropout_rate=0.2, reduction=0.5,
                        weight_decay=1e-4, nesterov_momentum=0.9)

    def train_input_fn(epochs, learning_rate):
        dataset = cifar.train_set
        dataset = dataset.repeat(epochs)
        # dataset = dataset.skip(16)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'images': features, 'learning_rate': learning_rate}, labels

    def eval_input_fn():
        dataset = cifar.test_set
        dataset = dataset.repeat(1)
        # dataset = dataset.skip(16)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'images': features}, labels

    # Create the Estimator
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig().replace(session_config=sess_config)

    classifier = tf.estimator.Estimator(
        model_fn=densenet.cifar_model_fn, model_dir="/backups/work/logs/cifar_model1", config=config)

    # Set up logging
    tensors_to_log = {"accuracy": "Accuracy/train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # debug_hook = tfdbg.LocalCLIDebugHook()
    # debug_hook.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)

    # # Train the model
    for i in range(20):
        # Train the model
        classifier.train(
            input_fn=lambda: train_input_fn(epochs=10, learning_rate=0.1),
            steps=5000, hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn, steps=100)
        print(eval_results)

    for i in range(10):
        # Train the model
        classifier.train(
            input_fn=lambda: train_input_fn(epochs=10, learning_rate=0.01),
            steps=5000, hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn, steps=100)
        print(eval_results)

    for i in range(10):
        # Train the model
        classifier.train(
            input_fn=lambda: train_input_fn(epochs=10, learning_rate=0.001),
            steps=5000, hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn, steps=100)
        print(eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
