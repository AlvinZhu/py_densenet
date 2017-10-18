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

import numpy as np
import tensorflow as tf

from datasets.cifar import CIFAR, export_cifar
from models.dense_net import DenseNet

# tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    # export_cifar('/backups/datasets/cifar-10-python.tar.gz', '/backups/work/CIFAR10')
    cifar = CIFAR('/backups/work/CIFAR10', shuffle=True, normalize=True, augment=True)

    densenet = DenseNet((32, 32, 3), 10, 12, 40, 3, 0.8, 1e-4, 0.9, 0.5)

    def train_input_fn():
        dataset = cifar.train_set
        dataset = dataset.batch(64)
        dataset = dataset.repeat(10)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'image': features}, labels

    def eval_input_fn():
        dataset = cifar.train_set
        dataset = dataset.batch(64)
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'image': features}, labels

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=densenet.model_fn, model_dir="/tmp/cifar_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # classifier.train(input_fn=train_input_fn)

    for i in range(300):
        # Train the model
        classifier.train(
            input_fn=train_input_fn,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
