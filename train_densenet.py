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

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    # export_cifar('/backups/datasets/cifar-10-python.tar.gz', '/backups/work/CIFAR10')
    cifar = CIFAR('/backups/work/CIFAR10', shuffle=True, normalize=True, augment=True)

    densenet = DenseNet(num_classes=10, growth_rate=12, depth=100, bc_mode=True,
                        total_blocks=3, keep_prob=0.8, reduction=0.5,
                        weight_decay=1e-4, nesterov_momentum=0.9)

    def train_input_fn(epochs, learning_rate):
        dataset = cifar.train_set
        dataset = dataset.repeat(epochs)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'image': features, 'learning_rate': learning_rate}, labels

    def eval_input_fn():
        dataset = cifar.train_set
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'image': features}, labels

    # Create the Estimator
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig().replace(session_config=sess_config)

    classifier = tf.estimator.Estimator(
        model_fn=densenet.cifar_model_fn, model_dir="/tmp/cifar_model", config=config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"accuracy": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    # classifier.train(
    #     input_fn=lambda: train_input_fn(epochs=150, learning_rate=0.1),
    #     hooks=[logging_hook]
    # ).train(
    #     input_fn=lambda: train_input_fn(epochs=75, learning_rate=0.01),
    #     hooks=[logging_hook]
    # ).train(
    #     input_fn=lambda: train_input_fn(epochs=75, learning_rate=0.001),
    #     hooks=[logging_hook]
    # )

    # Evaluate the model and print results
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
