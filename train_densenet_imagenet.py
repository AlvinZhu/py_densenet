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

from datasets.imagenet import ImageNet
from datasets.cifar import CIFAR
from models.dense_net import DenseNet

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    # export_cifar('/backups/datasets/cifar-10-python.tar.gz', '/backups/work/CIFAR10')
    imagenet = ImageNet('/backups/work/ILSVRC2017/ILSVRC',
                        shuffle=True, normalize=True, augment=False, one_hot=False, batch_size=128)
    # imagenet = CIFAR('/backups/work/CIFAR10',
    #                  shuffle=True, normalize=True, augment=True, one_hot=False, batch_size=32)

    densenet = DenseNet(num_classes=imagenet.num_classes, growth_rate=12, bc_mode=True, block_config=(6, 12, 24, 16),
                        dropout_rate=0.2, reduction=0.5, weight_decay=1e-4, nesterov_momentum=0.9)

    def train_input_fn(learning_rate):
        dataset = imagenet.train_set
        dataset = dataset.repeat(1)
        dataset = dataset.skip(imagenet.train_set_size % imagenet.batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'images': features, 'learning_rate': learning_rate}, labels

    def eval_input_fn():
        dataset = imagenet.val_set
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'images': features}, labels

    # Create the Estimator
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig().replace(session_config=sess_config)

    classifier = tf.estimator.Estimator(
        model_fn=densenet.imagenet_model_fn,
        model_dir="/backups/work/logs/imagenet_model1",
        # params={'image_shape': imagenet.image_shape},
        config=config)

    # Set up logging
    # tensors_to_log = {"accuracy": "Accuracy/train_accuracy"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=100)

    # debug_hook = tfdbg.LocalCLIDebugHook()
    # debug_hook.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)

    # # Train the model
    for i in range(30):
        # Train the model
        classifier.train(input_fn=lambda: train_input_fn(learning_rate=0.1))

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    for i in range(30):
        # Train the model
        classifier.train(input_fn=lambda: train_input_fn(learning_rate=0.01))

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    for i in range(30):
        # Train the model
        classifier.train(input_fn=lambda: train_input_fn(learning_rate=0.001))

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    for i in range(10):
        # Train the model
        classifier.train(input_fn=lambda: train_input_fn(learning_rate=0.0001))

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
