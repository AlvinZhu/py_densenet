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

"""Train AlexNet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

import numpy as np
import tensorflow as tf

from datasets.cifar import CIFAR, export_cifar
from models.alex_net import AlexNet

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    # export_cifar('/backups/datasets/cifar-10-python.tar.gz', '/backups/work/CIFAR10')
    cifar = CIFAR('/backups/work/CIFAR10', shuffle=True, normalize=True, augment=True, one_hot=False)

    alexnet = AlexNet(num_classes=10)

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
        model_fn=alexnet.cifar_model_fn, model_dir="/tmp/logs/cifar_model_alexnet", config=config)

    # Train the model
    classifier.train(
        input_fn=lambda: train_input_fn(epochs=150, learning_rate=0.1)
    )

    # Evaluate the model and print results
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
