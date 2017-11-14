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
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

import tensorflow as tf

from datasets.mnist import MNIST
from models.cnn import cnn_model_fn

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load training and eval data
    mnist = MNIST('/backups/work/mnist', shuffle=True, normalize=True, augment=False, one_hot=False)

    def train_input_fn():
        dataset = mnist.train_set
        dataset = dataset.repeat(100)
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

    mnist_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/backups/work/logs/mnist_convnet_model", config=config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=60000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(mnist_estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
