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

"""DenseNet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DenseNet:
    def __init__(self, num_classes, growth_rate, bc_mode, block_config, reduction,
                 dropout_rate, weight_decay, nesterov_momentum):
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.bc_mode = bc_mode
        self.first_output_features = growth_rate * 2 if bc_mode else 16
        self.block_config = block_config
        self.reduction = reduction

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

    @staticmethod
    def conv2d(input_, out_features, kernel_size,
               stride=1, padding='SAME'):
        in_features = int(input_.get_shape()[-1])
        kernel = tf.get_variable(
            name='kernel',
            shape=[kernel_size, kernel_size, in_features, out_features],
            initializer=tf.variance_scaling_initializer())
        strides = (1, stride, stride, 1)
        output = tf.nn.conv2d(input_, kernel, strides, padding)
        return output

    @staticmethod
    def avg_pool(input_, ksize):
        kernel_size = (1, ksize, ksize, 1)
        strides = (1, ksize, ksize, 1)
        padding = 'VALID'
        output = tf.nn.avg_pool(input_, kernel_size, strides, padding)
        return output

    @staticmethod
    def max_pool(input_, ksize, stride):
        kernel_size = (1, ksize, ksize, 1)
        strides = (1, stride, stride, 1)
        padding = 'VALID'
        output = tf.nn.max_pool(input_, kernel_size, strides, padding)
        return output

    @staticmethod
    def batch_norm(input_, training):
        output = tf.layers.batch_normalization(
            input_, center=True, scale=True, training=training, fused=True)
        return output

    def dropout(self, input_, training):
        output = tf.layers.dropout(input_, self.dropout_rate, training)
        return output

    def composite_function(self, input_, out_features, kernel_size, training):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(input_, training)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output, training)
        return output

    def bottleneck(self, input_, out_features, training):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(input_, training)
            output = tf.nn.relu(output)
            output = self.conv2d(
                output, out_features=out_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output, training)
        return output

    def add_layer(self, input_, out_features, training):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if self.bc_mode:
            output = self.bottleneck(input_, out_features * 4, training)
        else:
            output = input_
        output = self.composite_function(
            output, out_features, kernel_size=3, training=training)
        # concatenate _input with out from composite function
        output = tf.concat(axis=3, values=(input_, output))

        return output

    def add_dense_block(self, input_, growth_rate, num_layers, training):
        """Add N H_l internal layers"""
        output = input_
        for layer in range(num_layers):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_layer(output, growth_rate, training)
        return output

    def transition_layer(self, input_, training):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(input_.get_shape()[-1])
        if self.bc_mode:
            out_features = int(out_features * self.reduction)
        output = self.composite_function(
            input_, out_features=out_features, kernel_size=1, training=training)
        # run average pooling
        output = self.avg_pool(output, ksize=2)
        return output

    @staticmethod
    def fully_connected(input_, out_dim):
        with tf.name_scope('fully_connected'):
            output = tf.layers.dense(
                input_, out_dim,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        return output

    def classification_layer(self, input_, num_classes, training):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(input_, training)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, ksize=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])

        logits = self.fully_connected(output, num_classes)
        return logits

    @staticmethod
    def pre_process(images):
        with tf.variable_scope("Image_Processing"):
                mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
                std = tf.sqrt(variance)
                images = tf.cast(images, tf.float32) / 255.0
                images = (images - mean) / std
        return images

    def forward_pass(self, input_, training):
        output = input_
        with tf.variable_scope("DenseNet"):
            for i in range(len(self.block_config)):
                with tf.variable_scope("Dense_Block_%d" % i):
                    output = self.add_dense_block(output, self.growth_rate, self.block_config[i], training)
                if i != len(self.block_config) - 1:
                    with tf.variable_scope("Transition_Layer_%d" % i):
                        output = self.transition_layer(output, training)
            with tf.variable_scope("Classification_Layer"):
                logits = self.classification_layer(output, self.num_classes, training)
        return logits

    def model_fn(self, logits, labels, mode, hyper_params):
        with tf.variable_scope("Predictions"):
            probabilities = tf.nn.softmax(logits)
            classes = tf.argmax(input=logits, axis=1)

        predictions = {
            "classes": classes,
            "probabilities": probabilities
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        with tf.variable_scope("Loss"):
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.num_classes)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=onehot_labels))

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        with tf.variable_scope("L2_Loss"):
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        with tf.variable_scope("Train_OP"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.MomentumOptimizer(
                    hyper_params['learning_rate'], self.nesterov_momentum, use_nesterov=True)
                train_op = optimizer.minimize(
                    loss + l2_loss * self.weight_decay,
                    global_step=tf.train.get_global_step())

        # with tf.variable_scope("Accuracy"):
        #     correct_prediction = tf.equal(
        #         classes,
        #         labels)
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='train_accuracy')

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar("loss_per_batch", loss)
            # tf.summary.scalar("accuracy_per_batch", accuracy)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

    def cifar_model_fn(self, features, labels, mode):
        training = tf.constant(mode == tf.estimator.ModeKeys.TRAIN)

        images = features["images"]
        if mode == tf.estimator.ModeKeys.PREDICT:
            images = self.pre_process(images)

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                images,
                out_features=self.first_output_features,
                kernel_size=3)

        logits = self.forward_pass(output, training)

        return self.model_fn(logits, labels, mode, features)

    def imagenet_model_fn(self, features, labels, mode):
        training = tf.constant(mode == tf.estimator.ModeKeys.TRAIN)

        images = features["images"]
        if mode == tf.estimator.ModeKeys.PREDICT:
            images = self.pre_process(images)

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                images,
                out_features=self.first_output_features,
                kernel_size=7,
                stride=2
            )
            output = self.batch_norm(output, training)
            output = tf.nn.relu(output)
            output = self.max_pool(output, ksize=3, stride=2)

        logits = self.forward_pass(output, training)

        return self.model_fn(logits, labels, mode, features)
