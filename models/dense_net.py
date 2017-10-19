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


class DenseNet(object):
    def __init__(self, num_classes, growth_rate, depth, bc_mode,
                 total_blocks, keep_prob, reduction,
                 weight_decay, nesterov_momentum):
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.depth = depth
        self.bc_mode = bc_mode
        self.first_output_features = growth_rate * 2 if bc_mode else 16
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        if self.bc_mode:
            self.layers_per_block = self.layers_per_block // 2
        self.reduction = reduction
        self.keep_prob = keep_prob

        self.nesterov_momentum = nesterov_momentum
        self.weight_decay = weight_decay

        print("Build DenseNet model with {} blocks, {} bottleneck layers and {} composite layers each.".format(
            self.total_blocks, self.layers_per_block,
            self.layers_per_block))
        print("Reduction at transition layers: {:.1f}".format(self.reduction))

    def cifar_model_fn(self, features, labels, mode):
        layers_per_block = (self.depth - 4) // 3
        if self.bc_mode:
            layers_per_block = layers_per_block // 2

        training = tf.constant(mode == tf.estimator.ModeKeys.TRAIN)

        input_ = features["image"]

        if mode == tf.estimator.ModeKeys.PREDICT:
            mean, variance = tf.nn.moments(input_, axes=[0, 1, 2])
            std = tf.sqrt(variance)
            input_ = tf.cast(input_, tf.float32) / 255.0
            input_ = (input_ - mean) / std

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                input_,
                out_features=self.first_output_features,
                kernel_size=3)

        for block in range(self.total_blocks):
            with tf.variable_scope("Block_{}".format(block)):
                output = self.add_dense_block(output, self.growth_rate, layers_per_block, training)
            with tf.variable_scope("Transition_after_block_{}".format(block)):
                output = self.transition_layer(output, block == self.total_blocks - 1, training)

        with tf.variable_scope("FC"):
            features_total = int(output.get_shape()[-1])
            output = tf.reshape(output, [-1, features_total])
            weights = self.weight_variable_xavier(
                [features_total, self.num_classes], name='W')
            bias = self.bias_variable([self.num_classes])
            logits = tf.matmul(output, weights) + bias

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        # Losses
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=onehot_labels))

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # self.cross_entropy = cross_entropy
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

            # optimizer and train step
            optimizer = tf.train.MomentumOptimizer(
                features["learning_rate"], self.nesterov_momentum, use_nesterov=True)
            train_op = optimizer.minimize(
                loss=loss + l2_loss * self.weight_decay,
                global_step=tf.train.get_global_step())

            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions["classes"], labels), tf.float32), name='train_accuracy')
            tf.summary.scalar("loss_train", loss)
            tf.summary.scalar("accuracy_train", accuracy)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

    @staticmethod
    def weight_variable_msra(shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.variance_scaling_initializer())

    @staticmethod
    def weight_variable_xavier(shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    @staticmethod
    def bias_variable(shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def conv2d(self, input_, out_features, kernel_size,
               strides=(1, 1, 1, 1), padding='SAME'):
        in_features = int(input_.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(input_, kernel, strides, padding)
        return output

    @staticmethod
    def avg_pool(input_, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(input_, ksize, strides, padding)
        return output

    @staticmethod
    def batch_norm(input_, training):
        output = tf.contrib.layers.batch_norm(
            input_, scale=True, is_training=training,
            updates_collections=None)
        return output

    def dropout(self, input_, training):
        if self.keep_prob < 1:
            output = tf.cond(
                training,
                lambda: tf.nn.dropout(input_, self.keep_prob),
                lambda: input_
            )
        else:
            output = input_
        return output

    def composite_function(self, input_, out_features, training, kernel_size=3):
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
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output, training)
        return output

    def add_layer(self, input_, growth_rate, training):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if self.bc_mode:
            output = self.bottleneck(input_, growth_rate, training)
        else:
            output = input_
        comp_out = self.composite_function(
            output, growth_rate, training, kernel_size=3)
        # concatenate _input with out from composite function
        output = tf.concat(axis=3, values=(input_, comp_out))

        return output

    def add_dense_block(self, input_, growth_rate, layers_per_block, training):
        """Add N H_l internal layers"""
        output = input_
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_layer(input_, growth_rate, training)
        return output

    def transition_layer(self, input_, last, training):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # BN
        output = self.batch_norm(input_, training)
        # ReLU
        output = tf.nn.relu(output)
        if last:
            last_pool_kernel = int(output.get_shape()[-2])
            output = self.avg_pool(output, k=last_pool_kernel)
        else:
            if self.bc_mode:
                out_features = int(int(input_.get_shape()[-1]) * self.reduction)
            else:
                out_features = int(input_.get_shape()[-1])
            # convolution 1x1
            output = self.conv2d(
                output, out_features=out_features, kernel_size=1)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output, training)
            # run average pooling
            output = self.avg_pool(output, k=2)
        return output
