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
    def __init__(self, num_classes, growth_rate, depth, bc_mode,
                 total_blocks, dropout_rate, reduction,
                 weight_decay, nesterov_momentum):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            dropout_rate: `float`, If dropout_rate = 0, dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.depth = depth
        self.bc_mode = bc_mode
        self.first_output_features = growth_rate * 2 if bc_mode else 16
        self.total_blocks = total_blocks
        self.reduction = reduction

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

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

    @staticmethod
    def conv2d(input_, out_features, kernel_size,
               strides=(1, 1, 1, 1), padding='SAME'):
        in_features = int(input_.get_shape()[-1])
        kernel = tf.get_variable(
            name='kernel',
            shape=[kernel_size, kernel_size, in_features, out_features],
            initializer=tf.variance_scaling_initializer())
        output = tf.nn.conv2d(input_, kernel, strides, padding)
        return output

    @staticmethod
    def avg_pool(input_, k):
        kernel_size = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(input_, kernel_size, strides, padding)
        return output

    @staticmethod
    def batch_norm(input_, training):
        output = tf.contrib.layers.batch_norm(
            input_, scale=True, is_training=training,
            updates_collections=None)
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
        if not self.bc_mode:
            comp_out = self.composite_function(
                input_, out_features=growth_rate, kernel_size=3, training=training)
        else:
            bottleneck_out = self.bottleneck(input_, out_features=growth_rate, training=training)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3, training=training)
        # concatenate input_ with out from composite function
        output = tf.concat(axis=3, values=(input_, comp_out))
        return output

    def add_dense_block(self, input_, growth_rate, layers_per_block, training):
        """Add N H_l internal layers"""
        output = input_
        for layer in range(layers_per_block):
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
        output = self.avg_pool(output, k=2)
        return output

    def classification_layer(self, input_, training):
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
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.num_classes], name='W')
        bias = self.bias_variable([self.num_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def cifar_model_fn(self, features, labels, mode):
        layers_per_block = (self.depth - 4) // 3
        if self.bc_mode:
            layers_per_block = layers_per_block // 2
        growth_rate = self.growth_rate
        training = tf.constant(mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope("Image_Processing"):
            images = features["images"]

            if mode == tf.estimator.ModeKeys.PREDICT:
                mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
                std = tf.sqrt(variance)
                images = tf.cast(images, tf.float32) / 255.0
                images = (images - mean) / std

            images = tf.image.resize_image_with_crop_or_pad(images, 32, 32)
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        with tf.variable_scope("DenseNet"):
            for block in range(self.total_blocks):
                with tf.variable_scope("Dense_Block_%d" % block):
                    output = self.add_dense_block(output, growth_rate, layers_per_block, training)
                # last block exist without transition layer
                if block != self.total_blocks - 1:
                    with tf.variable_scope("Transition_Layer_%d" % block):
                        output = self.transition_layer(output, training)

            with tf.variable_scope("Classification_Layer"):
                logits = self.classification_layer(output, training)
        with tf.variable_scope("Predictions"):
            probabilities = tf.nn.softmax(logits)
            classes = tf.argmax(input=probabilities, axis=1)

        predictions = {
            "classes": classes,
            "probabilities": probabilities
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Losses
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

        # optimizer and train step
        with tf.variable_scope("Train_OP"):
            optimizer = tf.train.MomentumOptimizer(
                features['learning_rate'], self.nesterov_momentum, use_nesterov=True)
            train_op = optimizer.minimize(
                loss + l2_loss * self.weight_decay,
                global_step=tf.train.get_global_step())

        with tf.variable_scope("Accuracy"):
            correct_prediction = tf.equal(
                classes,
                labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='train_accuracy')

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar("loss_per_batch", loss)
            tf.summary.scalar("accuracy_per_batch", accuracy)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)
