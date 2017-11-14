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

"""MNIST dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import cv2

from tensorflow.examples.tutorials.mnist import input_data


def export_images(images, labels, dataset_path, subset='train'):
    """
    Export images form dict
    """
    for i in range(images.shape[0]):
        image_path = os.path.join(dataset_path, subset, str(labels[i]), '{}.png'.format(i))
        image = np.reshape(images[i], (28, 28)) * 255
        cv2.imwrite(image_path, image)


def export_mnist(src_path, dataset_path):
    """
    Export dataset.
    :param src_path: ubyte.gz files folder.
    :param dataset_path: output path of images.
    """
    print('Reading ubyte.gz files(Maybe downloading) ...')
    mnist = input_data.read_data_sets(src_path, one_hot=False)

    print('Exporting images ...')
    for i in range(10):
        train_path = os.path.join(dataset_path, 'train', str(i))
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        test_path = os.path.join(dataset_path, 'test', str(i))
        if not os.path.exists(test_path):
            os.makedirs(test_path)

    train_images = np.vstack((mnist.train.images, mnist.validation.images))
    train_labels = np.hstack((mnist.train.labels, mnist.validation.labels))

    export_images(train_images, train_labels, dataset_path, 'train')

    test_images = mnist.test.images
    test_labels = mnist.test.labels

    export_images(test_images, test_labels, dataset_path, 'test')


class MNIST(object):
    def __init__(self, dataset_path, num_threads=8, batch_size=100,
                 shuffle=True, normalize=True, augment=True, one_hot=True):
        """
        :param dataset_path: The dataset folder path.
        :param num_threads: number of threads.
        :param batch_size: batch size.
        :param shuffle: shuffle or not.
        :param normalize: normalize or not.
        :param augment: augment or not.
        """
        self.num_classes = 10
        self.dataset_path = dataset_path
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.augment = augment
        self.normalize = normalize
        self.shuffle = shuffle
        self.one_hot = one_hot

        self._load()
        self._measure_mean_and_std()
        self._pre_process()

    def _load(self):
        """Load dataset from folder which contains images."""
        all_files = {'train': [], 'test': []}
        labels = {'train': [], 'test': []}
        for subset in all_files.keys():
            for i in range(self.num_classes):
                folder = os.path.join(self.dataset_path, subset, str(i))
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
                labels[subset] += [i] * len(files)
                all_files[subset] += files

        train_set_size = len(all_files['train'])
        test_set_size = len(all_files['test'])

        shuffle_index = np.arange(train_set_size)
        np.random.shuffle(shuffle_index)

        train_files = tf.constant(np.array(all_files['train'])[shuffle_index])
        train_labels = tf.constant(np.array(labels['train'])[shuffle_index])

        shuffle_index = np.arange(test_set_size)
        np.random.shuffle(shuffle_index)

        test_files = tf.constant(np.array(all_files['test'])[shuffle_index])
        test_labels = tf.constant(np.array(labels['test'])[shuffle_index])

        self.train_set = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
        self.train_set_size = train_set_size

        self.test_set = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
        self.test_set_size = test_set_size

    def _measure_mean_and_std(self):
        """
        measure mean and std of random samples from train set. number of samples is min(50000, train_set_size).
        """
        num_samples = min(100000, self.train_set_size)
        dataset = self.train_set.shuffle(buffer_size=num_samples)
        dataset = dataset.map(
            self._read_image_func,
            num_parallel_calls=self.num_threads)
        dataset = dataset.batch(num_samples)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
        std = tf.sqrt(variance)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            out_mean, out_std = sess.run([mean, std])
        self.mean = tf.constant(out_mean)
        self.std = tf.constant(out_std)

    def _read_image_func(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=1)
        image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
        return image_float, label

    def _normalize_func(self, image, label):
        image_float = (image - self.mean) / self.std
        # image_float = tf.image.per_image_standardization(image)
        return image_float, label

    def _augment_func(self, image, label):
        image = tf.image.resize_image_with_crop_or_pad(image, 36, 36)
        image = tf.random_crop(image, (28, 28, 1))
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image)
        return image, label

    def _one_hot_func(self, image, label):
        onehot_label = tf.one_hot(indices=tf.cast(label, tf.int32), depth=self.num_classes)
        return image, onehot_label

    def _pre_process(self):
        # self.train_set_size = self.train_set_size - self.train_set_size % self.batch_size
        if self.shuffle:
            self.train_set = self.train_set.shuffle(buffer_size=self.train_set_size)
        else:
            self.train_set = self.train_set.take(self.train_set_size)

        def _train_pre_process_fun(filename, label):
            image, label = self._read_image_func(filename, label)
            if self.normalize:
                image, label = self._normalize_func(image, label)
            if self.augment:
                image, label = self._augment_func(image, label)
            if self.one_hot:
                image, label = self._one_hot_func(image, label)
            return image, label

        def _test_pre_process_fun(filename, label):
            image, label = self._read_image_func(filename, label)
            if self.normalize:
                image, label = self._normalize_func(image, label)
            if self.one_hot:
                image, label = self._one_hot_func(image, label)
            return image, label

        self.train_set = self.train_set.map(
            _train_pre_process_fun,
            num_parallel_calls=self.num_threads)

        self.test_set = self.test_set.map(
            _test_pre_process_fun,
            num_parallel_calls=self.num_threads)

        self.train_set = self.train_set.batch(self.batch_size)
        self.test_set = self.test_set.batch(self.batch_size)


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    # export_mnist('/home/alvin/Work/MNIST_data', '/home/alvin/Work/MNIST')
    mnist = MNIST('/home/alvin/Work/MNIST', shuffle=True, normalize=True, augment=True, one_hot=False)
    dataset = mnist.test_set
    dataset = dataset.batch(10000)
    iterator = dataset.make_one_shot_iterator()
    features, label = iterator.get_next()
    with tf.Session() as sess:
        images_path, labels = sess.run([features, label])
        for image_path, label in zip(images_path, labels):
            if not os.path.dirname(image_path).endswith(str(label)):
                print(image_path)
                print(label)


if __name__ == '__main__':
    main()

