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

"""ImageNet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import xml.etree.ElementTree as Et
import json

import numpy as np
import tensorflow as tf


def index_all(dataset_path):
    """
    Index dataset.
    """
    train_image_dir = 'Data/CLS-LOC/train'
    val_image_dir = 'Data/CLS-LOC/val'
    val_label_dir = 'Annotations/CLS-LOC/val'

    # index train set
    file_list = os.listdir(os.path.join(dataset_path, train_image_dir))
    label_names = [filename for filename in file_list if filename.startswith('n')]
    assert len(file_list) == 1000

    label_names.sort()

    train_images = []
    train_labels = []
    for i in range(len(label_names)):
        file_list = os.listdir(os.path.join(dataset_path, train_image_dir, label_names[i]))
        file_list = [os.path.join(train_image_dir, label_names[i], filename) for filename in file_list if filename.endswith('JPEG')]
        train_labels += [i] * len(file_list)
        train_images += file_list

    # index val set
    label_names_dict = {}
    for i in range(len(label_names)):
        label_names_dict[label_names[i]] = i

    file_list = os.listdir(os.path.join(dataset_path, val_image_dir))
    file_list = [filename for filename in file_list if filename.endswith('JPEG')]
    assert len(file_list) == 50000

    val_images = []
    val_labels = []
    for filename in file_list:
        label_file = os.path.splitext(filename)[0] + '.xml'
        root = Et.parse(os.path.join(dataset_path, val_label_dir, label_file)).getroot()
        obj = root.find('object')
        label = label_names_dict[obj.find('name').text]
        val_images.append(os.path.join(val_image_dir, filename))
        val_labels.append(label)

    with open(os.path.join(dataset_path, 'meta.json'), 'w') as f:
        json.dump({'label_names': label_names}, f)

    with open(os.path.join(dataset_path, 'train.json'), 'w') as f:
        json.dump({'images': train_images, 'labels': train_labels}, f)

    with open(os.path.join(dataset_path, 'val.json'), 'w') as f:
        json.dump({'images': val_images, 'labels': val_labels}, f)


def measure_mean_and_std(imagenet):
    """
    measure mean and std of random samples from train set. number of samples is min(50000, train_set_size).
    :type imagenet: ImageNet
    """
    dataset = imagenet.train_set
    num_samples = min(50000, imagenet.train_set_size)
    batch_size = 1000

    def pre_process_fn(filename, label):
        image, label = imagenet.read_image_func(filename, label)
        shape = tf.shape(image)
        shorter_edge = tf.minimum(shape[0], shape[1])
        image = tf.image.resize_image_with_crop_or_pad(image, shorter_edge, shorter_edge)
        image = tf.image.resize_images(image, (imagenet.image_shape[0], imagenet.image_shape[1]))
        return image, label
    dataset = dataset.map(
        pre_process_fn,
        num_threads=imagenet.num_threads,
        output_buffer_size=2 * imagenet.batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
    std = tf.sqrt(variance)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        out_mean = np.zeros([imagenet.image_shape[2]])
        out_std = np.zeros([imagenet.image_shape[2]])

        for i in range(num_samples // batch_size):
            tmp_mean, tmp_std = sess.run([mean, std])
            out_mean += tmp_mean
            out_std += tmp_std

        out_mean /= num_samples // batch_size
        out_std /= num_samples // batch_size

        print('mean:{}, std:{}'.format(out_mean, out_std))

    with open(os.path.join(imagenet.dataset_path, 'meta.json')) as f:
        meta_dict = json.load(f)
    meta_dict['mean'] = out_mean.tolist()
    meta_dict['std'] = out_std.tolist()

    with open(os.path.join(imagenet.dataset_path, 'meta.json'), 'w') as f:
        json.dump(meta_dict, f)


class ImageNet(object):
    def __init__(self, dataset_path, image_shape=(224, 224, 3), num_threads=8, batch_size=64,
                 shuffle=True, normalize=True, augment=True, one_hot=False, read=True):
        """
        :param dataset_path: The dataset folder path.
        :param num_threads: number of threads.
        :param batch_size: batch size.
        :param shuffle: shuffle or not.
        :param normalize: normalize or not.
        :param augment: augment or not.
        """
        self.image_shape = image_shape
        self.read = read
        self.dataset_path = dataset_path
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.augment = augment
        self.normalize = normalize
        self.shuffle = shuffle
        self.one_hot = one_hot

        self._load()
        if self.read:
            self._pre_process()

    def _load(self):
        """Load dataset from folder which contains images."""
        with open(os.path.join(self.dataset_path, 'meta.json')) as f:
            meta_dict = json.load(f)
        self.label_names = meta_dict['label_names']
        self.num_classes = len(self.label_names)

        if 'mean' in meta_dict and 'std' in meta_dict:
            self.mean = tf.constant(meta_dict['mean'])
            self.std = tf.constant(meta_dict['std'])
        else:
            self.normalize = False
            self.mean = tf.constant(0.0)
            self.std = tf.constant(1.0)

        with open(os.path.join(self.dataset_path, 'train.json')) as f:
            train_set = json.load(f)
        train_set['images'] = [os.path.join(self.dataset_path, filename) for filename in train_set['images']]

        with open(os.path.join(self.dataset_path, 'val.json')) as f:
            val_set = json.load(f)
        val_set['images'] = [os.path.join(self.dataset_path, filename) for filename in val_set['images']]

        train_set_size = len(train_set['labels'])
        val_set_size = len(val_set['labels'])

        shuffle_index = np.arange(train_set_size)
        np.random.shuffle(shuffle_index)

        train_files = tf.constant(np.array(train_set['images'])[shuffle_index])
        train_labels = tf.constant(np.array(train_set['labels'])[shuffle_index])

        shuffle_index = np.arange(val_set_size)
        np.random.shuffle(shuffle_index)

        val_files = tf.constant(np.array(val_set['images'])[shuffle_index])
        val_labels = tf.constant(np.array(val_set['labels'])[shuffle_index])

        self.train_set = tf.contrib.data.Dataset.from_tensor_slices((train_files, train_labels))
        self.train_set_size = train_set_size

        self.val_set = tf.contrib.data.Dataset.from_tensor_slices((val_files, val_labels))
        self.val_set_size = val_set_size

    def _pre_process(self):
        # self.train_set_size = self.train_set_size - self.train_set_size % self.batch_size
        if self.shuffle:
            self.train_set = self.train_set.shuffle(buffer_size=self.train_set_size)
        else:
            self.train_set = self.train_set.take(self.train_set_size)

        def _train_pre_process_fun(filename, label):
            image, label = self.read_image_func(filename, label)
            if self.normalize:
                image, label = self.normalize_func(image, label)
            if self.augment:
                image, label = self.train_augment_func(image, label)
            else:
                image = tf.image.resize_images(image, (self.image_shape[0], self.image_shape[1]))
            image.set_shape(self.image_shape)
            if self.one_hot:
                image, label = self.one_hot_func(image, label)
            return image, label

        def _val_pre_process_fun(filename, label):
            image, label = self.read_image_func(filename, label)
            if self.normalize:
                image, label = self.normalize_func(image, label)
            if self.augment:
                image, label = self.val_augment_func(image, label)
            else:
                image = tf.image.resize_images(image, (self.image_shape[0], self.image_shape[1]))
            image.set_shape(self.image_shape)
            if self.one_hot:
                image, label = self.one_hot_func(image, label)
            return image, label

        self.train_set = self.train_set.map(
            _train_pre_process_fun,
            num_threads=self.num_threads,
            output_buffer_size=2 * self.batch_size)

        self.val_set = self.val_set.map(
            _val_pre_process_fun,
            num_threads=self.num_threads,
            output_buffer_size=2 * self.batch_size)

        self.train_set = self.train_set.batch(self.batch_size)
        self.val_set = self.val_set.batch(self.batch_size)

    def read_image_func(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string, channels=self.image_shape[2])
        image_decoded = tf.reverse(image_decoded, axis=[-1])
        image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
        return image_float, label

    def normalize_func(self, image, label):
        # image_float = (image - self.mean) / self.std
        image_float = tf.image.per_image_standardization(image)
        return image_float, label

    def train_augment_func(self, image, label):
        shape = tf.shape(image)
        scale = tf.random_uniform([], minval=self.image_shape[0], maxval=self.image_shape[0] * 3, dtype=tf.float32)
        shorter_edge = tf.minimum(shape[0], shape[1])
        crop_size = tf.to_int32(tf.to_float(shorter_edge) / scale * self.image_shape[0])
        image = tf.random_crop(image, (crop_size, crop_size, self.image_shape[2]))
        image = tf.image.resize_images(image, (self.image_shape[0], self.image_shape[1]))
        image = tf.image.random_flip_left_right(image)
        # image_float = tf.image.random_brightness(image_float, 0.4)
        # image_float = tf.image.random_hue(image_float, 0.4)
        # image_float = tf.image.random_flip_up_down(image_float)
        return image, label

    def val_augment_func(self, image, label):
        shape = tf.shape(image)
        shorter_edge = tf.minimum(shape[0], shape[1])
        image = tf.image.resize_image_with_crop_or_pad(image, shorter_edge, shorter_edge)
        image = tf.image.resize_images(image, (self.image_shape[0], self.image_shape[1]))
        return image, label

    def one_hot_func(self, image, label):
        onehot_label = tf.one_hot(indices=tf.cast(label, tf.int32), depth=self.num_classes)
        return image, onehot_label


def main():
    # index_all('/backups/work/ILSVRC2017/ILSVRC')
    imagenet = ImageNet('/backups/work/ILSVRC2017/ILSVRC')
    # measure_mean_and_std(imagenet)


if __name__ == '__main__':
    main()
