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

"""CIFAR dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import pickle
import tarfile
import json

import numpy as np
import tensorflow as tf
import cv2


def unpickle(file_path):
    # type: (str) -> dict
    """
    Unpickle a dict form file.
    :param file_path: file path.
    :return: dict.
    """
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo)
    return data_dict


def export_images(file_path, dataset_path, subset='train', cifar_10=True):
    # type: (str, str, str) -> None
    """
    Export images form dict
    :param file_path: path of file that contains labels, file names and data of images.
    :param dataset_path: output path of images.
    :param subset: train or test set
    """
    if cifar_10:
        label_key = 'labels'
    else:
        label_key = 'fine_labels'
    data_dict = unpickle(file_path)
    for label, file_name, data in zip(data_dict[label_key], data_dict['filenames'], data_dict['data']):
        image_path = os.path.join(dataset_path, subset, str(label), file_name)

        image = np.reshape(data, (3, 32, 32))
        image = np.rollaxis(image, 0, 3)[:, :, [2, 1, 0]]

        cv2.imwrite(image_path, image)


def export_cifar(tar_path, dataset_path, cifar_10=True, tmp_dir='/tmp'):
    # type: (str, str, bool, str) -> None
    """
    Export dataset.
    :param tar_path: dataset tar.gz file path
    :param dataset_path: output path of images.
    :param cifar_10: dataset type. CIFAR-10 or CIFAR-100.
    :param tmp_dir: folder to store temporary files.
    """
    if cifar_10:
        file_names = {
            'folder': 'cifar-10-batches-py',
            'train': 'data_batch_{}',
            'test': 'test_batch',
            'meta': 'batches.meta'
        }
        num_classes = 10
    else:
        file_names = {
            'folder': 'cifar-100-python',
            'train': 'train',
            'test': 'test',
            'meta': 'meta'
        }
        num_classes = 100

    print('Extracting {} to {} ...'.format(tar_path, tmp_dir))
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_dir)

    tmp_path = os.path.join(tmp_dir, file_names['folder'])

    print('Exporting images ...')
    for i in range(num_classes):
        train_path = os.path.join(dataset_path, 'train', str(i))
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        test_path = os.path.join(dataset_path, 'test', str(i))
        if not os.path.exists(test_path):
            os.makedirs(test_path)

    if cifar_10:
        for n in range(5):
            export_images(os.path.join(tmp_path, file_names['train'].format(n + 1)), dataset_path, 'train', cifar_10)
    else:
        export_images(os.path.join(tmp_path, file_names['train']), dataset_path, 'train', cifar_10)

    export_images(os.path.join(tmp_path, file_names['test']), dataset_path, 'test', cifar_10)

    print('Exporting meta.json to {} ...'.format(dataset_path))
    meta_file_path = os.path.join(tmp_path, file_names['meta'])
    meta_dict = unpickle(meta_file_path)

    if cifar_10:
        output_dict = {'label_names': meta_dict['label_names']}
    else:
        output_dict = {
            'label_names': meta_dict['fine_label_names'],
            'coarse_label_names': meta_dict['coarse_label_names']
        }
    with open(os.path.join(dataset_path, 'meta.json'), 'w') as f:
        json.dump(output_dict, f)

    print('Removing temporary files ...')
    shutil.rmtree(tmp_path)


class CIFAR(object):
    def __init__(self, dataset_path, num_threads=8, batch_size=64,
                 shuffle=True, normalize=True, augment=True, one_hot=True):
        # type: (str, int) -> CIFAR
        """
        :param dataset_path: The dataset folder path.
        :param num_threads: number of threads.
        :param batch_size: batch size.
        :param shuffle: shuffle or not.
        :param normalize: normalize or not.
        :param augment: augment or not.
        """
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
        with open(os.path.join(self.dataset_path, 'meta.json')) as f:
            meta_dict = json.load(f)
        self.label_names = meta_dict['label_names']
        self.num_classes = len(self.label_names)

        all_files = {'train': [], 'test': []}
        for subset in all_files.keys():
            for i in range(self.num_classes):
                folder = os.path.join(self.dataset_path, subset, str(i))
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
                all_files[subset] += files

        train_set_size = len(all_files['train'])
        test_set_size = len(all_files['test'])

        shuffle_index = np.arange(train_set_size)
        np.random.shuffle(shuffle_index)

        train_files = tf.constant(np.array(all_files['train'])[shuffle_index])
        train_labels = tf.constant(
            np.arange(self.num_classes).repeat(train_set_size // self.num_classes)[shuffle_index])

        shuffle_index = np.arange(test_set_size)
        np.random.shuffle(shuffle_index)

        test_files = tf.constant(np.array(all_files['test'])[shuffle_index])
        test_labels = tf.constant(
            np.arange(self.num_classes).repeat(test_set_size // self.num_classes)[shuffle_index])

        self.train_set = tf.contrib.data.Dataset.from_tensor_slices((train_files, train_labels))
        self.train_set_size = train_set_size

        self.test_set = tf.contrib.data.Dataset.from_tensor_slices((test_files, test_labels))
        self.test_set_size = test_set_size

    def _measure_mean_and_std(self):
        """
        measure mean and std of random samples from train set. number of samples is min(50000, train_set_size).
        """
        num_samples = min(50000, self.train_set_size)
        dataset = self.train_set.shuffle(buffer_size=num_samples)
        dataset = dataset.map(
            self._read_image_func,
            num_threads=self.num_threads,
            output_buffer_size=2 * self.batch_size)
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
        image_decoded = tf.image.decode_image(image_string)
        image_decoded = tf.reverse(image_decoded, axis=[-1])
        image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_float.set_shape([32, 32, 3])
        return image_float, label

    def _normalize_func(self, image, label):
        image_float = (image - self.mean) / self.std
        # image_float = tf.image.per_image_standardization(image)
        return image_float, label

    def _augment_func(self, image, label):
        image_float = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image_float = tf.random_crop(image_float, (32, 32, 3))
        image_float = tf.image.random_flip_left_right(image_float)
        # image_float = tf.image.random_flip_up_down(image_float)
        return image_float, label

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
            num_threads=self.num_threads,
            output_buffer_size=2 * self.batch_size)

        self.test_set = self.test_set.map(
            _test_pre_process_fun,
            num_threads=self.num_threads,
            output_buffer_size=2 * self.batch_size)

        self.train_set = self.train_set.batch(self.batch_size)
        self.test_set = self.test_set.batch(self.batch_size)


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    # export_cifar('/home/alvin/Work/cifar-10-python.tar.gz', '/home/alvin/Work/CIFAR10')
    cifar = CIFAR('/home/alvin/Work/CIFAR10', shuffle=False, normalize=False, augment=False, one_hot=False)
    dataset = cifar.test_set
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


if __name__ == '__main__':
    main()
