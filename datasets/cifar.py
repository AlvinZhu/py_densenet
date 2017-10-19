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
    def __init__(self, dataset_path, num_threads=8, batch_size=64, shuffle=True, normalize=True, augment=True):
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

        self._load()
        self._pre_process()

    def _load(self):
        # type: () -> None
        """Load dataset from folder which contains images."""
        with open(os.path.join(self.dataset_path, 'meta.json')) as f:
            meta_dict = json.load(f)
        self.label_names = meta_dict['label_names']
        self.num_classes = len(self.label_names)
        self.mean = tf.constant(meta_dict['mean'])
        self.std = tf.constant(meta_dict['std'])

        all_files = {'train': [], 'test': []}
        for subset in all_files.keys():
            for i in range(self.num_classes):
                folder = os.path.join(self.dataset_path, subset, str(i))
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
                all_files[subset] += files

        train_files = tf.constant(np.array(all_files['train']))
        train_labels = tf.constant(np.arange(self.num_classes).repeat(len(all_files['train']) // self.num_classes))

        test_files = tf.constant(all_files['test'])
        test_labels = tf.constant(np.arange(self.num_classes).repeat(len(all_files['test']) // self.num_classes))

        self.train_set = tf.contrib.data.Dataset.from_tensor_slices((train_files, train_labels))
        self.train_set_size = len(all_files['train'])

        self.test_set = tf.contrib.data.Dataset.from_tensor_slices((test_files, test_labels))
        self.train_set_size = len(all_files['test'])

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
        image_float = tf.random_crop(image, (24, 24, 3))
        image_float = tf.image.resize_image_with_crop_or_pad(image_float, 32, 32)
        # image_float = tf.image.random_flip_left_right(image)
        # image_float = tf.image.random_flip_up_down(image_float)
        return image_float, label

    def normalize(self, images):
        images_float = tf.cast(images, tf.float32) / 255.0
        images_float = (images_float - self.mean) / self.std
        return images_float

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
            return image, label

        def _test_pre_process_fun(filename, label):
            image, label = self._read_image_func(filename, label)
            if self.normalize:
                image, label = self._normalize_func(image, label)
            return image, label

        self.train_set = self.train_set.map(
            _train_pre_process_fun,
            num_threads=self.num_threads,
            output_buffer_size=self.num_threads + self.batch_size)

        self.test_set = self.test_set.map(
            _test_pre_process_fun,
            num_threads=self.num_threads,
            output_buffer_size=self.num_threads + self.batch_size)

        self.train_set = self.train_set.batch(self.batch_size)
        self.test_set = self.test_set.batch(self.batch_size)


def measure_mean_and_std(dataset, meta_path, num_samples):
    # type: (tf.contrib.data.Dataset, str, int) -> None
    """
    measure mean and std of random samples from dataset. number of samples is num_samples.
    :param dataset: tf.contrib.data.Dataset object.
    :param meta_path: meta.json path. write mean and std to this file.
    :param num_samples: number of random samples.
    """
    dataset = dataset.shuffle(buffer_size=num_samples)
    dataset = dataset.batch(num_samples)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
    std = tf.sqrt(variance)

    with tf.Session() as sess:
        out_mean, out_std = sess.run([mean, std])
        print('mean: {}'.format(out_mean))
        print('std: {}'.format(out_std))

    with open(meta_path) as f:
        meta_dict = json.load(f)

    meta_dict['mean'] = out_mean.tolist()
    meta_dict['std'] = out_std.tolist()

    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f)


if __name__ == '__main__':
    # export_cifar('/backups/datasets/cifar-10-python.tar.gz', '/backups/work/CIFAR10')
    # cifar = CIFAR('/backups/work/CIFAR10')
    # export_cifar('/backups/datasets/cifar-100-python.tar.gz', '/backups/work/CIFAR100', cifar_10=False)
    # cifar = CIFAR('/backups/work/CIFAR100')
    # measure_mean_and_std(cifar.train_set, os.path.join(cifar.dataset_path, 'meta.json'), cifar.train_set_size)
    cifar = CIFAR('/backups/work/CIFAR10', shuffle=False, normalize=False, augment=True)
    dataset = cifar.train_set
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    with tf.Session() as sess:
        a = sess.run(features)
        # mean = sess.run(cifar.mean)
        # std = sess.run(cifar.std)
        # b = cv2.imread('/backups/work/CIFAR10/train/0/jumbo_jet_s_001462.png')
        # b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB) / 255.0
        # for i in range(b.shape[-1]):
        #     b[:, :, i] = ((b[:, :, i] - mean[i]) / std[i])
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('image', a)
        cv2.waitKeyEx(0)
        pass
