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


def measure_mean_and_std(dataset):
    """
    measure mean and std of random samples from train set. number of samples is min(50000, train_set_size).
    :type dataset: dataset object
    """
    subset = dataset.train_set
    num_samples = min(50000, dataset.train_set_size)
    batch_size = 1000

    def pre_process_fn(filename, label):
        image, label = dataset.read_image_func(filename, label)
        shape = tf.shape(image)
        shorter_edge = tf.minimum(shape[0], shape[1])
        image = tf.image.resize_image_with_crop_or_pad(image, shorter_edge, shorter_edge)
        image = tf.image.resize_images(image, (dataset.image_shape[0], dataset.image_shape[1]))
        return image, label
    subset = subset.map(
        pre_process_fn,
        num_threads=dataset.num_threads,
        output_buffer_size=2 * dataset.batch_size)
    subset = subset.batch(batch_size)
    iterator = subset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
    std = tf.sqrt(variance)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        out_mean = np.zeros([dataset.image_shape[2]])
        out_std = np.zeros([dataset.image_shape[2]])

        for i in range(num_samples // batch_size):
            tmp_mean, tmp_std = sess.run([mean, std])
            out_mean += tmp_mean
            out_std += tmp_std

        out_mean /= num_samples // batch_size
        out_std /= num_samples // batch_size

        print('mean:{}, std:{}'.format(out_mean, out_std))

    with open(os.path.join(dataset.dataset_path, 'meta.json')) as f:
        meta_dict = json.load(f)
    meta_dict['mean'] = out_mean.tolist()
    meta_dict['std'] = out_std.tolist()

    with open(os.path.join(dataset.dataset_path, 'meta.json'), 'w') as f:
        json.dump(meta_dict, f)


class CIFAR(object):
    def __init__(self, dataset_path, image_shape=(32, 32, 3), num_threads=8, batch_size=128,
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
            # self.normalize = False
            self.mean = tf.constant(0.0)
            self.std = tf.constant(1.0)

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

    def read_image_func(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=self.image_shape[2])
        image_decoded = tf.reverse(image_decoded, axis=[-1])
        image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
        return image_float, label

    def normalize_func(self, image, label):
        # image_float = (image - self.mean) / self.std
        image_float = tf.image.per_image_standardization(image)
        return image_float, label

    def augment_func(self, image, label):
        image = tf.image.resize_image_with_crop_or_pad(image, self.image_shape[0] + 8, self.image_shape[1] + 8)
        image = tf.random_crop(image, self.image_shape)
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image)
        return image, label

    def one_hot_func(self, image, label):
        onehot_label = tf.one_hot(indices=tf.cast(label, tf.int32), depth=self.num_classes)
        return image, onehot_label

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
                image, label = self.augment_func(image, label)
            else:
                image = tf.image.resize_images(image, (self.image_shape[0], self.image_shape[1]))
            image.set_shape(self.image_shape)
            if self.one_hot:
                image, label = self.one_hot_func(image, label)
            return image, label

        def _test_pre_process_fun(filename, label):
            image, label = self.read_image_func(filename, label)
            if self.normalize:
                image, label = self.normalize_func(image, label)
            image.set_shape(self.image_shape)
            if self.one_hot:
                image, label = self.one_hot_func(image, label)
            return image, label

        self.train_set = self.train_set.map(
            _train_pre_process_fun,
            num_parallel_calls=self.num_threads)

        self.test_set = self.test_set.map(
            _test_pre_process_fun,
            num_parallel_calls=self.num_threads)

        self.train_set = self.train_set.batch(self.batch_size)
        self.test_set = self.test_set.batch(self.batch_size)


def main_prepare_dataset():
    export_cifar('/backups/datasets/cifar-10-python.tar.gz', '/backups/work/CIFAR10')
    cifar = CIFAR('/backups/work/CIFAR10', read=False)
    measure_mean_and_std(cifar)


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    cifar = CIFAR('/backups/work/CIFAR10', shuffle=False, normalize=False, augment=False, one_hot=False)
    dataset = cifar.test_set
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    features, label = iterator.get_next()
    with tf.Session() as sess:
        images, labels = sess.run([features, label])
        for image, label in zip(images, labels):
            print(cifar.label_names[label])
            cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('image', image)
            cv2.waitKeyEx(0)


if __name__ == '__main__':
    # main_prepare_dataset()
    main()
