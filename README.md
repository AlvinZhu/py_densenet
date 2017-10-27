# Tensorflow Implementation of DenseNets

Two types of [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (DenseNets) are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- Cifar10
- Cifar10+ (with data augmentation)
- Cifar100
- Cifar100+ (with data augmentation)
- ImageNet

Example run:

```sh
python train_densenet_cifar.py
```



There are also many [other implementations](https://github.com/liuzhuang13/DenseNet) - they may be useful also.

Citation:

```
@article{Huang2016Densely,
       author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
       title = {Densely Connected Convolutional Networks},
       journal = {arXiv preprint arXiv:1608.06993},
       year = {2016}
}
```



## Dependencies

- Model was tested with Python 2.7 with and without CUDA.
- Model should work as expected with TensorFlow >= 1.3. 

Repo supported with requirements file - so the easiest way to install all just run `pip install -r requirements.txt`.