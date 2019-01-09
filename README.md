# Learning Very Sparse Convolution Under a Pseudo-Tree Constraint

This repository is a codebase for the CS222 course project *Learning Very Sparse Convolution Under a Pseudo-Tree Constraint* in PyTorch. The author is Hao-Tian Tang.

## Dependencies

This codebase is written in Python. Dependencies include:

* Python >= 3.6
* [PyTorch](https://github.com/pytorch/pytorch) >= 0.4

We **don't** support CPU training and evaluation, please run it on GPUs. The GPU memory requirement is 7GB (maximum).

## CIFAR-10

### Dataset

We assume that the dataset is located at `/home/tang/machine-learning/CS222-project/data` by default.

This location is hardcoded in my code, if you want to change it, please search for this string and replace it as you will.

### Training

The following line of code prunes VGG16 network with batch normalization. Typically, you can achieve 17-25x compression rate (because we use random grouping, the results have high variance) with 3-3.5% accuracy loss with this code.
```
python tree_convolution.py
```

The following line of code prunes ResNet164 network. If you only prune layer 115-164, the pruning ratio will be around 2.6x, the accuracy loss is 0.6%.
```
python tree_convolution_resnet.py
```

The following line of code prunes DenseNet40 network. You will achieve a compression ratio of 3.1, accuracy loss is around 3.8%.
```
python tree_convolution_densenet.py
```

You may adjust the code by yourself, commenting functions like *pretrain*, *train_all* and *retrain* will help you perform only specific stage of training in this project, instead of all. 