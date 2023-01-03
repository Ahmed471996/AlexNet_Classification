# AlexNet_Classification

## Introduction
we will be building AlexNet (paper). We are building this CNN from scratch in PyTorch, and will also see how it performs on a real-world dataset.

We will start by exploring the architecture of AlexNet. We will then load and analyze our dataset, CIFAR10, using the provided class from torchvision. Using PyTorch, we will build our AlexNet from scratch and train it on our data. Finally, we will see how the model performs on the unseen test data.

## AlexNet

AlexNet is a deep convolutional neural network, which was initially developed by Alex Krizhevsky and his colleagues back in 2012. It was designed to classify images for the ImageNet LSVRC-2010 competition where it achieved state of the art results. You can read in detail about the model in the original research paper.

Here, I'll summarize the key takeaways about the AlexNet network. Firstly, it operated with 3-channel images that were (224x224x3) in size. It used max pooling along with ReLU activations when subsampling. The kernels used for convolutions were either 11x11, 5x5, or 3x3 while kernels used for max pooling were 3x3 in size. It classified images into 1000 classes. It also utilized multiple GPUs.

![image](https://user-images.githubusercontent.com/101316217/210410813-4d4897ab-cb51-469d-ab85-a03164cf193e.png)


## Dataset

Let's start by loading and then pre-processing the data. For our purposes, we will be using the CIFAR10 dataset. The dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Here are the classes in the dataset, as well as 10 random sample images from each:


![image](https://user-images.githubusercontent.com/101316217/210411606-8953247f-14e5-45df-995e-2e06da473e75.png)


## Tools
PyTorch

Colab

## Future Work
You can try using different datasets. 

You can experiment with different hyperparameters and see the best combination of them for the model.
