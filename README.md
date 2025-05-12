# Deep Learning Course Notebooks

## Environment

These notebooks are designed to run in the JupyterLab environment provided by NVIDIA through their [Deep Learning Institute course](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-01+V3). NVIDIA provides a ready-to-execute JupyterLab environment with all necessary dependencies pre-installed and configured for GPU acceleration. No local setup is required as the environment is fully managed by NVIDIA's learning platform.

## Description

This repository contains a series of Jupyter notebooks covering various topics in Deep Learning. The notebooks progress through fundamental concepts to more advanced applications.

## Contents

The following notebooks are included in this project:

**00_jupyterlab.ipynb**: Introduction to the JupyterLab environment, covering its interface, how to execute code cells, and manage GPU memory.

**01_mnist.ipynb**: Image classification of handwritten digits using the MNIST dataset. Covers data loading (TorchVision), preprocessing, building a simple neural network (PyTorch), training, and validation.

**02_asl.ipynb**: Image classification of American Sign Language (ASL) alphabet signs. Covers loading data from CSV (Pandas), creating custom PyTorch Datasets/DataLoaders, building a neural network, training, and discussion of overfitting.

**03_asl_cnn.ipynb**: Applying Convolutional Neural Networks (CNNs) for ASL image classification. Details data preparation for CNNs (reshaping for channels), explains various CNN layers (Conv2D, BatchNorm2d, MaxPool2D, Dropout), and demonstrates improved model training and performance.

**04a_asl_augmentation.ipynb**: Explores data augmentation techniques (RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter via TorchVision) to improve the ASL CNN model's generalization. Includes creating a custom PyTorch module for convolutional blocks and saving the trained model.

**04b_asl_predictions.ipynb**: Demonstrates deploying the trained ASL CNN model. Covers loading the saved model, preprocessing new unseen images (resizing, grayscaling), performing inference, and interpreting prediction outputs to identify ASL letters.

**05a_doggy_door.ipynb**: Introduces using pre-trained models (VGG16 from TorchVision, trained on ImageNet) for an "automated doggy door" application. Covers loading the model, preprocessing images to match model input, performing inference, and using ImageNet classes to identify dogs/cats.

**05b_presidential_doggy_door.ipynb**: Demonstrates transfer learning with a "presidential doggy door" example (identifying a specific dog, Bo). Covers freezing pre-trained VGG16 layers, adding new classification layers, training on a small custom dataset (Bo/not_bo images) with data augmentation, and fine-tuning the model.

**06_nlp.ipynb**: Introduction to Natural Language Processing (NLP) using BERT. Covers text tokenization (BertTokenizer, special tokens, WordPiece), segment IDs, masked language modeling (BertForMaskedLM) for word prediction and embeddings, and question answering (BertForQuestionAnswering) to extract answers from text.

**07_assessment.ipynb**: Final assessment. Involves training an image classification model to distinguish between fresh and rotten fruits (apples, oranges, bananas - 6 classes) with a target validation accuracy of 92%. Guides through using transfer learning (VGG16), data augmentation, and fine-tuning.
