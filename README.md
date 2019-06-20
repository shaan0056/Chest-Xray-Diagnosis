# Chest-Xray-Diagnosis using Generative Adversarial Networks

## Introduction

This study proposes to simualate and augment CheXpert Chest x-ray imaging dataset using Generative Adversarial Networks techniques
for accurate classification of rare diseases, which otherwise could not be effectively diagnosed in an imbalanced
dataset and then compare the accuracy of the classifier against the CheXpert baseline model.

Dataset can be downloaded from [here](https://stanfordmlgroup.github.io/competitions/chexpert/)

## Approach

The inital approach will use the DenseNet121 pretrained models implemented in Pytorch, perform chest x-ray image classification with the CheXpert training dataset using Tranfer Learning and set the benchmarks to improve upon. Then, GAN techniques for data augmentation
and simulation will be applied to the trained models using Pytorch-GAN and train the aforementioned ConvNets against the newly generated images.Finally, the final trained models will be evaluated using AUC ROC curve on the validation set and the performances will be
compared.

The project is implemented using Python and Pytorch. More details can be found in the final paper ChestXrayUsingGanPaper.pdf.

