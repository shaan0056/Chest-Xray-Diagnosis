# Chest-Xray-Diagnosis using Generative Adversarial Networks

## Introduction

This study proposes to simulate and augment CheXpert Chest x-ray imaging dataset using Generative Adversarial Networks techniques
for accurate classification of rare diseases, which otherwise could not be effectively diagnosed in an imbalanced
dataset and then compare the accuracy of the classifier against the CheXpert baseline model.

Dataset can be downloaded from [here](https://stanfordmlgroup.github.io/competitions/chexpert/)

## Approach

The inital approach used the DenseNet121 pretrained models implemented in Pytorch, performed chest x-ray image classification with the CheXpert training dataset using Tranfer Learning and set the benchmarks to improve upon. Then, GAN techniques for data augmentation
and simulation was applied to the trained models using Pytorch-GAN and trained the aforementioned ConvNets against the newly generated images.Finally, the final trained models was evaluated using AUC ROC curve on the validation set and the performances was compared.

The project is implemented using Python and Pytorch. More details can be found in the final paper ChestXrayUsingGanPaper.pdf.

