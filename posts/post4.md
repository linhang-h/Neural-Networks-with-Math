+++
title = "Introduction to $G$-CNNs"

excerpt = "*Group convolution is all you need.*"

image = ""
+++

# Group Equivariant Neural Networks

This webpage consists of a set of notes we created while going through the course *An Introduction to Group Equivariant Deep Learning* by Erik Bekkers at the University of Amsterdam. The course materials are freely available online at [uvagedl](https://uvagedl.github.io/), and the lecture series is available as a public [playlist](https://www.youtube.com/playlist?list=PL8FnQMH2k7jzPrxqdYufoiYVHim8PyZWd) on YouTube. We owe our deepest gratitude for *Dr.* Erik Bekkers and all other people who created this wonderful course and made the materials publically available. We largely follow the structure of these lectures, even though some of the mathematical details are our own takes on the original arguments. The intrigued reader is highly recommended to check out the original lectures, where concepts are illustrated with many amazing graphics. We hope to duely justify the slogan '*Group convolution is all you need*' through this short digest. 

\toc

## Introduction

Convolutional Neural Networks (CNNs) possess the distinguishing property of translation invariance in their convolutional layers, allowing them to maintain the inherent spatial structure of image data. This characteristic enables CNNs to excel in various complex tasks, including edge detection, feature extraction for object recognition, and semantic segmentation.

However, image data often come with additional structures, such as rotational or reflectional symmetries, which are not built-in symmetries of traditional CNNs. Group-equivariant CNNs ($G$-CNNs) extend the translation-invariance property of traditional CNNs into equivariance of more general symmetry groups, such as rotations, reflections, or scaling transformations. This is achieved by defining convolutional layers over topological groups rather than the usual vector space $\mathbb{R}^n$. As a result, $G$-CNNs efficiently represents and processes data with a lot of symmetries, in many cases improving the performance margin significantly. One of the first applications where $G$-CNNs are extremely useful is medical imaging, where cells and organs can appear in various orientations ([Bekkers et.al. 2018]). 

!!!TBA

## $G$-CNN: The First Class

Say we want to build a network which recognizes the letter $G$. Regardless of where we position the $G$ in our input image, we want the feature map of the network to activate in the same way where it detects the $G$ somewhere in the image. This sort of *translation equivariance* is a defining feature of Convolutional Neural Networks (CNNs), and in fact what makes it so useful in pattern recognition tasks, especially in image and spatial data. 

CNNs achieve translation invariance through the structure of their convolutional layers. Mathematically, a convolutional layer computes a feature map \( f(x) \) as \( (k * I)(x) = \int_{\mathbb{R}^n} k(u) I(x - u) \, du \), where \( k(u) \) is the kernel (filter), and \( I(x) \) is the input. If the input is translated by a vector \( t \in \mathbb{R}^n \), such that \( I'(x) = I(x - t) \), the resulting feature map shifts correspondingly: \( f'(x) = (k * I')(x) = f(x - t) \). This demonstrates that the convolution operation is *compatible with translational symmetries*. 

Our physical world is brimming with symmetries -- from the discrete polytopal symmetries of a virus to the rotational symmetries of the surface of a planet. It is an inevitable result that the data inputted in neural networks 







## Representation Theory and Harmonic Analysis on Locally Compact Groups





## $G$-CNN: Regular v.s. Steerable Networks






## $G$-GNN: Equivariant Graph Neural Networks