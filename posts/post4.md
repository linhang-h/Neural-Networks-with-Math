+++
title = "Introduction to G-CNNs"

excerpt = "Group convolution is all you need"

image = ""

authors = "Michael"
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

### Convolutional Neural Networks

CNNs achieve translation invariance through the structure of their convolutional layers. Mathematically, a convolutional layer takes in a signal $f:\mathbb{R}^n\to \mathbb{R}$ and computes a feature map $\hat{f}(x)$ as $$\hat{f}(x):=(\kappa * f)(x) = \int_{\mathbb{R}^n} \kappa(y) f(x - y) \, dy$$, where $\kappa$ is the convolution *kernel* (aka. filter). If the signal $f$ is translated by a vector $t \in \mathbb{R}^n$, say $f^\prime(x) = f(x - t)$, then the resulting feature map shifts correspondingly: $$\hat{f}^\prime(x) = (\kappa * f')(x) = \int_{\mathbb{R}^n}\kappa(y)f(x-t-y)dy = \hat{f}(x-t).$$ This can be organized into a commutative diagram: 
$$\begin{tikzcd}
	f & {\hat{f}} \\
	{f^\prime} & {\hat{f}^\prime}
	\arrow["{\tiny{\text{ convolution }}}"{marking, allow upside down}, from=1-1, to=1-2]
	\arrow["{\tiny{\text{ translation }}}"{marking, allow upside down}, from=1-1, to=2-1]
	\arrow["{\tiny{\text{ translation }}}"{marking, allow upside down}, from=1-2, to=2-2]
	\arrow["{\tiny{\text{ convolution }}}"{marking, allow upside down}, from=2-1, to=2-2]
\end{tikzcd}$$
In plain words, *convolving first and then translating results the same as translating first and then convolving*. This demonstrates that the convolution operation is *compatible with translational symmetries* and is exactly what makes CNNs so good with acoustic and visual signals. 

### Group Equivariant CNNs

Our physical world is brimming with symmetries -- from the discrete polytopal symmetries of a virus to the rotational symmetries of the surface of a planet. 

*TODO: add pictures of the icosahedral and SO(3) groups*

It is an inevitable consequence that the data inputted in neural networks come with a lot of built-in symmetries. Respecting, or better, leveraging those symmetries has proven to be the key in speeding up training and improving accuracy. It is therefore requisite that we build network architectures that are *compatible with more general symmetries*, i.e. groups other than $\R^n$. 

To do this, we ask the generalized convolution layers to satisfy a similar property: 

<p style="text-align: center;"> Group convolution followed by group translation should be the same as group translation followed by group convolution. </p>

$$\begin{tikzcd}
	f & {\hat{f}} \\
	{f^\prime} & {\hat{f}^\prime}
	\arrow["{\tiny{G-\text{ convolution }}}"{marking, allow upside down}, from=1-1, to=1-2]
	\arrow["{\tiny{G-\text{ translation }}}"{marking, allow upside down}, from=1-1, to=2-1]
	\arrow["{\tiny{G-\text{ translation }}}"{marking, allow upside down}, from=1-2, to=2-2]
	\arrow["{\tiny{G-\text{ convolution }}}"{marking, allow upside down}, from=2-1, to=2-2]
\end{tikzcd}$$




## Representation Theory and Harmonic Analysis on Locally Compact Groups





## $G$-CNN: Regular v.s. Steerable Networks






## $G$-GNN: Equivariant Graph Neural Networks
