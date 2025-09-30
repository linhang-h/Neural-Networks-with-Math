+++
title = "Learning Efficiency under Manifold Hypothesis"

excerpt = "How the underlining geometry of the data affects the difficulty of learning"

image = "/assets/images/manifold.png"

authors = "Linhang"
+++

# Learning Efficiency under Manifold Hypothesis

In this post, we will go through the concept of manifold hypothesis. We will how the dimension of the underlining manifold can be estimated and how the geometry of the underlining manifold can affect the difficulty of the learning task.

\toc

## What are manifolds and what is the manifold hypothesis?

Manifolds in the machine learning context are usually embedded Riemannian manifolds to a mathematician. That means they are subsets of $\R^n$ that are locally diffeomorphic to $\R^k$ for some $k \leq n$.

The manifold hypothesis posits that most of the high-dimensional real-world data sets lie on or near a low-dimensional manifold. For instance, while we encode image datas using pixel values (the dimension of the data equal to the number of pixels), for computer vision tasks, the information we need is usually independent of brightness, saturation and constrast of the images themselves, which cuts down the actually dimension of data.

## Estimating intrinsic dimension
Intrinsic dimension can be a useful indicator of the geometric complexity of the data set. There have been many algorithms developed for  estimating intrinsic dimension. One of the techniques that rises to popularity lately is one that utilizes diffusion models (see the previous post for the detailed discussion). 

The idea behind it is as follows: Given a data point $x_0$ on the data manifold $M$, we can perturb it slightly to get a noised data point $x_{t_0}$. Then from $x_{t_0}$, the score function $\nabla_x \log p_{t_0}(x_{t_0})$, which the diffusion model aimed to learn, approximately points in the direction of the orthogonal projection of $x_{t_0}$ on $M$. This implies that when $t_0$ is small, $\nabla_x \log p_{t_0}(x_{t_0})$ is approximately in the normal bundle $N_{x_0}M$. Note that if $M$ is a manifold with ambient dimension $n$, we have \begin{equation}
    \mathrm{dim}(M) = n - \mathrm{dim}(N_xM).
\end{equation} We can estimate $\mathrm{dim}(N_xM)$ by repeatedly sampling $x_{t_0}$ and applying PCA to all the resulting $\nabla_x \log p_{t_0}(x_{t_0})$ vectors.

## Efficiently PAC learnability & statistical queries

## Efficiently sampleable manifolds

## Construct hard-to-learn functions using Boolean functions