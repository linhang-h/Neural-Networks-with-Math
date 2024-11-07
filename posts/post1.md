# Equivariant Neural Networks
---

## What is equivariance

Given a group $G$ that acts on sets $E$ and $F$, a function $f: E \to F$ is $G$-[**equivariant**](https://en.wikipedia.org/wiki/Equivariant_map#:~:text=In%20mathematics%2C%20equivariance%20is%20a,the%20action%20of%20the%20group.) if \begin{equation}
    f(g\cdot e) = g\cdot f(e)\quad \text{for all}\quad e\in E.
\end{equation}

Basically, function $f$ is $G$-equivariant when it preserves the symmetry imposed by the group $G$.

---

## Why do we need equivariant neural networks

The task of the feedforward neural networks is to transform the input data into output data. That is, they are functions from the input space to the output space. We might want the neural networks to be equivariant under the assumption that data have symmetry informtation we want to preserve through the tranformation. 

An example one can think of is that of image classification. When classifying images, the labels we give to the images are usually rotationally invariant. Hence, if the classifier neural network is equivariant to rotation, one can potentially save significant amount of computational power and make the model more generalizable.

---

## Equivariant kernels for CNN

To build an equivariant network, we can take the framwork of the convolutional neural network and make the kernels/filters equivariant.

Suppose our data are vectors indexed by set $E$ (or functions from $E$ to $\C$) acted by group $G$. Then group $G$ also acts on our input data space $\C^E$ via \begin{equation}
    (g\cdot f)(x) = f(g\cdot x).
\end{equation}

To transform the data $G$-equivariantly, we set a kernel function $h: E\to \C$ with a pole $\eta\in E$ and define the [**group convolution**](Convolutions on groups) \begin{equation}
    f*_{G,\eta}h(x):= \int_G f(g\cdot \eta)h(g^{-1}\cdot x)d\mu(g),\label{conv}
\end{equation} where $\mu$ is the [**Haar measure**](https://en.wikipedia.org/wiki/Haar_measure) of the group.

One can check that the function \begin{equation}
    \text{Cov}_{h,\eta}: f\mapsto f*_{G,\eta}h
\end{equation} is $G$-equivariant.

---
##  $SO(3)$-equivariance with spherical harmonics

Now let's look at an implementation of $SO(3)$-equivariant network from [this paper][paper1]. Here, we consider spherical functions in $L^2(\mathbb{S}^2)$ as data. Under the integrability assumption, we can perform the **spherical Fourier transform (SFT)** \begin{equation}
    f = \sum_{l=0}^\infty\sum_{m=-\ell}^\ell f^m_\ell Y^m_\ell,\quad f^m_\ell = \langle f,Y^m_\ell\rangle_{L^2(\mathbb{S}^2)} = \int_{\mathbb{S}^2}f(x)\overline{Y^m_\ell}(x)d\mathbb{S}^2.
\end{equation}

Here, $\{Y^m_\ell\}$ are the [**spherical harmonics**](https://en.wikipedia.org/wiki/Spherical_harmonics) where $\ell$ is the **degree** of the corresponding homogeunous polynomials and $m$ is the **order**. Given a kernel function $h\in L^2(\mathbb{S}^2)$ and the north pole $N = (0,0,1)\in \mathbb{S}^2$, we can write \eqref{conv} as \begin{equation}
    f * h (x) = \int_{SO(3)} f(g\cdot N)h(g^{-1}\cdot x)d\mu(g).
\end{equation}
using the fact that the Haar measure $\mu$ is essentially a multiple of the uniform measure on $\mathbb{S}^2$, we can derive \begin{equation}
    (f*h)^m_\ell = 2\pi\sqrt{\frac{4\pi}{2\ell+2}}f^\ell_m h^\ell_0.
\end{equation}

The key observation here is that the information we need for the kernel $h$ is only the 0-th order coefficients $\{h^{\ell}_0\}_{\ell=0}^\infty$.

---
## Implementation of convolution

---
## Spectral filtering



---


[paper1]: https://arxiv.org/abs/1711.06721