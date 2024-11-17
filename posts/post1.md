# Equivariant Neural Networks

In this post, we will talk about the mathematical concept of equivariance and its use in building neural networks that respect symmetry.


\toc

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

One can check that the operation \begin{equation}
    \text{Cov}_{h,\eta}: f\mapsto f*_{G,\eta}h
\end{equation} is $G$-equivariant.

---
##  $SO(3)$-equivariance with spherical harmonics

Now let's look at an implementation of $SO(3)$-equivariant network from [this paper][paper1]. Here, we consider spherical functions in $L^2(\mathbb{S}^2)$ as data. Under the integrability assumption, we can perform the **spherical Fourier transform (SFT)** \begin{align}
    f &= \sum_{l=0}^\infty\sum_{m=-\ell}^\ell f^m_\ell Y^m_\ell,\\ f^m_\ell &= \langle f,Y^m_\ell\rangle_{L^2(\mathbb{S}^2)} = \int_{\mathbb{S}^2}f(x)\overline{Y^m_\ell}(x)d\mathbb{S}^2.
\end{align}

Here, $\{Y^m_\ell\}$ are the [**spherical harmonics**](https://en.wikipedia.org/wiki/Spherical_harmonics) where $\ell$ is the **degree** of the corresponding homogeunous polynomials and $m$ is the **order**. Given a kernel function $h\in L^2(\mathbb{S}^2)$ and the north pole $N = (0,0,1)\in \mathbb{S}^2$, we can write \eqref{conv} as \begin{equation}
    f * h (x) = \int_{SO(3)} f(g\cdot N)h(g^{-1}\cdot x)d\mu(g).
\end{equation}
using the fact that the Haar measure $\mu$ is essentially a multiple of the uniform measure on $\mathbb{S}^2$, we can derive \begin{equation}\label{prod}
    (f*h)^m_\ell = \sqrt{\frac{16\pi^3}{2\ell+2}}f^m_\ell h^0_\ell.
\end{equation}

The key observation here is that the information we need for the kernel $h$ is only the 0-th order coefficients $\{h^{\ell}_0\}_{\ell=0}^\infty$.

---
## Implementation of convolution
Suppose that $f$ has **bandwidth** $b>0$. That is, $f^m_\ell = 0$ for all $\ell\geq b$. Then by \eqref{prod}, $f*h$ will also have bandwidth $b$ and thus it is suffice to keep track of $(h^0_0,\cdots,h^{b-1}_0)$.

```python
class SO3_Conv(nn.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth
        h0 = torch.Tensor(bandwidth, 1)
        self.h0 = nn.Parameter(h0)
        nn.init.kaiming_uniform_(self.h0)

    def forward(self, x):
        x = SFT(x, self.bandwidth)
        weights = torch.sqrt(16*math.pi**3/torch.arange(2, 2*b+2, 2)) * self.h0
        return ISFT(x*weights)
```

Due to finite bandwidth, we can also calculate the Fourier coefficients based on only $(2b+1)^2$ equi-angular sample points on $\mathbb{S}^2$: \begin{align}\label{sampling}
    f^m_\ell  &=  \frac{\sqrt{2\pi}}{2b}\sum_{j=0}^{2b-1}\sum_{k=0}^{2b-1} w^{(b)}_jf(x_{j,k})\overline{Y^m_\ell}(x_{j,k}),\\
    x_{j,k}&= (\cos(j\pi/b)\sin(k\pi/b),\sin(j\pi/b)\cos(k\pi/b),\cos(j\pi/b)),
\end{align} where $w^{(b)}_j$ are predetermined weights on $\{x_{j,k}\}$. Hence, to implement $\text{Cov}_{h,\eta}$, we can first find coefficients $f^m_\ell$ with \eqref{sampling}. Then we find apply the pointwise product by \eqref{prod} to get the coefficients for $\text{Cov}_{h,\eta}(f)$.

---
## Non-linearity
In practice, the nonlinear layer is done by the standard pointwise operation: \begin{equation}
    \text{NL}_\sigma: f \mapsto \sigma \circ f.
\end{equation} One can easily check that $\text{NL}_\sigma$ is equivariant. 

\block{Warning!}{Operation $\text{NL}_\sigma$ **does not** preserve the bandwith of the data. In fact, $\text{NL}_\sigma(f)$ can have infinite bandwidth regardless of the bandwith of $f$. Therefore, computing the Fourier coefficients with \eqref{sampling} after a non-linearity operation will introduce errors (See [equivariance error analysis](../post1/#equivariant_error_analysis)).}

---

## Spectral pooling
Here we introduced a pooling layer that acts as a [low-pass filter](https://en.wikipedia.org/wiki/Low-pass_filter) with cutoff frequency $b/2$. In practice, we can simply set $f^m_\ell$ to be zero for all $b/2<\ell < b$. 

---

## Invariant descriptor
In tasks such as image classification, the output is invariant to $SO(3)$ actions (equivalently, $SO(3)$ acts trivially on the output space). Therefore, we would like the output to be $SO(3)$-**invariant**. One way to achieve this is to use the following operation to produce an output vector: \begin{align}
    \text{Des}: f &\mapsto (||\mathbf{f}^0||,\dots,||\mathbf{f}^{b-1}||)\\
    \mathbf{f}^\ell &= (f^{-\ell}_\ell,f^{-\ell+1}_\ell,\dots, f^\ell_\ell).
\end{align} The fact that each $\mathbf{f}^\ell$ is $SO(3)$-invariant follow from that the action $SO(3)$ on $\mathbf{Y}^\ell := \text{span}\{Y^m_\ell| |m|\leq \ell\}$ is representable by [Wigner D-matrices](https://en.wikipedia.org/wiki/Wigner_D-matrix), which are unitary.

---

## Equivariant error analysis

The non-linearity layers are the only ones that introduce equivariant errors. To see this, we define the distribution \begin{equation}
    s = \frac{\sqrt{2\pi}}{2b} \sum_{j=0}^{2b-1}\sum_{k=0}^{2b-1} w^{(b)}_j\delta_{x_{i,j}},
\end{equation} given the equi-angular grid $\{x_{i,j}\}$.

We note that given a function (or a nice enough distribution) in $L^2(\mathbb{S}^2)$. When we use the sampling algorithm to approximate the Fourier coefficients, we are implicitly implementing an orthogonal projection of the atomic distribution \begin{equation}
    fs = \frac{\sqrt{2\pi}}{2b} \sum_{j=0}^{2b-1}\sum_{k=0}^{2b-1} w^{(b)}_jf(x_{i,j})\delta_{x_{i,j}}
\end{equation} onto the subspace $\mathbf{Y}^{[b]}:=\text{span}\{Y^m_\ell| |m| \leq \ell < b\}$. We can see that the operation $f\mapsto fs$ is **not** $SO(3)$-equivariant as \begin{align}
    (g\cdot f)s &= \frac{\sqrt{2\pi}}{2b} \sum_{j=0}^{2b-1}\sum_{k=0}^{2b-1} w^{(b)}_jf(g\cdot x_{i,j})\delta_{x_{i,j}},\\
    g\cdot (fs) &= \frac{\sqrt{2\pi}}{2b} \sum_{j=0}^{2b-1}\sum_{k=0}^{2b-1} w^{(b)}_jf(x_{i,j})\delta_{g^{-1}\cdot x_{i,j}}.\\
\end{align}

If we write $f = f_b + f_r$ where $f_b\in \mathbf{Y}^{[b]}$ and $f^b \perp f^r$, we can see that operation $\text{Proj}_{\mathbf{Y}^{[b]}}(\cdot s)$ is only $SO(3)$-equivariant on the set \begin{equation}
    \{f\in L^2(\mathbb{S}^2)~|~(g\cdot f_r)s - g\cdot (f_rs) \perp \mathbf{Y}^{[b]}\quad \text{for all}\quad g\in SO(3)\}.
\end{equation}

[paper1]: https://arxiv.org/abs/1711.06721