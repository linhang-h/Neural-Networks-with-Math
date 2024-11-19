# Diffusion Models and Ornstein-Uhlenbeck Processes

In this post, we will give a brief introduction to diffusion models and talk about their connections to Ornstein-Uhlenbeck processes.


\toc

---

## What are diffusion models
Diffusion models are a family of generative models. Broadly, these are the models that learn the data distribution by injecting noise via a diffusion process and then generate new samples by denoising. 

Diffusion models currently achieve the state-of-the-art performance in image generation and have also seen applications in other tasks.

---

## Ornstein-Uhlenbeck processes
The diffusion processes the models usually use are [Ornstein–Uhlenbeck processes](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), which are defined using SDE\begin{equation}\label{eq1}
    dX_t = \frac{g(t)^2}{2\sigma^2}(\mu - X_t)dt + g(t)dB_t,
\end{equation} where $\mu\in \R^n$ and $B_t$ is the $n$-dimensional Brownian motion. 

Unlike Brownian motion, the process $X_t$ has the stationary distribution $N(\mu,\sigma^2)$ and an explicit transition probabiliy: for $s\leq t$, conditioning on $X_s = x_s$, $X_t$ is normally distributed with mean $m_{s,t}$ and variance $\sigma^2_{s,t}$, where \begin{align*}
    m_{s,t} &= \mu + (x_s-\mu)\exp\left(-\frac{1}{2\sigma^2}\int_s^t g(t)^2 dt\right),\\
    \sigma^2_{s,t} &= \sigma^2\left(1-\exp\left(-\frac{1}{\sigma^2}\int_s^t g(t)^2 dt\right)\right).
\end{align*}

Fix an end time $T>0$ and set $\widetilde X_t = X_{T-t}$ be the time-reversal process for $X_t$. Then $\widetilde X_t$ satisfies the SDE \begin{equation*}
    d\widetilde{X}_t = \left[-\frac{g(T-t)^2}{2\sigma^2}(\mu-\widetilde{X}_t) + g(T-t)^2\nabla_xp_{T-t}(\widetilde{X}_t)\right]dt + g(T-t)dB'_t,
\end{equation*} where $p_t$ is the density of $X_t$ and $B'_t$ is a different Brownian motion.

---

## Noise injection & denoising
The framework of a diffusion model generally goes as follows: we first fix a sequence of time $0=t_0< t_1 <\dots< t_n = T$. Then given each  point $x_0\in\R^n$ sampled from the data distribution $p$, we can then add Gaussian noise to the data iteratively by sampling $x_i := X_{t_i}$, where $X_t$ follows SDE \eqref{eq1} starting at $x_0$. Since the transition probability is just Gaussian, we can sample $(X_n)$ efficiently.



---

## Score matching

---

## Trainable objective

---

## Conditioning