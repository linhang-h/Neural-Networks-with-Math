+++
title = "Diffusion Models and Ornstein-Uhlenbeck Processes"

excerpt = "Exploring diffusion models with a closer look at the underlying random processes"

image = "/assets/images/diffusion.jpg"

authors = "Linhang"
+++

# Diffusion Models and Ornstein-Uhlenbeck Processes

In this post, we will give a brief introduction to diffusion models and talk about their connections to Ornstein-Uhlenbeck processes.


\toc


## What are diffusion models
Diffusion models are a family of generative models. Broadly, these are the models that learn the data distribution by injecting noise via a diffusion process and then generate new samples by denoising. 

Diffusion models currently achieve the state-of-the-art performance in image generation and have also seen applications in other tasks.


## Ornstein-Uhlenbeck processes
The diffusion processes the models usually use are [Ornstein–Uhlenbeck processes](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), which are defined using SDE\begin{equation}\label{eq1}
    dX_t = \frac{g(t)^2}{2\sigma^2}(\mu - X_t)dt + g(t)dB_t,
\end{equation} where $\mu\in \R^d$, $B_t$ is the $d$-dimensional Brownian motion and $g(t)$ is a strictly positive function. 

Unlike Brownian motion, the process $X_t$ has the stationary distribution $N(\mu,\sigma^2)$ and an explicit transition probabiliy: for $s\leq t$, conditioning on $X_s = x_s$, $X_t$ is normally distributed with mean $m_{s,t}$ and variance $\sigma^2_{s,t}\mathbf{Id}$, where \begin{align}
    m_{s,t} &= \mu + (x_s-\mu)\exp\left(-\frac{1}{2\sigma^2}\int_s^t g(t)^2 dt\right),\\
    \sigma^2_{s,t} &= \sigma^2\left(1-\exp\left(-\frac{1}{\sigma^2}\int_s^t g(t)^2 dt\right)\right).\label{mean_var}
\end{align}

Fix an end time $T>0$ and set $\widetilde X_t = X_{T-t}$ be the time-reversal process for $X_t$. Then $\widetilde X_t$ satisfies the SDE \begin{equation}\label{eq2}
    d\widetilde{X}_t = \left[-\frac{g(T-t)^2}{2\sigma^2}(\mu-\widetilde{X}_t) + g(T-t)^2\nabla_x\log p_{T-t}(\widetilde{X}_t)\right]dt + g(T-t)dB'_t,
\end{equation} where $p_t$ is the density of $X_t$ and $B'_t$ is a different Brownian motion.


## Noise injection & denoising
The framework of a diffusion model generally goes as follows: we first fix a sequence of time $0=t_0< t_1 <\dots< t_n = T$, $g(t)$, $\mu$ and $\sigma$. Then given each  point $x_0\in\R^d$ sampled from the data distribution $p$, we can then add Gaussian noise to the data iteratively by sampling $x_i := X_{t_i}$, where $X_t$ follows SDE \eqref{eq1} starting at $x_0$. Since the transition probability is just Gaussian, we can sample $(X_n)$ efficiently. 

Assuming $T$ is large enough that $x_n$ is approximately distributed according to the stationary Gaussian distribtion. Then the objective will be learning the the information of $p_t$ so that we can approximate the time-reversal process \eqref{eq2} with initial data $\widetilde{X}_0 \sim N(\mu,\sigma^2)$.


## Score matching

Since $g(t)$, $\mu$ and $\sigma$ are known, we only need an estimate of $\nabla_x\log p_t$ to approximate the time-reserval process. The gradient of log density $\nabla \log p:\R^d\to \R^d$ is known as the **score**. The idea of learning the score of a data distribution rather than the distribution itself is usually referred to as **score matching**. 

One major advantage of this idea is that we do not need to normalize our estimate: If we use a neural network $f(\theta;x)$ to approximate some density $p(x)$, we need to make sure that $\int f(\theta;x)dx=1$. Computing such an integral can be challenging or even intractable. 

For diffusion models, our training objective will be given by \begin{equation}\label{eq3}
    \sum_{i=1}^n\lambda_i\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i) - \nabla_x\log p_{t_i}(x_i)\right\Vert^2\right],
\end{equation} where $(\lambda_i)$ are positive weights summed up to $1$ and determined by the choices of $\sigma$ and $g(t)$.


## Trainable objective

Since $\nabla_xp_t$ is not tractable, we need to convert \eqref{eq3} to a trainable objective. To do this, we note that \begin{align*}
    &\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i) - \nabla_x\log p_{t_i}(x_i)\right\Vert^2\right]\\
    =~&\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i)\right\Vert^2\right] - \mathbb{E}\left[f(\theta;x_i,t_i)\cdot\nabla_x\log p_{t_i}(x_i)\right] + \text{Const}\\
    =~&\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i)\right\Vert^2\right] - \int \left(f(\theta;x_i,t_i)\cdot\nabla_x\log p_{t_i}(x_i) \right)p_{t_i}(x_i)dx_i+ \text{Const}\\
    =~&\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i)\right\Vert^2\right] - \int \left(f(\theta;x_i,t_i)\cdot \frac{1}{p_{t_i}(x_i)}\nabla_x p_{t_i}(x_i) \right)p_{t_i}(x_i)dx_i+ \text{Const}\\
    =~&\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i)\right\Vert^2\right] - \int \left(f(\theta;x_i,t_i)\cdot \nabla_x p_{t_i}(x_i) \right)dx_i+ \text{Const},
\end{align*} where we can use the fact that \begin{align*}
    p_{t_i}(x_i) = \int p_{0,t_i}(x_i|x_0)p_0(x_0)dx_0
\end{align*} to conclude that \begin{equation}
    \mathbb{E}\left[\left\Vert f(\theta;x_i,t_i) - \nabla_x\log p_{t_i}(x_i)\right\Vert^2\right] = \mathbb{E}\left[\left\Vert f(\theta;x_i,t_i) - \nabla_x\log p_{0,t_i}(x_i|x_0)\right\Vert^2\right] \label{eq4} + \text{Const}.
\end{equation} The expection on the right-hand side of \eqref{eq4} is particularly nice as we know $x_i$ conditioned on $x_0$ is normally distributed with mean and variances given explicitly by \eqref{mean_var}. Therefore, we can set our loss function to be \begin{align}
    &\sum_{i=1}^n\lambda_i\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i) - \nabla_x\log p_{0,t_i}(x_i|x_0)\right\Vert^2\right]\\
    =~&\sum_{i=1}^n\lambda_i\mathbb{E}\left[\left\Vert f(\theta;x_i,t_i) + \frac{x_i-m_{0,t_i}(x_0)}{\sigma_{0,t_i}}\right\Vert^2\right] \label{loss}
\end{align}

## Training
To sample training data given the loss function \eqref{loss}, we can first sample an initial data and a random index\begin{equation*}
    x_0\sim p,\quad I \sim \sum_{i=1}^n \lambda_i \delta_i.
\end{equation*} We can then sample $x_I$ using \eqref{mean_var} and compute the gradient for the loss \begin{equation*}
    \left\Vert f(\theta;x_I,t_I) + \frac{x_I-m_{0,t_I}(x_0)}{\sigma_{0,t_I}}\right\Vert^2.
\end{equation*}

Here is a sketch for the training code when sampling from a DataLoader.

```python
for epoch in range(N):
    for x0 in loader:
        t = t_seq[torch.multinomial(pmf)] # pmf for lambda_i
        mean = m(0, t, x0) 
        std = sigma(0, t) # both mean and std computed using g(t)
        xt = torch.distributions.multivariate_normal.
            MultivariateNormal(mean, std*torch.eye(d))
        
        pred_score = model(xt, t)
        loss = F.mse_loss(pred_score, (xt-mean)/std)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}: loss {loss.item():.4f}")
```

## Conditioning
In many practical applications of generative models, instead of simply sampling from the data distribution $p$, we want to generate data given certain properties. For example, in image generation, we want an image that fits a certain text discription. 

Mathematically, this means that we want to learn the conditional distribution $p(\cdot|L)$ where $\ell$ is an additional input. In this case, we can still use our noise injection and denoising but the score $\nabla_x \log p_t(x)$ needs to be replaced with **conditional score** $\nabla_x \log p_t(x|\ell)$. 

Bayes' rule implies that \begin{equation*}
    \nabla_x\log p_t(x|\ell) = \nabla_x\log p_t(x) + \nabla_x \log p_t(\ell|x).
\end{equation*} Hence, suppose we have a good estimate for the score. It is suffice to learn the information of $\nabla_x \log p_t(\ell|x)$. We refer to this method of learning condition score **gradient guidance**.

## Sampling
Once we have trained the network $f(\theta;x,t)\approx \nabla_xp_t(x)$. We can generate new data by sampling $x_n$ from $N(\mu,\sigma^2)$ and then simulate either the SDE \eqref{eq2} or the ODE \begin{equation}
    d\widetilde{x}_t = \left[-\frac{g(T-t)^2}{2\sigma^2}(\mu-\widetilde{x}_t) + \frac{1}{2}g(T-t)^2\nabla_x\log p_{T-t}(\widetilde{x}_t)\right]dt.
\end{equation} The two processes give the same marginal distributions for $x_t$. Emperically, the SDE gives samples of better quality while the ODE benefits in faster convergence.


## References
1. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
2. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR. OpenReview.net, 2021.
3. Conghan Yue, Zhengwei Peng, Junlong Ma, Shiyan Du, Pengxu Wei, and Dongyu Zhang. Image restoration through generalized Ornstein-Uhlenbck bridge. In ICML. OpenReview.net, 2024.