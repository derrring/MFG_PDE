# Mathematical Background for Mean Field Games (MFGs)

This document provides the mathematical foundation for the numerical schemes implemented in the MFG_PDE package.

## Hamilton-Jacobi-Bellman (HJB) equation

HJB equation is a fully nonlinear PDE with form

$$
\begin{cases}
    \partial_t u + H(x, u, Du, D^2u) = 0, & x \in \Omega, t \in [0, T] \\
    u(x, T) = g(x), & x \in \Omega \\
    u(x, t) = \phi(x, t), & x \in \partial \Omega, t \in [0, T]
\end{cases}
$$

where $H$ is the Hamiltonian, $g$ is the terminal condition, $\phi$ is the boundary condition.

## Fokker-Planck (FP) equation

Suppose that the dynamic of a stochastic process is governed by the following stochastic differential equation (SDE)

$$
dX_t = b(X_t,t) dt + \sigma(X_t,t) dW_t
$$

where $W_t$ is a $d$-dimensional Brownian motion. The probability density function (PDF) of $X_t$, denoted as $p(x,t)$ satisfies the following Fokker-Planck equation (parabolic PDE)

$$
\begin{cases}   
    \partial_t p + \nabla \cdot (b(x,t) p) = \nabla \cdot (\sigma(x) \nabla p), & x \in \Omega, t \in [0, T] \\
    p(x, 0) = p_0(x), & x \in \Omega \\
    p(x, t) = \phi(x, t), & x \in \partial \Omega, t \in [0, T]
\end{cases}
$$

with drift term $b = (b_1,b_2,\ldots,b_n)$ and diffusion term $\sigma = (\sigma_1,\sigma_2,\ldots,\sigma_n)$, where the diffusion tensor $D=\frac{1}{2}\sigma\sigma^T$, i.e.

$$
D_{ij}(x,t) = \frac{1}{2} \sum_{k=1}^d \sigma_k(x,t) \sigma_k(x,t)
$$

By introducing compact notations:

$$
\mathrm{div} F = \sum_{i=1}^d \frac{\partial F_i}{\partial x_i} = \mathrm{tr} (\nabla F) = \nabla \cdot F
$$

we rewrite the Fokker-Planck equation as

$$
\partial_t p + \sum_{i=1}^d \frac{\partial}{\partial x_i} (b_i(x,t) p(x,t)) = \sum_{i,j=1}^d \frac{\partial^2}{\partial x_i \partial x_j} (D_{ij}(x,t) p(x,t))
$$

$$
\partial_t p + \mathrm{div} (b(x,t)\cdot p(x,t)) = \sigma \Delta p, \quad x \in \Omega, t \in [0, T] 
$$

where $\Delta$ denotes the Laplace operator.
