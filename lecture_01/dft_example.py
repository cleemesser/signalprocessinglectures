# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Implementing DFT
#
# Copyright 2019 Allen Downey, [MIT License](http://opensource.org/licenses/MIT)

# %%
import numpy as np

# %% [markdown]
# Let's start with a known result.  The DFT of an impulse is a constant.

# %%
N = 4
x = [1, 0, 0, 0]

# %%
np.fft.fft(x)

# %% [markdown]
# ### Literal translation
#
# The usual way the DFT is expressed is as a summation.  Following the notation on [Wikipedia](https://en.wikipedia.org/wiki/Discrete_Fourier_transform):
#
# $ X_k = \sum_{n=0}^N x_n \cdot e^{-2 \pi i n k / N} $
#
# Here's a straightforward translation of that formula into Python.

# %%
pi = np.pi
exp = np.exp

# %%
k = 0
sum(x[n] * exp(-2j * pi * k * n / N) for n in range(N))


# %% [markdown]
# Wrapping this code in a function makes the roles of `k` and `n` clearer, where `k` is a free parameter and `n` is the bound variable of the summation.

# %%
def dft_k(x, k):
    return sum(x[n] * exp(-2j * pi * k * n / N) for n in range(N))


# %% [markdown]
# Of course, we usually we compute $X$ all at once, so we can wrap this function in another function:

# %%
def dft(x):
    N = len(x)
    X = [dft_k(x, k) for k in range(N)]
    return X


# %%
dft(x)


# %% [markdown]
# And the results check out.

# %% [markdown]
# ### DFT as a matrix operation
#
# It is also common to express the DFT as a [matrix operation](https://en.wikipedia.org/wiki/DFT_matrix):
#
# $ X = W x $
#
# with 
#
# $ W_{j, k} = \omega^{j k} $
#
# and
#
# $ \omega = e^{-2 \pi i / N}$
#
# If we recognize the construction of $W$ as an outer product, we can use `np.outer` to compute it.
#
# Here's an implementation of DFT using outer product to construct the DFT matrix, and dot product to compute the DFT.

# %%
def dft(x):
    N = len(x)
    ks = range(N)
    args = -2j * pi * np.outer(ks, ks) / N
    W = exp(args)
    X = W.dot(x)
    return X


# %% [markdown]
# And the results check out.

# %%
dft(x)


# %% [markdown]
# ### Implementing FFT
#
# Finally, we can implement the FFT by translating from math notation:
#
# $ X_k = E_k + e^{-2 \pi i k / N} O_k $
#
# Where $E$ is the FFT of the even elements of $x$ and $O$ is the DFT of the odd elements of $x$.
#
# Here's what that looks like in code.

# %%
def fft(x):
    N = len(x)
    if N == 1:
        return x
    
    E = fft(x[::2])
    O = fft(x[1::2])
    
    ks = np.arange(N)
    args = -2j * pi * ks / N
    W = np.exp(args)
    
    return np.tile(E, 2) + W * np.tile(O, 2)


# %% [markdown]
# The length of $E$ and $O$ is half the length of $W$, so I use `np.tile` to double them up.
#
# And the results check out.

# %%
fft(x)

# %%
