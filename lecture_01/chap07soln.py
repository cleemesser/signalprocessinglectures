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
# ## ThinkDSP
#
# This notebook contains solutions to exercises in Chapter 7: Discrete Fourier Transform
#
# Copyright 2015 Allen Downey
#
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)

# %%
import numpy as np
PI2 = 2 * np.pi

# %% [markdown]
# ## Exercise 1
#
# In this chapter, I showed how we can express the DFT and inverse DFT
# as matrix multiplications.  These operations take time proportional to
# $N^2$, where $N$ is the length of the wave array.  That is fast enough
# for many applications, but there is a faster
# algorithm, the Fast Fourier Transform (FFT), which takes time
# proportional to $N \log N$.
#
# The key to the FFT is the Danielson-Lanczos lemma:
#
# $DFT(y)[n] = DFT(e)[n] + \exp(-2 \pi i n / N) DFT(o)[n]$
#
# Where $ DFT(y)[n]$ is the $n$th element of the DFT of $y$; $e$ is the even elements of $y$, and $o$ is the odd elements of $y$.
#
# This lemma suggests a recursive algorithm for the DFT:
#
# 1. Given a wave array, $y$, split it into its even elements, $e$, and its odd elements, $o$.
#
# 2. Compute the DFT of $e$ and $o$ by making recursive calls.
#
# 3. Compute $DFT(y)$ for each value of $n$ using the Danielson-Lanczos lemma.
#
# For the base case of this recursion, you could wait until the length
# of $y$ is 1.  In that case, $DFT(y) = y$.  Or if the length of $y$
# is sufficiently small, you could compute its DFT by matrix multiplication,
# possibly using a precomputed matrix.
#
# Hint: I suggest you implement this algorithm incrementally by starting
# with a version that is not truly recursive.  In Step 2, instead of
# making a recursive call, use `dft` or `np.fft.fft`.  Get Step 3 working,
# and confirm that the results are consistent with the other
# implementations.  Then add a base case and confirm that it works.
# Finally, replace Step 2 with recursive calls.
#
# One more hint: Remember that the DFT is periodic; you might find `np.tile` useful.
#
# You can read more about the FFT at https://en.wikipedia.org/wiki/Fast_Fourier_transform.

# %% [markdown]
# As the test case, I'll start with a small real signal and compute its FFT:

# %% jupyter={"outputs_hidden": false}
ys = [-0.5, 0.1, 0.7, -0.1]
hs = np.fft.fft(ys)
print(hs)


# %% [markdown]
# Here's my implementation of DFT from the book:

# %% jupyter={"outputs_hidden": false}
def dft(ys):
    N = len(ys)
    ts = np.arange(N) / N
    freqs = np.arange(N)
    args = np.outer(ts, freqs)
    M = np.exp(1j * PI2 * args)
    amps = M.conj().transpose().dot(ys)
    return amps


# %% [markdown]
# We can confirm that this implementation gets the same result.

# %% jupyter={"outputs_hidden": false}
hs2 = dft(ys)
np.sum(np.abs(hs - hs2))


# %% [markdown]
# As a step toward making a recursive FFT, I'll start with a version that splits the input array and uses np.fft.fft to compute the FFT of the halves.

# %% jupyter={"outputs_hidden": false}
def fft_norec(ys):
    N = len(ys)
    He = np.fft.fft(ys[::2])
    Ho = np.fft.fft(ys[1::2])
    
    ns = np.arange(N)
    W = np.exp(-1j * PI2 * ns / N)
    
    return np.tile(He, 2) + W * np.tile(Ho, 2)


# %% [markdown]
# And we get the same results:

# %% jupyter={"outputs_hidden": false}
hs3 = fft_norec(ys)
np.sum(np.abs(hs - hs3))


# %% [markdown]
# Finally, we can replace `np.fft.fft` with recursive calls, and add a base case:

# %% jupyter={"outputs_hidden": false}
def fft(ys):
    N = len(ys)
    if N == 1:
        return ys
    
    He = fft(ys[::2])
    Ho = fft(ys[1::2])
    
    ns = np.arange(N)
    W = np.exp(-1j * PI2 * ns / N)
    
    return np.tile(He, 2) + W * np.tile(Ho, 2)


# %% [markdown]
# And we get the same results:

# %% jupyter={"outputs_hidden": false}
hs4 = fft(ys)
np.sum(np.abs(hs - hs4))

# %% [markdown]
# This implementation of FFT takes time proportional to $n \log n$.  It also takes space proportional to $n \log n$, and it wastes some time making and copying arrays.  It can be improved to run "in place"; in that case, it requires no additional space, and spends less time on overhead.

# %% jupyter={"outputs_hidden": true}
