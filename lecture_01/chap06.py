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
# This notebook contains code examples from Chapter 6: Discrete Cosine Transform
#
# Copyright 2015 Allen Downey
#
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)

# %%
# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py

# %%
import numpy as np
PI2 = np.pi * 2

# %% [markdown]
# ### Synthesis
#
# The simplest way to synthesize a mixture of sinusoids is to add up sinusoid signals and evaluate the sum.

# %%
from thinkdsp import CosSignal, SumSignal

def synthesize1(amps, fs, ts):
    components = [CosSignal(freq, amp)
                  for amp, freq in zip(amps, fs)]
    signal = SumSignal(*components)

    ys = signal.evaluate(ts)
    return ys


# %% [markdown]
# Here's an example that's a mixture of 4 components.

# %%
from thinkdsp import Wave

amps = np.array([0.6, 0.25, 0.1, 0.05])
fs = [100, 200, 300, 400]
framerate = 11025

ts = np.linspace(0, 1, framerate, endpoint=False)
ys = synthesize1(amps, fs, ts)
wave = Wave(ys, ts, framerate)
wave.apodize()
wave.make_audio()


# %% [markdown]
# We can express the same process using matrix multiplication.

# %%
def synthesize2(amps, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    ys = np.dot(M, amps)
    return ys


# %% [markdown]
# And it should sound the same.

# %%
ys = synthesize2(amps, fs, ts)
wave = Wave(ys, framerate)
wave.apodize()
wave.make_audio()

# %% [markdown]
# And we can confirm that the differences are small.

# %%
ys1 = synthesize1(amps, fs, ts)
ys2 = synthesize2(amps, fs, ts)
np.max(np.abs(ys1 - ys2))


# %% [markdown]
# ### Analysis
#
# The simplest way to analyze a signal---that is, find the amplitude for each component---is to create the same matrix we used for synthesis and then solve the system of linear equations.

# %%
def analyze1(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.linalg.solve(M, ys)
    return amps


# %% [markdown]
# Using the first 4 values from the wave array, we can recover the amplitudes.

# %%
n = len(fs)
amps2 = analyze1(ys[:n], fs, ts[:n])
amps2

# %% [markdown]
# What we have so far is a simple version of a discrete cosine tranform (DCT), but it is not an efficient implementation because the matrix we get is not orthogonal.

# %%
# suppress scientific notation for small numbers
np.set_printoptions(precision=3, suppress=True)


# %%
def test1():
    N = 4.0
    time_unit = 0.001
    ts = np.arange(N) / N * time_unit
    max_freq = N / time_unit / 2
    fs = np.arange(N) / N * max_freq
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    return M

M = test1()
M

# %% [markdown]
# To check whether a matrix is orthogonal, we can compute $M^T M$, which should be the identity matrix:

# %%
M.transpose().dot(M)


# %% [markdown]
# But it's not, which means that this choice of M is not orthogonal.
#
# Solving a linear system with a general matrix (that is, one that does not have nice properties like orthogonality) takes time proportional to $N^3$.  With an orthogonal matrix, we can get that down to $N^2$.  Here's how:

# %%
def test2():
    N = 4.0
    ts = (0.5 + np.arange(N)) / N
    fs = (0.5 + np.arange(N)) / 2
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    return M
    
M = test2()
M

# %% [markdown]
# Now $M^T M$ is $2I$ (approximately), so M is orthogonal except for a factor of two.

# %%
M.transpose().dot(M)


# %% [markdown]
# And that means we can solve the analysis problem using matrix multiplication.

# %%
def analyze2(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = M.dot(ys) / 2
    return amps


# %% [markdown]
# It works:

# %%
n = len(fs)
amps2 = analyze1(ys[:n], fs, ts[:n])
amps2


# %% [markdown]
# ### DCT
#
# What we've implemented is DCT-IV, which is one of several versions of DCT using orthogonal matrices.

# %%
def dct_iv(ys):
    N = len(ys)
    ts = (0.5 + np.arange(N)) / N
    fs = (0.5 + np.arange(N)) / 2
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.dot(M, ys) / 2
    return amps


# %% [markdown]
# We can check that it works:

# %%
amps = np.array([0.6, 0.25, 0.1, 0.05])
N = 4.0
ts = (0.5 + np.arange(N)) / N
fs = (0.5 + np.arange(N)) / 2
ys = synthesize2(amps, fs, ts)

amps2 = dct_iv(ys)
np.max(np.abs(amps - amps2))


# %% [markdown]
# DCT and inverse DCT are the same thing except for a factor of 2.

# %%
def inverse_dct_iv(amps):
    return dct_iv(amps) * 2


# %% [markdown]
# And it works:

# %%
amps = [0.6, 0.25, 0.1, 0.05]
ys = inverse_dct_iv(amps)
amps2 = dct_iv(ys)
np.max(np.abs(amps - amps2))

# %% [markdown]
# ###  Dct
#
# `thinkdsp` provides a `Dct` class that encapsulates the DCT in the same way the Spectrum class encapsulates the FFT.

# %%
from thinkdsp import TriangleSignal

signal = TriangleSignal(freq=400)
wave = signal.make_wave(duration=1.0, framerate=10000)
wave.make_audio()

# %% [markdown]
# To make a Dct object, you can invoke `make_dct` on a Wave.

# %%
from thinkdsp import decorate

dct = wave.make_dct()
dct.plot()
decorate(xlabel='Frequency (Hz)', ylabel='DCT')

# %% [markdown]
# Dct provides `make_wave`, which performs the inverse DCT.

# %%
wave2 = dct.make_wave()

# %% [markdown]
# The result is very close to the wave we started with.

# %%
np.max(np.abs(wave.ys-wave2.ys))

# %% [markdown]
# Negating the signal changes the sign of the DCT.

# %%
signal = TriangleSignal(freq=400, offset=0)
wave = signal.make_wave(duration=1.0, framerate=10000)
wave.ys *= -1
wave.make_dct().plot()

# %% [markdown]
# Adding phase offset $\phi=\pi$ has the same effect.

# %%
signal = TriangleSignal(freq=400, offset=np.pi)
wave = signal.make_wave(duration=1.0, framerate=10000)
wave.make_dct().plot()

# %% jupyter={"outputs_hidden": true}
