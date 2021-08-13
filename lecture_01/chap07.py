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
# This notebook contains code examples from Chapter 7: Discrete Fourier Transform
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
import matplotlib.pyplot as plt

from thinkdsp import decorate
PI2 = 2 * np.pi

# %%
# suppress scientific notation for small numbers
np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## Complex sinusoid
#
# Here's the definition of ComplexSinusoid, with print statements to display intermediate results.

# %%
from thinkdsp import Sinusoid

class ComplexSinusoid(Sinusoid):
    """Represents a complex exponential signal."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        print(ts)
        phases = PI2 * self.freq * ts + self.offset
        print(phases)
        ys = self.amp * np.exp(1j * phases)
        return ys


# %% [markdown]
# Here's an example:

# %%
signal = ComplexSinusoid(freq=1, amp=0.6, offset=1)
wave = signal.make_wave(duration=1, framerate=4)
print(wave.ys)

# %% [markdown]
# The simplest way to synthesize a mixture of signals is to evaluate the signals and add them up.

# %%
from thinkdsp import SumSignal

def synthesize1(amps, freqs, ts):
    components = [ComplexSinusoid(freq, amp)
                  for amp, freq in zip(amps, freqs)]
    signal = SumSignal(*components)
    ys = signal.evaluate(ts)
    return ys


# %% [markdown]
# Here's an example that's a mixture of 4 components.

# %%
amps = np.array([0.6, 0.25, 0.1, 0.05])
freqs = [100, 200, 300, 400]
framerate = 11025

ts = np.linspace(0, 1, framerate, endpoint=False)
ys = synthesize1(amps, freqs, ts)
print(ys)

# %% [markdown]
# Now we can plot the real and imaginary parts:

# %%
n = 500
plt.plot(ts[:n], ys[:n].real)
plt.plot(ts[:n], ys[:n].imag)
decorate(xlabel='Time')

# %% [markdown]
# The real part is a mixture of cosines; the imaginary part is a mixture of sines.  They contain the same frequency components with the same amplitudes, so they sound the same to us:

# %%
from thinkdsp import Wave

wave = Wave(ys.real, framerate)
wave.apodize()
wave.make_audio()

# %%
wave = Wave(ys.imag, framerate)
wave.apodize()
wave.make_audio()


# %% [markdown]
# We can express the same process using matrix multiplication.

# %%
def synthesize2(amps, freqs, ts):
    args = np.outer(ts, freqs)
    M = np.exp(1j * PI2 * args)
    ys = np.dot(M, amps)
    return ys


# %% [markdown]
# And it should sound the same.

# %%
amps = np.array([0.6, 0.25, 0.1, 0.05])
ys = synthesize2(amps, freqs, ts)
print(ys)

# %%
wave = Wave(ys.real, framerate)
wave.apodize()
wave.make_audio()

# %% [markdown]
# To see the effect of a complex amplitude, we can rotate the amplitudes by 1.5 radian:

# %%
phi = 1.5
amps2 = amps * np.exp(1j * phi)
ys2 = synthesize2(amps2, freqs, ts)

n = 500
plt.plot(ts[:n], ys.real[:n], label=r'$\phi_0 = 0$')
plt.plot(ts[:n], ys2.real[:n], label=r'$\phi_0 = 1.5$')
decorate(xlabel='Time')


# %% [markdown]
# Rotating all components by the same phase offset changes the shape of the waveform because the components have different periods, so the same offset has a different effect on each component.

# %% [markdown]
# ### Analysis
#
# The simplest way to analyze a signal---that is, find the amplitude for each component---is to create the same matrix we used for synthesis and then solve the system of linear equations.

# %%
def analyze1(ys, freqs, ts):
    args = np.outer(ts, freqs)
    M = np.exp(1j * PI2 * args)
    amps = np.linalg.solve(M, ys)
    return amps


# %% [markdown]
# Using the first 4 values from the wave array, we can recover the amplitudes.

# %%
n = len(freqs)
amps2 = analyze1(ys[:n], freqs, ts[:n])
print(amps2)

# %% [markdown]
# If we define the `freqs` from 0 to N-1 and `ts` from 0 to (N-1)/N, we get a unitary matrix. 

# %%
N = 4
ts = np.arange(N) / N
freqs = np.arange(N)
args = np.outer(ts, freqs)
M = np.exp(1j * PI2 * args)
print(M)

# %% [markdown]
# To check whether a matrix is unitary, we can compute $M^* M$, which should be the identity matrix:

# %%
MstarM = M.conj().transpose().dot(M)
print(MstarM.real)


# %% [markdown]
# The result is actually $4 I$, so in general we have an extra factor of $N$ to deal with, but that's a minor problem.
#
# We can use this result to write a faster version of `analyze1`:
#

# %%
def analyze2(ys, freqs, ts):
    args = np.outer(ts, freqs)
    M = np.exp(1j * PI2 * args)
    amps = M.conj().transpose().dot(ys) / N
    return amps


# %%
N = 4
amps = np.array([0.6, 0.25, 0.1, 0.05])
freqs = np.arange(N)
ts = np.arange(N) / N
ys = synthesize2(amps, freqs, ts)

amps3 = analyze2(ys, freqs, ts)
print(amps3)


# %% [markdown]
# Now we can write our own version of DFT:

# %%
def synthesis_matrix(N):
    ts = np.arange(N) / N
    freqs = np.arange(N)
    args = np.outer(ts, freqs)
    M = np.exp(1j * PI2 * args)
    return M


# %%
def dft(ys):
    N = len(ys)
    M = synthesis_matrix(N)
    amps = M.conj().transpose().dot(ys)
    return amps


# %% [markdown]
# And compare it to analyze2:

# %%
print(dft(ys))

# %% [markdown]
# The result is close to `amps * 4`.
#
# We can also compare it to `np.fft.fft`.  FFT stands for Fast Fourier Transform, which is an even faster implementation of DFT.

# %%
print(np.fft.fft(ys))


# %% [markdown]
# The inverse DFT is almost the same, except we don't have to transpose $M$ and we have to divide through by $N$.

# %%
def idft(ys):
    N = len(ys)
    M = synthesis_matrix(N)
    amps = M.dot(ys) / N
    return amps


# %% [markdown]
# We can confirm that `dft(idft(amps))` yields `amps`:

# %%
ys = idft(amps)
print(dft(ys))

# %% [markdown]
# ### Real signals
#
# Let's see what happens when we apply DFT to a real-valued signal.

# %%
from thinkdsp import SawtoothSignal

framerate = 10000
signal = SawtoothSignal(freq=500)
wave = signal.make_wave(duration=0.1, framerate=framerate)
wave.make_audio()

# %% [markdown]
# `wave` is a 500 Hz sawtooth signal sampled at 10 kHz.

# %%
hs = dft(wave.ys)
len(wave.ys), len(hs)

# %% [markdown]
# `hs` is the DFT of this wave, and `amps` contains the amplitudes.

# %%
amps = np.abs(hs)
plt.plot(amps)
decorate(xlabel='Frequency (unspecified units)', ylabel='DFT')

# %% [markdown]
# The DFT assumes that the sampling rate is N per time unit, for an arbitrary time unit.  We have to convert to actual units -- seconds -- like this:

# %%
N = len(hs)
fs = np.arange(N) * framerate / N

# %% [markdown]
# Also, the DFT of a real signal is symmetric, so the right side is redundant.  Normally, we only compute and plot the first half:

# %%
plt.plot(fs[:N//2+1], amps[:N//2+1])
decorate(xlabel='Frequency (Hz)', ylabel='DFT')

# %% [markdown]
# Let's get a better sense for why the DFT of a real signal is symmetric.  I'll start by making the inverse DFT matrix for $N=8$.

# %%
M = synthesis_matrix(N=8)

# %% [markdown]
# And the DFT matrix:

# %%
Mstar = M.conj().transpose()

# %% [markdown]
# And a triangle wave with 8 elements:

# %%
from thinkdsp import TriangleSignal

wave = TriangleSignal(freq=1).make_wave(duration=1, framerate=8)
wave.ys

# %% [markdown]
# Here's what the wave looks like.

# %%
wave.plot()

# %% [markdown]
# Now let's look at rows 3 and 5 of the DFT matrix:

# %%
row3 = Mstar[3, :]
print(row3)

# %%
row5 = Mstar[5, :]
row5


# %% [markdown]
# They are almost the same, but row5 is the complex conjugate of row3.

# %%
def approx_equal(a, b, tol=1e-10):
    return np.sum(np.abs(a-b)) < tol


# %%
approx_equal(row3, row5.conj())

# %% [markdown]
# When we multiply the DFT matrix and the wave array, the element with index 3 is:

# %%
X3 = row3.dot(wave.ys)
X3

# %% [markdown]
# And the element with index 5 is:

# %%
X5 = row5.dot(wave.ys)
X5

# %% [markdown]
# And they are the same, within floating point error.

# %%
abs(X3 - X5)

# %% [markdown]
# Let's try the same thing with a complex signal:

# %%
wave2 = ComplexSinusoid(freq=1).make_wave(duration=1, framerate=8)
plt.plot(wave2.ts, wave2.ys.real)
plt.plot(wave2.ts, wave2.ys.imag)

# %% [markdown]
# Now the elements with indices 3 and 5 are different:

# %%
X3 = row3.dot(wave2.ys)
X3

# %%
X5 = row5.dot(wave2.ys)
X5

# %% [markdown]
# Visually we can confirm that the FFT of the real signal is symmetric:

# %%
hs = np.fft.fft(wave.ys)
plt.plot(abs(hs))

# %% [markdown]
# And the FFT of the complex signal is not.

# %%
hs = np.fft.fft(wave2.ys)
plt.plot(abs(hs))

# %% [markdown]
# Another way to think about all of this is to evaluate the DFT matrix for different frequencies.  Instead of $0$ through $N-1$, let's try $0, 1, 2, 3, 4, -3, -2, -1$.

# %%
N = 8
ts = np.arange(N) / N
freqs = np.arange(N)
freqs = [0, 1, 2, 3, 4, -3, -2, -1]
args = np.outer(ts, freqs)
M2 = np.exp(1j * PI2 * args)

# %%
approx_equal(M, M2)

# %% [markdown]
# So you can think of the second half of the DFT as positive frequencies that get aliased (which is how I explained them), or as negative frequencies (which is the more conventional way to explain them).  But the DFT doesn't care either way.
#
# The `thinkdsp` library provides support for computing the "full" FFT instead of the real FFT.

# %%
framerate = 10000
signal = SawtoothSignal(freq=500)
wave = signal.make_wave(duration=0.1, framerate=framerate)

# %%
spectrum = wave.make_spectrum(full=True)

# %%
spectrum.plot()
decorate(xlabel='Frequency (Hz)', ylabel='DFT')

# %%
