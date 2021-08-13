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
# This notebook contains solutions to exercises in Chapter 6: Discrete Cosine Transform
#
# Copyright 2015 Allen Downey
#
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)

# %% jupyter={"outputs_hidden": false}
# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py

# %%
import numpy as np
import matplotlib.pyplot as plt

from thinkdsp import decorate

# %% [markdown]
# ## Exercise 1
#
# In this chapter I claim that `analyze1` takes time proportional
# to $n^3$ and `analyze2` takes time proportional to $n^2$.  To
# see if that's true, run them on a range of input sizes and time
# them.  In IPython, you can use the magic command `%timeit`.
#
# If you plot run time versus input size on a log-log scale, you
# should get a straight line with slope 3 for  `analyze1` and
# slope 2 for `analyze2`.  You also might want to test `dct_iv`
# and `scipy.fftpack.dct`.
#
# I'll start with a noise signal and an array of power-of-two sizes

# %% jupyter={"outputs_hidden": false}
from thinkdsp import UncorrelatedGaussianNoise

signal = UncorrelatedGaussianNoise()
noise = signal.make_wave(duration=1.0, framerate=16384)
noise.ys.shape

# %% [markdown]
# The following function takes an array of results from a timing experiment, plots the results, and fits a straight line.

# %%
from scipy.stats import linregress

loglog = dict(xscale='log', yscale='log')

def plot_bests(ns, bests):    
    plt.plot(ns, bests)
    decorate(**loglog)
    
    x = np.log(ns)
    y = np.log(bests)
    t = linregress(x,y)
    slope = t[0]

    return slope


# %%
PI2 = np.pi * 2

def analyze1(ys, fs, ts):
    """Analyze a mixture of cosines and return amplitudes.

    Works for the general case where M is not orthogonal.

    ys: wave array
    fs: frequencies in Hz
    ts: times where the signal was evaluated    

    returns: vector of amplitudes
    """
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.linalg.solve(M, ys)
    return amps


# %% jupyter={"outputs_hidden": false}
def run_speed_test(ns, func):
    results = []
    for N in ns:
        print(N)
        ts = (0.5 + np.arange(N)) / N
        freqs = (0.5 + np.arange(N)) / 2
        ys = noise.ys[:N]
        # result = %timeit -r1 -o func(ys, freqs, ts)
        results.append(result)
        
    bests = [result.best for result in results]
    return bests


# %% [markdown]
# Here are the results for `analyze1`.

# %% jupyter={"outputs_hidden": false}
ns = 2 ** np.arange(6, 13)
ns

# %% jupyter={"outputs_hidden": false}
bests = run_speed_test(ns, analyze1)
plot_bests(ns, bests)


# %% [markdown]
# The estimated slope is close to 2, not 3, as expected.  One possibility is that the performance of `np.linalg.solve` is nearly quadratic in this range of array sizes.
#
# Here are the results for `analyze2`:

# %%
def analyze2(ys, fs, ts):
    """Analyze a mixture of cosines and return amplitudes.

    Assumes that fs and ts are chosen so that M is orthogonal.

    ys: wave array
    fs: frequencies in Hz
    ts: times where the signal was evaluated    

    returns: vector of amplitudes
    """
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.dot(M, ys) / 2
    return amps


# %% jupyter={"outputs_hidden": false}
bests2 = run_speed_test(ns, analyze2)
plot_bests(ns, bests2)

# %% [markdown]
# The results for `analyze2` fall in a straight line with the estimated slope close to 2, as expected.
#
# Here are the results for the `scipy.fftpack.dct`

# %% jupyter={"outputs_hidden": false}
import scipy.fftpack

def scipy_dct(ys, freqs, ts):
    return scipy.fftpack.dct(ys, type=3)


# %% jupyter={"outputs_hidden": false}
bests3 = run_speed_test(ns, scipy_dct)
plot_bests(ns, bests3)

# %% [markdown]
# This implementation of dct is even faster.  The line is curved, which means either we haven't seen the asymptotic behavior yet, or the asymptotic behavior is not a simple exponent of $n$.  In fact, as we'll see soon, the run time is proportional to $n \log n$.
#
# The following figure shows all three curves on the same axes.

# %% jupyter={"outputs_hidden": false}
plt.plot(ns, bests, label='analyze1')
plt.plot(ns, bests2, label='analyze2')
plt.plot(ns, bests3, label='fftpack.dct')
decorate(xlabel='Wave length (N)', ylabel='Time (s)', **loglog)

# %% [markdown]
# ## Exercise 2
#
# One of the major applications of the DCT is compression for both sound and images. In its simplest form, DCT-based compression works like this:
#
# 1. Break a long signal into segments.
# 2. Compute the DCT of each segment.
# 3. Identify frequency components with amplitudes so low they are inaudible, and remove them. Store only the frequencies and amplitudes that remain.
# 4. To play back the signal, load the frequencies and amplitudes for each segment and apply the inverse DCT.
#
# Implement a version of this algorithm and apply it to a recording of music or speech. How many components can you eliminate before the difference is perceptible?

# %% [markdown]
# `thinkdsp` provides a class, `Dct` that is similar to a `Spectrum`, but which uses DCT instead of FFT.

# %% [markdown]
# As an example, I'll use a recording of a saxophone:

# %%
if not os.path.exists('100475__iluppai__saxophone-weep.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/100475__iluppai__saxophone-weep.wav

# %% jupyter={"outputs_hidden": false}
from thinkdsp import read_wave

wave = read_wave('100475__iluppai__saxophone-weep.wav')
wave.make_audio()

# %% [markdown]
# Here's a short segment:

# %% jupyter={"outputs_hidden": false}
segment = wave.segment(start=1.2, duration=0.5)
segment.normalize()
segment.make_audio()

# %% [markdown]
# And here's the DCT of that segment:

# %% jupyter={"outputs_hidden": false}
seg_dct = segment.make_dct()
seg_dct.plot(high=4000)
decorate(xlabel='Frequency (Hz)', ylabel='DCT')


# %% [markdown]
# There are only a few harmonics with substantial amplitude, and many entries near zero.
#
# The following function takes a DCT and sets elements below `thresh` to 0.

# %% jupyter={"outputs_hidden": false}
def compress(dct, thresh=1):
    count = 0
    for i, amp in enumerate(dct.amps):
        if np.abs(amp) < thresh:
            dct.hs[i] = 0
            count += 1
            
    n = len(dct.amps)
    print(count, n, 100 * count / n, sep='\t')


# %% [markdown]
# If we apply it to the segment, we can eliminate more than 90% of the elements:

# %% jupyter={"outputs_hidden": false}
seg_dct = segment.make_dct()
compress(seg_dct, thresh=10)
seg_dct.plot(high=4000)

# %% [markdown]
# And the result sounds the same (at least to me):

# %% jupyter={"outputs_hidden": false}
seg2 = seg_dct.make_wave()
seg2.make_audio()

# %% [markdown]
# To compress a longer segment, we can make a DCT spectrogram.  The following function is similar to `wave.make_spectrogram` except that it uses the DCT.

# %% jupyter={"outputs_hidden": false}
from thinkdsp import Spectrogram

def make_dct_spectrogram(wave, seg_length):
    """Computes the DCT spectrogram of the wave.

    seg_length: number of samples in each segment

    returns: Spectrogram
    """
    window = np.hamming(seg_length)
    i, j = 0, seg_length
    step = seg_length // 2

    # map from time to Spectrum
    spec_map = {}

    while j < len(wave.ys):
        segment = wave.slice(i, j)
        segment.window(window)

        # the nominal time for this segment is the midpoint
        t = (segment.start + segment.end) / 2
        spec_map[t] = segment.make_dct()

        i += step
        j += step

    return Spectrogram(spec_map, seg_length)


# %% [markdown]
# Now we can make a DCT spectrogram and apply `compress` to each segment:

# %% jupyter={"outputs_hidden": false}
spectro = make_dct_spectrogram(wave, seg_length=1024)
for t, dct in sorted(spectro.spec_map.items()):
    compress(dct, thresh=0.2)

# %% [markdown]
# In most segments, the compression is 75-85%.
#
# To hear what it sounds like, we can convert the spectrogram back to a wave and play it.

# %% jupyter={"outputs_hidden": false}
wave2 = spectro.make_wave()
wave2.make_audio()

# %% [markdown]
# And here's the original again for comparison.

# %% jupyter={"outputs_hidden": false}
wave.make_audio()

# %% [markdown]
# As an experiment, you might try increasing `thresh` to see when the effect of compression becomes audible (to you).
#
# Also, you might try compressing a signal with some noisy elements, like cymbals.

# %% jupyter={"outputs_hidden": true}
