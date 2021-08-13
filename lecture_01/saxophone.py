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
# This notebook contains an example related to Chapter 5: Autocorrelation
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
# ## The case of the missing fundamental
#
# This notebook investigates autocorrelation, pitch perception and a phenomenon called the "missing fundamental".
#
# I'll start with a recording of a saxophone.

# %%
if not os.path.exists('100475__iluppai__saxophone-weep.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/100475__iluppai__saxophone-weep.wav

# %% jupyter={"outputs_hidden": false}
from thinkdsp import read_wave

wave = read_wave('100475__iluppai__saxophone-weep.wav')
wave.normalize()
wave.make_audio()

# %% [markdown]
# The spectrogram shows the harmonic structure over time.

# %% jupyter={"outputs_hidden": false}
gram = wave.make_spectrogram(seg_length=1024)
gram.plot(high=3000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

# %% [markdown]
# To see the harmonics more clearly, I'll take a segment near the 2 second mark and compute its spectrum.

# %% jupyter={"outputs_hidden": false}
start = 2.0
duration = 0.5
segment = wave.segment(start=start, duration=duration)
segment.make_audio()

# %% jupyter={"outputs_hidden": false}
spectrum = segment.make_spectrum()
spectrum.plot(high=3000)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# The peaks in the spectrum are at 1392, 928, and 464 Hz.

# %% jupyter={"outputs_hidden": false}
spectrum.peaks()[:10]

# %% [markdown]
# The pitch we perceive is the fundamental, at 464 Hz, even though it is not the dominant frequency.
#
# For comparison, here's a triangle wave at 464 Hz.

# %% jupyter={"outputs_hidden": false}
from thinkdsp import TriangleSignal

TriangleSignal(freq=464).make_wave(duration=0.5).make_audio()

# %% [markdown]
# And here's the segment again.

# %% jupyter={"outputs_hidden": false}
segment.make_audio()


# %% [markdown]
# They have the same perceived pitch.
#
# To understand why we perceive the fundamental frequency, even though it is not dominant, it helps to look at the autocorrelation function (ACF).
#
# The following function computes the ACF, selects the second half (which corresponds to positive lags) and normalizes the results:

# %% jupyter={"outputs_hidden": false}
def autocorr(segment):
    corrs = np.correlate(segment.ys, segment.ys, mode='same')
    N = len(corrs)
    lengths = range(N, N//2, -1)

    half = corrs[N//2:].copy()
    half /= lengths
    half /= half[0]
    return half


# %% [markdown]
# And here's what the result:

# %% jupyter={"outputs_hidden": false}
corrs = autocorr(segment)
plt.plot(corrs[:200])
decorate(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])


# %% [markdown]
# The first major peak is near lag 100.
#
# The following function finds the highest correlation in a given range of lags and returns the corresponding frequency.

# %% jupyter={"outputs_hidden": false}
def find_frequency(corrs, low, high):
    lag = np.array(corrs[low:high]).argmax() + low
    print(lag)
    period = lag / segment.framerate
    frequency = 1 / period
    return frequency


# %% [markdown]
# The highest peak is at a lag 95, which corresponds to frequency 464 Hz.

# %% jupyter={"outputs_hidden": false}
find_frequency(corrs, 80, 100)

# %% [markdown]
# At least in this example, the pitch we perceive corresponds to the highest peak in the autocorrelation function (ACF) rather than the highest component of the spectrum.

# %% [markdown]
# Surprisingly, the perceived pitch doesn't change if we remove the fundamental completely.  Here's what the spectrum looks like if we use a high-pass filter to clobber the fundamental.

# %% jupyter={"outputs_hidden": false}
spectrum2 = segment.make_spectrum()
spectrum2.high_pass(600)
spectrum2.plot(high=3000)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# And here's what it sounds like.

# %% jupyter={"outputs_hidden": false}
segment2 = spectrum2.make_wave()
segment2.make_audio()

# %% [markdown]
# The perceived pitch is still 464 Hz, even though there is no power at that frequency.  This phenomenon is called the "missing fundamental": https://en.wikipedia.org/wiki/Missing_fundamental

# %% [markdown]
# To understand why we hear a frequency that's not in the signal, it helps to look at the autocorrelation function (ACF).

# %% jupyter={"outputs_hidden": false}
corrs = autocorr(segment2)
plt.plot(corrs[:200])
decorate(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])

# %% [markdown]
# The third peak, which corresponds to 464 Hz, is still the highest:

# %% jupyter={"outputs_hidden": false}
find_frequency(corrs, 80, 100)

# %% [markdown]
# But there are two other peaks corresponding to 1297 Hz and 722 Hz.  

# %% jupyter={"outputs_hidden": false}
find_frequency(corrs, 20, 50)

# %% jupyter={"outputs_hidden": false}
find_frequency(corrs, 50, 80)

# %% [markdown]
# So why don't we perceive either of those pitches, instead of 464 Hz?  The reason is that the higher components that are present in the signal are harmonics of 464 Hz and they are not harmonics of 722 or 1297 Hz.
#
# Our ear interprets the high harmonics as evidence that the "right" fundamental is at 464 Hz.

# %% [markdown]
# If we get rid of the high harmonics, the effect goes away.  Here's a spectrum with harmonics above 1200 Hz removed.

# %% jupyter={"outputs_hidden": false}
spectrum4 = segment.make_spectrum()
spectrum4.high_pass(600)
spectrum4.low_pass(1200)
spectrum4.plot(high=3000)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# Now the perceived pitch is 928 Hz.

# %% jupyter={"outputs_hidden": false}
segment4 = spectrum4.make_wave()
segment4.make_audio()

# %% jupyter={"outputs_hidden": false}
TriangleSignal(freq=928).make_wave(duration=0.5).make_audio()

# %% [markdown]
# And if we look at the autocorrelation function, we find the highest peak at lag=47, which corresponds to 938 Hz.

# %% jupyter={"outputs_hidden": false}
corrs = autocorr(segment4)
plt.plot(corrs[:200])
decorate(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])

# %% jupyter={"outputs_hidden": false}
find_frequency(corrs, 30, 50)

# %% [markdown]
# For convenience, here are all the versions together.
#
# A triangle signal at 464 Hz:

# %% jupyter={"outputs_hidden": false}
TriangleSignal(freq=464).make_wave(duration=0.5).make_audio()

# %% [markdown]
# The original segment:

# %% jupyter={"outputs_hidden": false}
segment.make_audio()

# %% [markdown]
# After removing the fundamental:

# %% jupyter={"outputs_hidden": false}
segment2.make_audio()

# %% [markdown]
# After removing the harmonics above the dominant frequency, too.

# %% jupyter={"outputs_hidden": false}
segment4.make_audio()

# %% [markdown]
# And a pure sinusoid:

# %% jupyter={"outputs_hidden": false}
from thinkdsp import SinSignal

SinSignal(freq=928).make_wave(duration=0.5).make_audio()

# %% [markdown]
# In summary, these experiments suggest that pitch perception is not based entirely on spectral analysis, but is also informed by something like autocorrelation.

# %% jupyter={"outputs_hidden": true}
