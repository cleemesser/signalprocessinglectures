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
# This notebook contains solutions to exercises in Chapter 10: Signals and Systems
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

# %% [markdown]
# ## Exercise 1
#
# In this chapter I describe convolution as the sum of shifted,
# scaled copies of a signal.  Strictly speaking, this operation is
# *linear* convolution, which does not assume that the signal
# is periodic.
#
# But when we multiply the
# DFT of the signal by the transfer function, that operation corresponds
# to *circular* convolution, which assumes that the signal is
# periodic.  As a result, you might notice that the output contains
# an extra note at the beginning, which wraps around from the end.
#
# Fortunately, there is a standard solution to this problem.  If you
# add enough zeros to the end of the signal before computing the DFT,
# you can avoid wrap-around and compute a linear convolution.
#
# Modify the example in `chap10soln.ipynb` and confirm that zero-padding
# eliminates the extra note at the beginning of the output.

# %% [markdown]
# *Solution:* I'll truncate both signals to $2^{16}$ elements, then zero-pad them to $2^{17}$.  Using powers of two makes the FFT algorithm most efficient.
#
# Here's the impulse response:

# %%
if not os.path.exists('180960__kleeb__gunshot.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/180960__kleeb__gunshot.wav

# %% jupyter={"outputs_hidden": false}
from thinkdsp import read_wave

response = read_wave('180960__kleeb__gunshot.wav')

start = 0.12
response = response.segment(start=start)
response.shift(-start)

response.truncate(2**16)
response.zero_pad(2**17)

response.normalize()
response.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# And its spectrum:

# %% jupyter={"outputs_hidden": false}
transfer = response.make_spectrum()
transfer.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# Here's the signal:

# %%
if not os.path.exists('92002__jcveliz__violin-origional.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/92002__jcveliz__violin-origional.wav

# %% jupyter={"outputs_hidden": false}
violin = read_wave('92002__jcveliz__violin-origional.wav')

start = 0.11
violin = violin.segment(start=start)
violin.shift(-start)

violin.truncate(2**16)
violin.zero_pad(2**17)

violin.normalize()
violin.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# And its spectrum:

# %%
spectrum = violin.make_spectrum()

# %% [markdown]
# Now we can multiply the DFT of the signal by the transfer function, and convert back to a wave:

# %%
output = (spectrum * transfer).make_wave()
output.normalize()

# %% [markdown]
# The result doesn't look like it wraps around:

# %% jupyter={"outputs_hidden": false}
output.plot()

# %% [markdown]
# And we don't hear the extra note at the beginning:

# %% jupyter={"outputs_hidden": false}
output.make_audio()

# %% [markdown]
# We should get the same results from `np.convolve` and `scipy.signal.fftconvolve`.
#
# First I'll get rid of the zero padding:

# %% jupyter={"outputs_hidden": false}
response.truncate(2**16)
response.plot()

# %% jupyter={"outputs_hidden": false}
violin.truncate(2**16)
violin.plot()

# %% [markdown]
# Now we can compare to `np.convolve`:

# %%
output2 = violin.convolve(response)

# %% [markdown]
# The results are similar:

# %% jupyter={"outputs_hidden": false}
output2.plot()

# %% [markdown]
# And sound the same:

# %% jupyter={"outputs_hidden": false}
output2.make_audio()

# %% [markdown]
# But the results are not exactly the same length:

# %% jupyter={"outputs_hidden": false}
len(output), len(output2)

# %% [markdown]
# `scipy.signal.fftconvolve` does the same thing, but as the name suggests, it uses the FFT, so it is substantially faster:

# %% jupyter={"outputs_hidden": false}
from thinkdsp import Wave

import scipy.signal
ys = scipy.signal.fftconvolve(violin.ys, response.ys)
output3 = Wave(ys, framerate=violin.framerate)

# %% [markdown]
# The results look the same.

# %% jupyter={"outputs_hidden": false}
output3.plot()

# %% [markdown]
# And sound the same:

# %% jupyter={"outputs_hidden": false}
output3.make_audio()

# %% [markdown]
# And within floating point error, they are the same:

# %% jupyter={"outputs_hidden": false}
output2.max_diff(output3)

# %% [markdown]
# ## Exercise 2  
#
# The Open AIR library provides a "centralized... on-line resource for
# anyone interested in auralization and acoustical impulse response
# data" (http://www.openairlib.net).  Browse their collection
# of impulse response data and download one that sounds interesting.
# Find a short recording that has the same sample rate as the impulse
# response you downloaded.
#
# Simulate the sound of your recording in the space where the impulse
# response was measured, computed two way: by convolving the recording
# with the impulse response and by computing the filter that corresponds
# to the impulse response and multiplying by the DFT of the recording.

# %% [markdown]
# *Solution:* I downloaded the impulse response of the Lady Chapel at St Albans Cathedral http://www.openairlib.net/auralizationdb/content/lady-chapel-st-albans-cathedral
#
# Thanks to Audiolab, University of York: Marcin Gorzel, Gavin Kearney, Aglaia Foteinou, Sorrel Hoare, Simon Shelley.
#

# %%
if not os.path.exists('stalbans_a_mono.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/stalbans_a_mono.wav

# %% jupyter={"outputs_hidden": false}
response = read_wave('stalbans_a_mono.wav')

start = 0
duration = 5
response = response.segment(duration=duration)
response.shift(-start)

response.normalize()
response.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Here's what it sounds like:

# %% jupyter={"outputs_hidden": false}
response.make_audio()

# %% [markdown]
# The DFT of the impulse response is the transfer function:

# %% jupyter={"outputs_hidden": false}
transfer = response.make_spectrum()
transfer.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# Here's the transfer function on a log-log scale:

# %% jupyter={"outputs_hidden": false}
transfer.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude',
         xscale='log', yscale='log')

# %% [markdown]
# Now we can simulate what a recording would sound like if it were played in the same room and recorded in the same way.  Here's the trumpet recording we have used before:

# %%
if not os.path.exists('170255__dublie__trumpet.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/170255__dublie__trumpet.wav

# %% jupyter={"outputs_hidden": false}
wave = read_wave('170255__dublie__trumpet.wav')

start = 0.0
wave = wave.segment(start=start)
wave.shift(-start)

wave.truncate(len(response))
wave.normalize()
wave.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Here's what it sounds like before transformation:

# %% jupyter={"outputs_hidden": false}
wave.make_audio()

# %% [markdown]
# Now we compute the DFT of the violin recording.

# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum()

# %% [markdown]
# I trimmed the violin recording to the same length as the impulse response:

# %% jupyter={"outputs_hidden": false}
len(spectrum.hs), len(transfer.hs)

# %% jupyter={"outputs_hidden": false}
spectrum.fs

# %% jupyter={"outputs_hidden": false}
transfer.fs

# %% [markdown]
# Now we can multiply in the frequency domain and the transform back to the time domain.

# %% jupyter={"outputs_hidden": false}
output = (spectrum * transfer).make_wave()
output.normalize()

# %% [markdown]
# Here's a  comparison of the original and transformed recordings:

# %% jupyter={"outputs_hidden": false}
wave.plot()

# %% jupyter={"outputs_hidden": false}
output.plot()

# %% [markdown]
# And here's what it sounds like:

# %% jupyter={"outputs_hidden": false}
output.make_audio()

# %% [markdown]
# Now that we recognize this operation as convolution, we can compute it using the convolve method:

# %% jupyter={"outputs_hidden": false}
convolved2 = wave.convolve(response)
convolved2.normalize()
convolved2.make_audio()
