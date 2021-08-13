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
# ThinkDSP
# ========
#
# This notebook contains code examples from Chapter 10: Signals and Systems
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
# Impulse response
# --
# To understand why the impulse response is sufficient to characterize a system, it is informative to look at the DFT of an impulse:

# %%
from thinkdsp import Wave

impulse = np.zeros(8)
impulse[0] = 1
wave = Wave(impulse, framerate=8)
print(wave.ys)

# %% [markdown]
# The DFT of an impulse is all ones, which means that the impulse contains equal energy at all frequencies.  So testing a system with an impulse is like testing it will all frequency components at the same time:

# %%
impulse_spectrum = wave.make_spectrum(full=True)
print(impulse_spectrum.hs)

# %% [markdown]
# You might notice something about the impulse and its DFT:

# %%
np.sum(wave.ys**2)

# %%
np.sum(impulse_spectrum.hs**2)

# %% [markdown]
# In general, the total magnitue of DFT(y) is N times the total magnitude of y.
#

# %%
impulse = np.zeros(10000)
impulse[0] = 1
wave = Wave(impulse, framerate=10000)
wave.plot()
decorate(xlabel='Time (s)')

# %%
wave.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# System characterization
# --
#
# Let's look at a mini example of system characterization.  Suppose you have a system that smooths the signal by taking a moving average of adjacent elements:

# %%
window_array = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0,])
window = Wave(window_array, framerate=8)

# %% [markdown]
# For this moving average window, we can compute the transfer function:

# %%
filtr = window.make_spectrum(full=True)
print(filtr.hs)

# %% [markdown]
# Here are the magnitudes:

# %%
filtr.amps

# %%
filtr.plot()
decorate(xlabel='Frequency', ylabel='Amplitude')

# %% [markdown]
# If you multiply the transfer function by the spectrum of an impulse (which is all ones), the result is the filter:

# %%
product = impulse_spectrum * filtr
print(product.hs)

# %%
np.max(np.abs(product.hs - filtr.hs))

# %% [markdown]
# Now if you transform back to the time domain, you have the impulse response, which looks a lot like the window:

# %%
filtered = product.make_wave()
filtered.plot()
decorate(xlabel='Time')

# %%
print(filtered.ys.real)

# %% [markdown]
# This example is meant to demonstrate why a recording of an impulse response is sufficient to characterize a system: because it is the IDFT of the transfer function.

# %% [markdown]
# Acoustic impulse response
# --
#
# Here's a recording of a gunshot, which approximates the acoustic impulse response of the room:

# %%
if not os.path.exists('180960__kleeb__gunshot.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/180960__kleeb__gunshot.wav

# %%
from thinkdsp import read_wave

response = read_wave('180960__kleeb__gunshot.wav')

start = 0.12
response = response.segment(start=start)
response.shift(-start)

response.normalize()
response.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Here's what it sounds like:

# %%
response.make_audio()

# %% [markdown]
# The DFT of the impulse response is the transfer function:

# %%
transfer = response.make_spectrum()
transfer.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# Here's the transfer function on a log-log scale:

# %%
transfer.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude',
         xscale='log', yscale='log')

# %% [markdown]
# Now we can simulate what a recording would sound like if it were played in the same room and recorded in the same way.  Here's the violin recording we have used before:

# %%
if not os.path.exists('92002__jcveliz__violin-origional.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/92002__jcveliz__violin-origional.wav

# %%
violin = read_wave('92002__jcveliz__violin-origional.wav')

start = 0.11
violin = violin.segment(start=start)
violin.shift(-start)

violin.truncate(len(response))
violin.normalize()
violin.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Here's what it sounds like before transformation:

# %%
violin.make_audio()

# %% [markdown]
# Now we compute the DFT of the violin recording.

# %%
spectrum = violin.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# I trimmed the violin recording to the same length as the impulse response:

# %%
len(spectrum.hs), len(transfer.hs)

# %% [markdown]
# We we can multiply in the frequency domain and the transform back to the time domain.

# %%
output = (spectrum * transfer).make_wave()
output.normalize()

# %% [markdown]
# Here's a  comparison of the original and transformed recordings:

# %%
violin.plot()
decorate(xlabel='Time (s)')

# %%
output.plot()
decorate(xlabel='Time (s)')

# %%
spectrum = output.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# And here's what it sounds like:

# %%
output.make_audio()


# %% [markdown]
# At the beginning of the output, you might notice an extra note that has wrapped around from the end.  The reason is that multiplication in the frequency domain corresponds to *circular* convolution, which assumes that the signal is periodic.  When the signal is not periodic, we can avoid wrap-around by padding the signal with zeros.

# %% [markdown]
# Convolution
# --
#
# To understand how that worked, you can think about the input signal as a series of impulses, and the output as the sum of shifted, scaled versions of the impulse response.

# %%
def shifted_scaled(wave, shift, factor):
    """Make a shifted, scaled copy of a wave.
    
    wave: Wave
    shift: time shift, float
    factor: multiplier, float
    """
    res = wave.copy()
    res.shift(shift)
    res.scale(factor)
    return res


# %% [markdown]
# Here's what it would sound like if we fired a big gun followed by a small gun:

# %%
dt = 1
factor = 0.5

response2 = response + shifted_scaled(response, dt, factor)
response2.plot()
decorate(xlabel='Time (s)', ylabel='Amplitude')

# %% [markdown]
# Two gunshots:

# %%
response2.make_audio()

# %% [markdown]
# Adding up shifted, scaled copies of the impulse response doesn't always sounds like gunshots.  If there are enough of them, close enough together, it sounds like a wave.
#
# Here's what it sounds like if we fire 220 guns at a rate of 441 gunshots per second:

# %%
dt = 1 / 441
total = 0
for k in range(220):
    total += shifted_scaled(response, k*dt, 1.0)
total.normalize()

# %%
total.plot()

# %% [markdown]
# Here's what it sounds like:

# %%
total.make_audio()

# %% [markdown]
# To me it sounds a bit like a car horn in a garage.
#

# %% [markdown]
# We can do the same thing with an arbitrary input signal.

# %%
from thinkdsp import SawtoothSignal

signal = SawtoothSignal(freq=441)
wave = signal.make_wave(duration=0.2, framerate=response.framerate)
wave.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# And here's what we get if we use the wave to generate shifted, scaled versions of the impulse response:

# %%
total = 0
for t, y in zip(wave.ts, wave.ys):
    total += shifted_scaled(response, t, y)
total.normalize()

# %% [markdown]
# The result is a simulation of what the wave would sound like if it was recorded in the room where the gunshot was recorded:

# %%
total.plot()
decorate(xlabel='Time (s)', ylabel='Amplitude')

# %% [markdown]
# And here's what it sounds like:

# %%
total.make_audio()

# %% [markdown]
# Here's a comparison of the spectrum before and after convolution:

# %%
high = 5000
wave.make_spectrum().plot(high=high, color='0.7')

segment = total.segment(duration=0.2)
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# Now that we recognize this operation as convolution, we can compute it using the convolve method:

# %%
convolved = wave.convolve(response)
convolved.normalize()
convolved.make_audio()

# %% [markdown]
# And we can do the same thing with the violin recording:

# %%
convolved2 = violin.convolve(response)
convolved2.normalize()
convolved2.make_audio()

# %%
convolved2.plot()

# %%
