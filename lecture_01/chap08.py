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
# This notebook contains code examples from Chapter 8: Filtering and Convolution
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

# %%
# suppress scientific notation for small numbers
np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ### Smoothing
#
# As the first example, I'll look at [daily closing stock prices for Facebook](https://finance.yahoo.com/quote/FB/history?period1=1337299200&period2=1585353600&interval=1d&filter=history&frequency=1d), from its IPO on 2012-05-18 to 2020-03-27 (note: the dataset includes only trading days )

# %%
if not os.path.exists('FB_2.csv'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/FB_2.csv

# %%
import pandas as pd

df = pd.read_csv('FB_2.csv', header=0, parse_dates=[0])
df.head()

# %%
df.tail()

# %% [markdown]
# Extract the close prices and days since start of series:

# %%
close = df['Close']
dates = df['Date']
days = (dates - dates[0]) / np.timedelta64(1,'D')

# %% [markdown]
# Make a window to compute a 30-day moving average and convolve the window with the data.  The `valid` flag means the convolution is only computed where the window completely overlaps with the signal.

# %%
M = 30
window = np.ones(M)
window /= sum(window)
smoothed = np.convolve(close, window, mode='valid')
smoothed_days = days[M//2: len(smoothed) + M//2]

# %% [markdown]
# Plot the original and smoothed signals.

# %%
plt.plot(days, close, color='gray', alpha=0.6, label='daily close')
plt.plot(smoothed_days, smoothed, color='C1', alpha=0.6, label='30 day average')

decorate(xlabel='Time (days)', ylabel='Price ($)')

# %% [markdown]
# ## Smoothing sound signals
#
# Generate a 440 Hz sawtooth signal.

# %%
from thinkdsp import SawtoothSignal

signal = SawtoothSignal(freq=440)
wave = signal.make_wave(duration=1.0, framerate=44100)
wave.make_audio()

# %% [markdown]
# Make a moving average window.

# %%
window = np.ones(11)
window /= sum(window)
plt.plot(window)
decorate(xlabel='Index')

# %% [markdown]
# Plot the wave.

# %%
segment = wave.segment(duration=0.01)
segment.plot()
decorate(xlabel='Time (s)')


# %% [markdown]
# Pad the window so it's the same length as the signal, and plot it.

# %%
def zero_pad(array, n):
    """Extends an array with zeros.

    array: NumPy array
    n: length of result

    returns: new NumPy array
    """
    res = np.zeros(n)
    res[:len(array)] = array
    return res


# %%
N = len(segment)
padded = zero_pad(window, N)
plt.plot(padded)
decorate(xlabel='Index')

# %% [markdown]
# Apply the window to the signal (with lag=0).

# %%
prod = padded * segment.ys
np.sum(prod)

# %% [markdown]
# Compute a convolution by rolling the window to the right.

# %%
smoothed = np.zeros(N)
rolled = padded.copy()
for i in range(N):
    smoothed[i] = sum(rolled * segment.ys)
    rolled = np.roll(rolled, 1)

# %%
plt.plot(rolled)
decorate(xlabel='Index')

# %% [markdown]
# Plot the result of the convolution and the original.

# %%
from thinkdsp import Wave

segment.plot(color='gray')
smooth = Wave(smoothed, framerate=wave.framerate)
smooth.plot()
decorate(xlabel='Time(s)')

# %% [markdown]
# Compute the same convolution using `numpy.convolve`.

# %%
segment.plot(color='gray')
ys = np.convolve(segment.ys, window, mode='valid')
smooth2 = Wave(ys, framerate=wave.framerate)
smooth2.plot()
decorate(xlabel='Time(s)')

# %% [markdown]
# ## Frequency domain
#
# Let's see what's happening in the frequency domain.

# %% [markdown]
# Compute the smoothed wave using `np.convolve`, which is much faster than my version above.

# %%
convolved = np.convolve(wave.ys, window, mode='same')
smooth = Wave(convolved, framerate=wave.framerate)
smooth.make_audio()

# %% [markdown]
# Plot spectrums of the original and smoothed waves:

# %%
spectrum = wave.make_spectrum()
spectrum.plot(color='gray')

spectrum2 = smooth.make_spectrum()
spectrum2.plot()

decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# For each harmonic, compute the ratio of the amplitudes before and after smoothing.

# %%
amps = spectrum.amps
amps2 = spectrum2.amps
ratio = amps2 / amps    
ratio[amps<280] = 0

plt.plot(ratio)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude ratio')

# %% [markdown]
# Plot the ratios again, but also plot the FFT of the window.

# %%
padded =  zero_pad(window, len(wave))
dft_window = np.fft.rfft(padded)

plt.plot(np.abs(dft_window), color='gray', label='DFT(window)')
plt.plot(ratio, label='amplitude ratio')

decorate(xlabel='Frequency (Hz)', ylabel='Amplitude ratio')

# %% [markdown]
# ### Gaussian window
#
# Let's compare boxcar and Gaussian windows.

# %% [markdown]
# Make the boxcar window.

# %%
boxcar = np.ones(11)
boxcar /= sum(boxcar)

# %% [markdown]
# Make the Gaussian window.

# %%
import scipy.signal

gaussian = scipy.signal.gaussian(M=11, std=2)
gaussian /= sum(gaussian)

# %% [markdown]
# Plot the two windows.

# %%
plt.plot(boxcar, label='boxcar')
plt.plot(gaussian, label='Gaussian')
decorate(xlabel='Index')

# %% [markdown]
# Convolve the square wave with the Gaussian window.

# %%
ys = np.convolve(wave.ys, gaussian, mode='same')
smooth = Wave(ys, framerate=wave.framerate)
spectrum2 = smooth.make_spectrum()

# %% [markdown]
# Compute the ratio of the amplitudes.

# %%
amps = spectrum.amps
amps2 = spectrum2.amps
ratio = amps2 / amps    
ratio[amps<560] = 0

# %% [markdown]
# Compute the FFT of the window.

# %%
padded =  zero_pad(gaussian, len(wave))
dft_gaussian = np.fft.rfft(padded)

# %% [markdown]
# Plot the ratios and the FFT of the window.

# %%
plt.plot(np.abs(dft_gaussian), color='0.7', label='Gaussian filter')
plt.plot(ratio, label='amplitude ratio')

decorate(xlabel='Frequency (Hz)', ylabel='Amplitude ratio')

# %% [markdown]
# Combine the preceding example into one big function so we can interact with it.

# %%
from thinkdsp import SquareSignal

def plot_filter(M=11, std=2):
    signal = SquareSignal(freq=440)
    wave = signal.make_wave(duration=1, framerate=44100)
    spectrum = wave.make_spectrum()

    gaussian = scipy.signal.gaussian(M=M, std=std)
    gaussian /= sum(gaussian)

    ys = np.convolve(wave.ys, gaussian, mode='same')
    smooth =  Wave(ys, framerate=wave.framerate)
    spectrum2 = smooth.make_spectrum()

    # plot the ratio of the original and smoothed spectrum
    amps = spectrum.amps
    amps2 = spectrum2.amps
    ratio = amps2 / amps    
    ratio[amps<560] = 0

    # plot the same ratio along with the FFT of the window
    padded =  zero_pad(gaussian, len(wave))
    dft_gaussian = np.fft.rfft(padded)

    plt.plot(np.abs(dft_gaussian), color='gray', label='Gaussian filter')
    plt.plot(ratio, label='amplitude ratio')

    decorate(xlabel='Frequency (Hz)', ylabel='Amplitude ratio')
    plt.show()


# %% [markdown]
# Try out different values of `M` and `std`.

# %%
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

slider = widgets.IntSlider(min=2, max=100, value=11)
slider2 = widgets.FloatSlider(min=0, max=20, value=2)
interact(plot_filter, M=slider, std=slider2);

# %% [markdown]
# ## Convolution theorem
#
# Let's use the Convolution theorem to compute convolutions using FFT.  
#
# I'll use the Facebook data again, and smooth it using `np.convolve` and a 30-day Gaussian window.
#
# I ignore the dates and treat the values as if they are equally spaced in time.

# %%
window = scipy.signal.gaussian(M=30, std=6)
window /= window.sum()
smoothed = np.convolve(close, window, mode='valid')

len(close), len(smoothed)

# %% [markdown]
# Plot the original and smoothed data.

# %%
plt.plot(close, color='gray')
plt.plot(smoothed)
decorate(xlabel='Time (days)', ylabel='Price ($)')

# %% [markdown]
# Pad the window and compute its FFT.

# %%
N = len(close)
padded = zero_pad(window, N)
fft_window = np.fft.fft(padded)
plt.plot(np.abs(fft_window))
decorate(xlabel='Index')

# %% [markdown]
# Apply the convolution theorem.

# %%
fft_signal = np.fft.fft(close)
smoothed2 = np.fft.ifft(fft_signal * fft_window)
M = len(window)
smoothed2 = smoothed2[M-1:]

# %% [markdown]
# Plot the two signals (smoothed with numpy and FFT).

# %%
plt.plot(smoothed)
plt.plot(smoothed2.real)
decorate(xlabel='Time (days)', ylabel='Price ($)')

# %% [markdown]
# Confirm that the difference is small.

# %%
diff = smoothed - smoothed2
np.max(np.abs(diff))

# %% [markdown]
# `scipy.signal` provides `fftconvolve`, which computes convolutions using FFT.

# %%
smoothed3 = scipy.signal.fftconvolve(close, window, mode='valid')

# %% [markdown]
# Confirm that it gives the same answer, at least approximately.

# %%
diff = smoothed - smoothed3
np.max(np.abs(diff))


# %% [markdown]
# We can encapsulate the process in a function:

# %%
def fft_convolve(signal, window):
    fft_signal = np.fft.fft(signal)
    fft_window = np.fft.fft(window)
    return np.fft.ifft(fft_signal * fft_window)


# %% [markdown]
# And confirm that it gives the same answer.

# %%
smoothed4 = fft_convolve(close, padded)[M-1:]
len(smoothed4)

# %%
diff = smoothed - smoothed4
np.max(np.abs(diff))

# %% [markdown]
# ### Autocorrelation
#
# We can also use the convolution theorem to compute autocorrelation functions.
#
# Compute autocorrelation using `numpy.correlate`:
#

# %%
corrs = np.correlate(close, close, mode='same')
corrs[:7]


# %% [markdown]
# Compute autocorrelation using my `fft_convolve`.  The window is a reversed copy of the signal.  We have to pad the window and signal with zeros and then select the middle half from the result.

# %%
def fft_autocorr(signal):
    N = len(signal)
    signal = zero_pad(signal, 2*N)
    window = np.flipud(signal)

    corrs = fft_convolve(signal, window)
    corrs = np.roll(corrs, N//2+1)[:N]
    return corrs


# %% [markdown]
# Test the function.

# %%
corrs2 = fft_autocorr(close)
corrs2[:7]

# %% [markdown]
# Plot the results.

# %%
lags = np.arange(N) - N//2
plt.plot(lags, corrs, color='gray', alpha=0.5, label='np.convolve')
plt.plot(lags, corrs2.real, color='C1', alpha=0.5, label='fft_convolve')
decorate(xlabel='Lag', ylabel='Correlation')
len(corrs), len(corrs2)

# %% [markdown]
# Confirm that the difference is small.

# %%
diff = corrs - corrs2.real
np.max(np.abs(diff))

# %% jupyter={"outputs_hidden": true}
