# -*- coding: utf-8 -*-
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
# This notebook contains solutions to exercises in Chapter 4: Noise
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
# ``A Soft Murmur'' is a web site that plays a mixture of natural
# noise sources, including rain, waves, wind, etc.  At
# http://asoftmurmur.com/about/ you can find their list
# of recordings, most of which are at http://freesound.org.
#
# Download a few of these files and compute the spectrum of each
# signal.  Does the power spectrum look like white noise, pink noise,
# or Brownian noise?  How does the spectrum vary over time?

# %%
if not os.path.exists('132736__ciccarelli__ocean-waves.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/132736__ciccarelli__ocean-waves.wav

# %% jupyter={"outputs_hidden": false}
from thinkdsp import read_wave

wave = read_wave('132736__ciccarelli__ocean-waves.wav')
wave.make_audio()

# %% [markdown]
# I chose a recording of ocean waves.  I selected a short segment:

# %% jupyter={"outputs_hidden": false}
segment = wave.segment(start=1.5, duration=1.0)
segment.make_audio()

# %% [markdown]
# And here's its spectrum:

# %% jupyter={"outputs_hidden": false}
spectrum = segment.make_spectrum()
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)',
         ylabel='Power')

# %% [markdown]
# Amplitude drops off with frequency, so this might be red or pink noise.  We can check by looking at the power spectrum on a log-log scale.

# %% jupyter={"outputs_hidden": false}
spectrum.plot_power()

loglog = dict(xscale='log', yscale='log')
decorate(xlabel='Frequency (Hz)',
         ylabel='Power', 
         **loglog)

# %% [markdown]
# This structure, with increasing and then decreasing amplitude, seems to be common in natural noise sources.
#
# Above $f = 10^3$, it might be dropping off linearly, but we can't really tell.
#
# To see how the spectrum changes over time, I'll select another segment:

# %% jupyter={"outputs_hidden": false}
segment2 = wave.segment(start=2.5, duration=1.0)
segment2.make_audio()

# %% [markdown]
# And plot the two spectrums:

# %% jupyter={"outputs_hidden": false}
spectrum2 = segment2.make_spectrum()

spectrum.plot_power(alpha=0.5)
spectrum2.plot_power(alpha=0.5)
decorate(xlabel='Frequency (Hz)',
         ylabel='Power')

# %% [markdown]
# Here they are again, plotting power on a log-log scale.

# %% jupyter={"outputs_hidden": false}
spectrum.plot_power(alpha=0.5)
spectrum2.plot_power(alpha=0.5)
decorate(xlabel='Frequency (Hz)',
         ylabel='Power',
         **loglog)

# %% [markdown]
# So the structure seems to be consistent over time.
#
# We can also look at a spectrogram:

# %% jupyter={"outputs_hidden": false}
segment.make_spectrogram(512).plot(high=5000)
decorate(xlabel='Time(s)', ylabel='Frequency (Hz)')

# %% [markdown]
# Within this segment, the overall amplitude drops off, but the mixture of frequencies seems consistent.

# %% [markdown]
# **Exercise:** In a noise signal, the mixture of frequencies changes over time.
# In the long run, we expect the power at all frequencies to be equal,
# but in any sample, the power at each frequency is random.
#
# To estimate the long-term average power at each frequency, we can
# break a long signal into segments, compute the power spectrum for each segment, and then compute the average across
# the segments.  You can read more about this algorithm at
# http://en.wikipedia.org/wiki/Bartlett's_method.
#
# Implement Bartlett's method and use it to estimate the power
# spectrum for a noise wave.  Hint: look at the implementation
# of `make_spectrogram`.

# %%
from thinkdsp import Spectrum

def bartlett_method(wave, seg_length=512, win_flag=True):
    """Estimates the power spectrum of a noise wave.
    
    wave: Wave
    seg_length: segment length
    """
    # make a spectrogram and extract the spectrums
    spectro = wave.make_spectrogram(seg_length, win_flag)
    spectrums = spectro.spec_map.values()
    
    # extract the power array from each spectrum
    psds = [spectrum.power for spectrum in spectrums]
    
    # compute the root mean power (which is like an amplitude)
    hs = np.sqrt(sum(psds) / len(psds))
    fs = next(iter(spectrums)).fs
    
    # make a Spectrum with the mean amplitudes
    spectrum = Spectrum(hs, fs, wave.framerate)
    return spectrum


# %% [markdown]
# `bartlett_method` makes a spectrogram and extracts `spec_map`, which maps from times to Spectrum objects.  It computes the PSD for each spectrum, adds them up, and puts the results into a Spectrum object.

# %% jupyter={"outputs_hidden": false}
psd = bartlett_method(segment)
psd2 = bartlett_method(segment2)

psd.plot_power()
psd2.plot_power()

decorate(xlabel='Frequency (Hz)', 
         ylabel='Power', 
         **loglog)

# %% [markdown]
# Now we can see the relationship between power and frequency more clearly.  It is not a simple linear relationship, but it is consistent across different segments, even in details like the notches near 5000 Hz, 6000 Hz, and above 10,000 Hz. 

# %% [markdown]
# ## Exercise 2
#
# At [coindesk](https://www.coindesk.com/price/bitcoin) you can download the daily price of a BitCoin as a CSV file.  Read this file and compute
# the spectrum of BitCoin prices as a function of time.
# Does it resemble white, pink, or Brownian noise?

# %%
if not os.path.exists('BTC_USD_2013-10-01_2020-03-26-CoinDesk.csv'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/BTC_USD_2013-10-01_2020-03-26-CoinDesk.csv

# %% jupyter={"outputs_hidden": false}
import pandas as pd

df = pd.read_csv('BTC_USD_2013-10-01_2020-03-26-CoinDesk.csv', 
                 parse_dates=[0])
df

# %% jupyter={"outputs_hidden": false}
ys = df['Closing Price (USD)']
ts = df.index

# %% jupyter={"outputs_hidden": false}
from thinkdsp import Wave

wave = Wave(ys, ts, framerate=1)
wave.plot()
decorate(xlabel='Time (days)')

# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum()
spectrum.plot_power()
decorate(xlabel='Frequency (1/days)',
         ylabel='Power', 
         **loglog)

# %% jupyter={"outputs_hidden": false}
spectrum.estimate_slope()[0]

# %% [markdown]
# Red noise should have a slope of -2.  The slope of this PSD is close to 1.7, so it's hard to say if we should consider it red noise or if we should say it's a kind of pink noise.

# %% [markdown]
# ## Exercise 3
#
# A Geiger counter is a device that detects radiation. When an ionizing particle strikes the detector, it outputs a surge of current. The total output at a point in time can be modeled as uncorrelated Poisson (UP) noise, where each sample is a random quantity from a Poisson distribution, which corresponds to the number of particles detected during an interval.
#
# Write a class called `UncorrelatedPoissonNoise` that inherits from ` _Noise` and provides `evaluate`. It should use `np.random.poisson` to generate random values from a Poisson distribution. The parameter of this function, `lam`, is the average number of particles during each interval. You can use the attribute `amp` to specify `lam`. For example, if the framerate is 10 kHz and `amp` is 0.001, we expect about 10 “clicks” per second.
#
# Generate about a second of UP noise and listen to it. For low values of `amp`, like 0.001, it should sound like a Geiger counter. For higher values it should sound like white noise. Compute and plot the power spectrum to see whether it looks like white noise. 

# %%
from thinkdsp import Noise

class UncorrelatedPoissonNoise(Noise):
    """Represents uncorrelated Poisson noise."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        ys = np.random.poisson(self.amp, len(ts))
        return ys


# %% [markdown]
# Here's what it sounds like at low levels of "radiation".

# %% jupyter={"outputs_hidden": false}
amp = 0.001
framerate = 10000
duration = 1

signal = UncorrelatedPoissonNoise(amp=amp)
wave = signal.make_wave(duration=duration, framerate=framerate)
wave.make_audio()

# %% [markdown]
# To check that things worked, we compare the expected number of particles and the actual number:

# %% jupyter={"outputs_hidden": false}
expected = amp * framerate * duration
actual = sum(wave.ys)
print(expected, actual)

# %% [markdown]
# Here's what the wave looks like:

# %% jupyter={"outputs_hidden": false}
wave.plot()

# %% [markdown]
# And here's its power spectrum on a log-log scale.

# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum()
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)',
         ylabel='Power',
         **loglog)

# %% [markdown]
# Looks like white noise, and the slope is close to 0.

# %% jupyter={"outputs_hidden": false}
spectrum.estimate_slope().slope

# %% [markdown]
# With a higher arrival rate, it sounds more like white noise:

# %% jupyter={"outputs_hidden": false}
amp = 1
framerate = 10000
duration = 1

signal = UncorrelatedPoissonNoise(amp=amp)
wave = signal.make_wave(duration=duration, framerate=framerate)
wave.make_audio()

# %% [markdown]
# It looks more like a signal:

# %% jupyter={"outputs_hidden": false}
wave.plot()

# %% [markdown]
# And the spectrum converges on Gaussian noise.

# %%
import matplotlib.pyplot as plt

def normal_prob_plot(sample, fit_color='0.8', **options):
    """Makes a normal probability plot with a fitted line.

    sample: sequence of numbers
    fit_color: color string for the fitted line
    options: passed along to Plot
    """
    n = len(sample)
    xs = np.random.normal(0, 1, n)
    xs.sort()
    
    ys = np.sort(sample)
    
    mean, std = np.mean(sample), np.std(sample)
    fit_ys = mean + std * xs
    plt.plot(xs, fit_ys, color='gray', alpha=0.5, label='model')

    plt.plot(xs, ys, **options)


# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum()
spectrum.hs[0] = 0

normal_prob_plot(spectrum.real, label='real')
decorate(xlabel='Normal sample',
        ylabel='Power')

# %% jupyter={"outputs_hidden": false}
normal_prob_plot(spectrum.imag, label='imag', color='C1')
decorate(xlabel='Normal sample')

# %% [markdown]
# ## Exercise 4
#
# The algorithm in this chapter for generating pink noise is
# conceptually simple but computationally expensive.  There are
# more efficient alternatives, like the Voss-McCartney algorithm.
# Research this method, implement it, compute the spectrum of
# the result, and confirm that it has the desired relationship
# between power and frequency.

# %% [markdown]
# **Solution:** The fundamental idea of this algorithm is to add up several sequences of random numbers that get updates at different sampling rates.  The first source should get updated at every time step; the second source every other time step, the third source ever fourth step, and so on.
#
# In the original algorithm, the updates are evenly spaced.  In an alternative proposed at http://www.firstpr.com.au/dsp/pink-noise/, they are randomly spaced.
#
# My implementation starts with an array with one row per timestep and one column for each of the white noise sources.  Initially, the first row and the first column are random and the rest of the array is Nan.

# %% jupyter={"outputs_hidden": false}
nrows = 100
ncols = 5

array = np.empty((nrows, ncols))
array.fill(np.nan)
array[0, :] = np.random.random(ncols)
array[:, 0] = np.random.random(nrows)
array[0:6]

# %% [markdown]
# The next step is to choose the locations where the random sources change.  If the number of rows is $n$, the number of changes in the first column is $n$, the number in the second column is $n/2$ on average, the number in the third column is $n/4$ on average, etc.
#
# So the total number of changes in the matrix is $2n$ on average; since $n$ of those are in the first column, the other $n$ are in the rest of the matrix.
#
# To place the remaining $n$ changes, we generate random columns from a geometric distribution with $p=0.5$.  If we generate a value out of bounds, we set it to 0 (so the first column gets the extras).

# %% jupyter={"outputs_hidden": false}
p = 0.5
n = nrows
cols = np.random.geometric(p, n)
cols[cols >= ncols] = 0
cols

# %% [markdown]
# Within each column, we choose a random row from a uniform distribution.  Ideally we would choose without replacement, but it is faster and easier to choose with replacement, and I doubt it matters.

# %% jupyter={"outputs_hidden": false}
rows = np.random.randint(nrows, size=n)
rows

# %% [markdown]
# Now we can put random values at rach of the change points.

# %% jupyter={"outputs_hidden": false}
array[rows, cols] = np.random.random(n)
array[0:6]

# %% [markdown]
# Next we want to do a zero-order hold to fill in the NaNs.  NumPy doesn't do that, but Pandas does.  So I'll create a DataFrame:

# %% jupyter={"outputs_hidden": false}
df = pd.DataFrame(array)
df.head()

# %% [markdown]
# And then use `fillna` along the columns.

# %% jupyter={"outputs_hidden": false}
filled = df.fillna(method='ffill', axis=0)
filled.head()

# %% [markdown]
# Finally we add up the rows.

# %% jupyter={"outputs_hidden": false}
total = filled.sum(axis=1)
total.head()

# %% [markdown]
# If we put the results into a Wave, here's what it looks like:

# %% jupyter={"outputs_hidden": false}
wave = Wave(total.values)
wave.plot()


# %% [markdown]
# Here's the whole process in a function:

# %%
def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values


# %% [markdown]
# To test it I'll generate 11025 values:

# %% jupyter={"outputs_hidden": false}
ys = voss(11025)
ys

# %% [markdown]
# And make them into a Wave:

# %% jupyter={"outputs_hidden": false}
wave = Wave(ys)
wave.unbias()
wave.normalize()

# %% [markdown]
# Here's what it looks like:

# %% jupyter={"outputs_hidden": false}
wave.plot()

# %% [markdown]
# As expected, it is more random-walk-like than white noise, but more random looking than red noise.
#
# Here's what it sounds like:

# %% jupyter={"outputs_hidden": false}
wave.make_audio()

# %% [markdown]
# And here's the power spectrum:

# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum()
spectrum.hs[0] = 0
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)',
         ylabel='Power',
         **loglog)

# %% [markdown]
# The estimated slope is close to -1.

# %% jupyter={"outputs_hidden": false}
spectrum.estimate_slope().slope

# %% [markdown]
# We can get a better sense of the average power spectrum by generating a longer sample:

# %% jupyter={"outputs_hidden": false}
seg_length = 64 * 1024
iters = 100
wave = Wave(voss(seg_length * iters))
len(wave)

# %% [markdown]
# And using Barlett's method to compute the average.

# %% jupyter={"outputs_hidden": false}
spectrum = bartlett_method(wave, seg_length=seg_length, win_flag=False)
spectrum.hs[0] = 0
len(spectrum)

# %% [markdown]
# It's pretty close to a straight line, with some curvature at the highest frequencies.

# %% jupyter={"outputs_hidden": false}
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)',
         ylabel='Power',
         **loglog)

# %% [markdown]
# And the slope is close to -1.

# %% jupyter={"outputs_hidden": false}
spectrum.estimate_slope().slope

# %% jupyter={"outputs_hidden": true}
