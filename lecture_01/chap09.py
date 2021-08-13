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
# This notebook contains code examples from Chapter 9: Differentiation and Integration
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
# ## Difference
#
# As the first example, let's look at the Facebook data again.

# %%
if not os.path.exists('FB_2.csv'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/FB_2.csv

# %%
import pandas as pd

df = pd.read_csv('FB_2.csv', header=0, parse_dates=[0])
len(df)

# %%
from thinkdsp import Wave

ys = df['Close']

# for these examples, we need the wave to have 
# an even number of samples
if len(ys) % 2:
    ys = ys[:-1]

close = Wave(ys, framerate=1)
len(close)

# %% [markdown]
# Here's what the time series looks like (ignoring the gaps between trading days).

# %%
close.plot()
decorate(xlabel='Time (days)', ylabel='Price ($)')

# %% [markdown]
# And here's the spectrum on a log-log scale.

# %%
close_spectrum = close.make_spectrum()
close_spectrum.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude',
                 xscale='log', yscale='log')

# %% [markdown]
# The slope of the power spectrum is -1.86, which is similar to red noise (which should have a slope of -2).

# %%
close_spectrum.estimate_slope().slope

# %% [markdown]
# We can use `np.diff` to compute the difference between successive elements, which is the daily change.

# %%
change = Wave(np.diff(close.ys), framerate=1)
change.plot()
decorate(xlabel='Time (days)', ylabel='Price change($)')
len(change)

# %% [markdown]
# And here's the spectrum of the daily changes:

# %%
change_spectrum = change.make_spectrum()
change_spectrum.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude')

# %% [markdown]
# Recall that the spectrum of white noise looks like white noise.
#
# Here's the spectrum on a log-log scale.

# %%
change_spectrum.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude',
                 xscale='log', yscale='log')

# %% [markdown]
# The estimated slope is close to zero, which is consistent with white noise.

# %%
change_spectrum.estimate_slope().slope

# %% [markdown]
# We can think the diff operation as convolution with a difference window, [1, -1].
#
# And convolution with this window corresponds to multiplication by a filter.
#
# The following function computes the filter that corresponds to the window.

# %%
from thinkdsp import zero_pad

def make_filter(window, wave):
    """Computes the filter that corresponds to a window.
    
    window: NumPy array
    wave: wave used to choose the length and framerate
    
    returns: new Spectrum
    """
    padded = zero_pad(window, len(wave))
    window_wave = Wave(padded, framerate=wave.framerate)
    window_spectrum = window_wave.make_spectrum()
    return window_spectrum


# %% [markdown]
# And here's what the filter looks like for the difference window:

# %%
diff_window = np.array([1.0, -1.0])
diff_filter = make_filter(diff_window, close)
diff_filter.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude ratio')

# %% [markdown]
# And the angles:

# %%
plt.plot(diff_filter.angles)
decorate(xlabel='Frequency (1/day)', ylabel='Phase offset (radians)')

# %% [markdown]
# So we could also compute the daily changes by multiplying the spectrum of closing prices by the diff filter:

# %%
change_spectrum2 = close_spectrum * diff_filter
change_spectrum2.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude')

# %% [markdown]
# And then converting the spectrum to a wave.

# %%
change2 = change_spectrum2.make_wave()

# we have to trim the first element to avoid wrap-around
change2.ys = change2.ys[1:]
change2.ts = change2.ts[1:]

change.plot()
change2.plot()
decorate(xlabel='Time (day)', ylabel='Price change ($)')

# %% [markdown]
# Then we can confirm that we get the same result both ways (within floating point error).

# %%
change.max_diff(change2)

# %% [markdown]
# ### Differentiation
#
# This diff operation is an approximation of differentiation, and we can compute the filter for differentiation analytically: each complex component is multiplied by $2 \pi i f$.

# %%
#start with a filter that has the right size, then replace hs

PI2 = np.pi * 2
deriv_filter = close.make_spectrum()
deriv_filter.hs = PI2 * 1j * deriv_filter.fs
deriv_filter.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude ratio')

# %% [markdown]
# Now we can apply the derivative filter to the spectrum of closing prices:

# %%
deriv_spectrum = close.make_spectrum().differentiate()

deriv_spectrum.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude')

# %% [markdown]
# The results are similar to what we got from `np.diff`, with some differences due to (1) the difference window is only a coarse approximation of the derivative, especially at higher frequencies, and (2) the spectral derivative is based on the assumption that the signal is periodic, so the behavior at the beginning and end is different.

# %%
deriv = deriv_spectrum.make_wave()
len(deriv), len(change)

# %%
deriv = deriv_spectrum.make_wave()
change.plot(alpha=0.5)
deriv.plot(alpha=0.5)
decorate(xlabel='Time (day)')

# %% [markdown]
# We can see the differences more clearly by zooming in on a slice:

# %%
low, high = 0, 50
plt.plot(change.ys[low:high], label='diff')
plt.plot(deriv.ys[low:high], label='deriv')
decorate(xlabel='Time (day)', ylabel='Price change ($)')

# %% [markdown]
# The diffs and the spectral derivative are similar in many places, but sometimes substantially different.

# %% [markdown]
# Here's the difference between the derivative filter and the difference filter:

# %%
deriv_filter.plot()
diff_filter.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude ratio')

# %% [markdown]
# The difference filter does not amplify the highest frequencies as much, which is why the diffs are smoother than the derivative.

# %% [markdown]
# ## Integration
#
# Now let's think about integration.  We can compute the filter for integration analytically: each frequency component gets divided by $2 \pi i f$.
#
# I plot the result on a log-y scale so we can see it more clearly.

# %%
#start with a copy of the deriv filter and replace the hs
integ_filter = deriv_filter.copy()
integ_filter.hs[1:] = 1 / (PI2 * 1j * integ_filter.fs[1:])

# set the component at freq=0 to infinity
integ_filter.hs[0] = np.inf

integ_filter.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude ratio', 
                 yscale='log')

# %% [markdown]
# We can confirm that the integration filter is correct by applying it to the spectrum of the derivative we just computed:

# %%
integ_spectrum = deriv_spectrum.copy().integrate()
integ_spectrum.plot()
decorate(xlabel='Frequency (1/day)', ylabel='Amplitude')
decorate(yscale='log')

# %% [markdown]
# And then converting back to a wave.  The result is identical to the daily closing prices we started with, but shifted so the mean is 0.  
#
# The reason the mean is 0 is that the derivative clobbers the first element of the spectrum, which is the bias.  Once the bias information is lost, integration can't restore it.  So the result has an unspecified constant of integration.

# %%
close.plot(label='closing prices', alpha=0.7)

integ_spectrum.hs[0] = 0
integ_wave = integ_spectrum.make_wave()
integ_wave.plot(label='integrated derivative', alpha=0.7)
decorate(xlabel='Time (day)', ylabel='Price ($)')

# %%
shift = np.mean(close.ys) - np.mean(integ_wave.ys)

diff = integ_wave.ys - close.ys + shift
np.max(np.abs(diff))

# %% [markdown]
# ### Cumulative sum
#
# In the same way that the diff operator approximates differentiation, the cumulative sum approximates integration.
#
# I'll demonstrate with a Sawtooth signal.

# %%
from thinkdsp import SawtoothSignal

in_wave = SawtoothSignal(freq=50).make_wave(duration=0.1, framerate=44100)
in_wave.unbias()
in_wave.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Here's the spectrum before the cumulative sum:

# %%
in_spectrum = in_wave.make_spectrum()
in_spectrum.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# The output wave is the cumulative sum of the input

# %%
out_wave = in_wave.cumsum()
out_wave.unbias()
out_wave.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# And here's its spectrum

# %%
out_spectrum = out_wave.make_spectrum()
out_spectrum.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# Now we compute the ratio of the output to the input:

# %%
sum(in_spectrum.amps < 1), len(in_spectrum)

# %% [markdown]
# In between the harmonics, the input componenents are small, so I set those ratios to NaN.

# %%
ratio_spectrum = out_spectrum.ratio(in_spectrum, thresh=1)
ratio_spectrum.plot(marker='.', ms=4, ls='')

decorate(xlabel='Frequency (Hz)',
         ylabel='Amplitude ratio',
         yscale='log')

# %% [markdown]
# To get the cumsum filter, I compute the diff filter again and invert it.

# %%
# compute the diff filter
diff_window = np.array([1.0, -1.0])
padded = zero_pad(diff_window, len(in_wave))
diff_wave = Wave(padded, framerate=in_wave.framerate)
diff_filter = diff_wave.make_spectrum()

# %%
# compute the cumsum filter by inverting the diff filter
cumsum_filter = diff_filter.copy()
cumsum_filter.hs[1:] = 1 / cumsum_filter.hs[1:]
cumsum_filter.hs[0] = np.inf

# %%
# compute the integration filter
integ_filter = cumsum_filter.copy()
integ_filter.hs[1:] = integ_filter.framerate / (PI2 * 1j * integ_filter.fs[1:])
integ_filter.hs[0] = np.inf

# %%
cumsum_filter.plot(label='cumsum filter', alpha=0.7)
integ_filter.plot(label='integral filter', alpha=0.7)

decorate(xlabel='Frequency (Hz)',
         ylabel='Amplitude ratio',
         yscale='log')

# %% [markdown]
# Finally, we can compare the computed ratios to the filter.  They match, confirming that the cumsum filter is the inverse of the diff filter.

# %%
cumsum_filter.plot(label='cumsum filter')
ratio_spectrum.plot(label='ratio', marker='.', ms=4, ls='')
decorate(xlabel='Frequency (Hz)',
         ylabel='Amplitude ratio',
         yscale='log')

# %% [markdown]
# Now we can compute the output wave using the convolution theorem, and compare the results:

# %%
len(in_spectrum), len(cumsum_filter)

# %%
out_wave.plot(label='summed', alpha=0.7)

cumsum_filter.hs[0] = 0
out_wave2 = (in_spectrum * cumsum_filter).make_wave()
out_wave2.plot(label='filtered', alpha=0.7)

decorate(xlabel='Time (s)')

# %% [markdown]
# They are the same, within floating point error.

# %%
out_wave.max_diff(out_wave2)

# %%
