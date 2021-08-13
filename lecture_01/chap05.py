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
# This notebook contains code examples from Chapter 5: Autocorrelation
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
# To investigate serial correlation of signals, let's start with a sine wave at 440 Hz.

# %% jupyter={"outputs_hidden": false}
from thinkdsp import SinSignal

def make_sine(offset):
    signal = SinSignal(freq=440, offset=offset)
    wave = signal.make_wave(duration=0.5, framerate=10000)
    return wave


# %% [markdown]
# I'll make two waves with different phase offsets.

# %% jupyter={"outputs_hidden": false}
wave1 = make_sine(offset=0)
wave2 = make_sine(offset=1)

wave1.segment(duration=0.01).plot()
wave2.segment(duration=0.01).plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# The two waves appears correlated: when one is high, the other is usually high, too.
#
# We can use `np.corrcoef` to compute the correlation matrix.

# %% jupyter={"outputs_hidden": false}
print(np.corrcoef(wave1.ys, wave2.ys))

# %% [markdown]
# The diagonal elements are the correlations of the waves with themselves, which is why they are 1.
# The off-diagonal elements are the correlations between the two waves.  In this case, 0.54 indicates that there is a moderate correlation between these waves.
#
# The correlation matrix is more interesting when there are more than two waves.  With only two waves, there is really only one number in the matrix we care about.
#
# ` Wave` provides `corr`, which computes the correlation between waves:

# %% jupyter={"outputs_hidden": false}
wave1.corr(wave2)


# %% [markdown]
# To investigate the relationship between phase offset and correlation, I'll make an interactive function that computes correlation for each offset:

# %% jupyter={"outputs_hidden": false}
def compute_corr(offset):
    wave1 = make_sine(offset=0)
    wave2 = make_sine(offset=-offset)

    wave1.segment(duration=0.01).plot()
    wave2.segment(duration=0.01).plot()
    
    corr = wave1.corr(wave2)
    print('corr =', corr)
    
    decorate(xlabel='Time (s)')


# %% [markdown]
# The following interaction plots waves with different phase offsets and prints their correlations:

# %% jupyter={"outputs_hidden": false}
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

PI2 = np.pi * 2
slider = widgets.FloatSlider(min=0, max=PI2, value=1)
interact(compute_corr, offset=slider);

# %% [markdown]
# Finally, we can plot correlation as a function of offset:

# %% jupyter={"outputs_hidden": false}
offsets = np.linspace(0, PI2, 101)

corrs = []
for offset in offsets:
    wave2 = make_sine(offset)
    corr = np.corrcoef(wave1.ys, wave2.ys)[0, 1]
    corrs.append(corr)
    
plt.plot(offsets, corrs)
decorate(xlabel='Offset (radians)',
         ylabel='Correlation')


# %% [markdown]
# That curve is a cosine.
#
# Next we'll compute serial correlations for different kinds of noise.

# %% jupyter={"outputs_hidden": false}
def serial_corr(wave, lag=1):
    N = len(wave)
    y1 = wave.ys[lag:]
    y2 = wave.ys[:N-lag]
    corr = np.corrcoef(y1, y2)[0, 1]
    return corr


# %% [markdown]
# We expect uncorrelated noise to be... well... uncorrelated.

# %% jupyter={"outputs_hidden": false}
from thinkdsp import UncorrelatedGaussianNoise

signal = UncorrelatedGaussianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)

# %% [markdown]
# As expected, the serial correlation is near 0.
#
# In Brownian noise, each value is the sum of the previous value and a random "step", so we expect a strong serial correlation:

# %% jupyter={"outputs_hidden": false}
from thinkdsp import BrownianNoise

signal = BrownianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)

# %% [markdown]
# In fact, the correlation is near 1.
#
# Since pink noise is between white and Brownian, we expect an intermediate correlation.

# %% jupyter={"outputs_hidden": false}
from thinkdsp import PinkNoise

signal = PinkNoise(beta=1)
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)

# %% [markdown]
# And we get one.
#
# Now we can plot serial correlation as a function of the pink noise parameter $\beta$.

# %% jupyter={"outputs_hidden": false}
np.random.seed(19)

betas = np.linspace(0, 2, 21)
corrs = []

for beta in betas:
    signal =  PinkNoise(beta=beta)
    wave = signal.make_wave(duration=1.0, framerate=11025)
    corr = serial_corr(wave)
    corrs.append(corr)
    
plt.plot(betas, corrs)
decorate(xlabel=r'Pink noise parameter, $\beta$',
         ylabel='Serial correlation')


# %% [markdown]
# The autocorrelation function calls `serial_corr` with different values of `lag`.

# %% jupyter={"outputs_hidden": false}
def autocorr(wave):
    """Computes and plots the autocorrelation function.

    wave: Wave
    
    returns: tuple of sequences (lags, corrs)
    """
    lags = np.arange(len(wave.ys)//2)
    corrs = [serial_corr(wave, lag) for lag in lags]
    return lags, corrs


# %% [markdown]
# Now we can plot autocorrelation for pink noise with various values of $\beta$.

# %% jupyter={"outputs_hidden": false}
def plot_pink_autocorr(beta, label):
    signal = PinkNoise(beta=beta)
    wave = signal.make_wave(duration=1.0, framerate=10000)
    lags, corrs = autocorr(wave)
    plt.plot(lags, corrs, label=label)


# %% jupyter={"outputs_hidden": false}
np.random.seed(19)

for beta in [1.7, 1.0, 0.3]:
    label = r'$\beta$ = %.1f' % beta
    plot_pink_autocorr(beta, label)

decorate(xlabel='Lag', 
         ylabel='Correlation')

# %% [markdown]
# For low values of $\beta$, the autocorrelation function drops off quickly.  As $\beta$ increases, pink noise shows more long range dependency.

# %% [markdown]
# Now let's investigate using autocorrelation for pitch tracking.  I'll load a recording of someone singing a chirp:

# %%
if not os.path.exists('28042__bcjordan__voicedownbew.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/28042__bcjordan__voicedownbew.wav

# %% jupyter={"outputs_hidden": false}
from thinkdsp import read_wave

wave = read_wave('28042__bcjordan__voicedownbew.wav')
wave.normalize()
wave.make_audio()

# %% [markdown]
# The spectrum tells us what frequencies are present, but for chirps, the frequency components are blurred over a range:

# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# The spectrogram gives a better picture of how the components vary over time:

# %% jupyter={"outputs_hidden": false}
spectro = wave.make_spectrogram(seg_length=1024)
spectro.plot(high=4200)
decorate(xlabel='Time (s)', 
                 ylabel='Frequency (Hz)')

# %% [markdown]
# We can see the fundamental frequency clearly, starting near 500 Hz and dropping.  Some of the harmonics are also visible.
#
# To track the fundamental frequency, we can take a short window:

# %% jupyter={"outputs_hidden": false}
duration = 0.01
segment = wave.segment(start=0.2, duration=duration)
segment.plot()
decorate(xlabel='Time (s)')

# %% jupyter={"outputs_hidden": false}
spectrum = segment.make_spectrum()
spectrum.plot(high=1000)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# The spectrum shows a clear peak near 400 Hz, but we can't get an very accurate estimate of frequency, partly because the peak is blurry, and partly because even if it were a perfect spike, the frequency resolution is not very good.

# %% jupyter={"outputs_hidden": false}
len(segment), segment.framerate, spectrum.freq_res


# %% [markdown]
# Each element of the spectrum spans a range of 100 Hz, so we can't get an accurate estimate of the fundamental frequency.  
#
# For signals that are at least approximately periodic, we can do better by estimating the length of the period.
#
# The following function plots the segment, and a shifted version of the segment, and computes the correlation between them:

# %% jupyter={"outputs_hidden": false}
def plot_shifted(wave, offset=0.001, start=0.2):
    segment1 = wave.segment(start=start, duration=0.01)
    segment1.plot(linewidth=2, alpha=0.8)

    # start earlier and then shift times to line up
    segment2 = wave.segment(start=start-offset, duration=0.01)
    segment2.shift(offset)
    segment2.plot(linewidth=2, alpha=0.4)

    corr = segment1.corr(segment2)
    text = r'$\rho =$ %.2g' % corr
    plt.text(segment1.start+0.0005, -0.8, text)
    decorate(xlabel='Time (s)')

plot_shifted(wave, 0.0001)

# %% [markdown]
# With a small shift the segments are still moderately correlated.  As the shift increases, the correlation falls for a while, then rises again, peaking when the shift equals the period of the signal.
#
# You can use the following interaction to search for the shift that maximizes correlation:

# %% jupyter={"outputs_hidden": false}
end = 0.004
slider1 = widgets.FloatSlider(min=0, max=end, step=end/40, value=0)
slider2 = widgets.FloatSlider(min=0.1, max=0.5, step=0.05, value=0.2)
interact(plot_shifted, wave=fixed(wave), offset=slider1, start=slider2);

# %% [markdown]
# The `autocorr` function automates this process by computing the correlation for each possible lag, up to half the length of the wave.

# %% [markdown]
# The following figure shows this autocorrelation as a function of lag:

# %% jupyter={"outputs_hidden": false}
wave = read_wave('28042__bcjordan__voicedownbew.wav')
wave.normalize()
duration = 0.01
segment = wave.segment(start=0.2, duration=duration)

# %% jupyter={"outputs_hidden": false}
lags, corrs = autocorr(segment)
plt.plot(lags, corrs)
decorate(xlabel='Lag (index)', ylabel='Correlation')

# %% [markdown]
# The first peak (other than 0) is near lag=100.
#
# We can use `argmax` to find the index of that peak:

# %% jupyter={"outputs_hidden": false}
low, high = 90, 110
lag = np.array(corrs[low:high]).argmax() + low
lag

# %% [markdown]
# We can convert from an index to a time in seconds:

# %% jupyter={"outputs_hidden": false}
period = lag / segment.framerate
period

# %% [markdown]
# Given the period in seconds, we can compute frequency:

# %% jupyter={"outputs_hidden": false}
frequency = 1 / period
frequency

# %% [markdown]
# This should be a better estimate of the fundamental frequency.  We can approximate the resolution of this estimate by computing how much we would be off by if the index were off by 1:

# %% jupyter={"outputs_hidden": false}
segment.framerate / 102, segment.framerate / 100

# %% [markdown]
# The range is less than 10 Hz.
#
# The function I wrote to compute autocorrelations is slow; `np.correlate` is much faster.

# %% jupyter={"outputs_hidden": false}
N = len(segment)
corrs2 = np.correlate(segment.ys, segment.ys, mode='same')
lags = np.arange(-N//2, N//2)
plt.plot(lags, corrs2)
decorate(xlabel='Lag', ylabel='Correlation')

# %% [markdown]
# `np.correlate` computes correlations for positive and negative lags, so lag=0 is in the middle.  For our purposes, we only care about positive lags.
#
# Also, `np.correlate` doesn't correct for the fact that the number of overlapping elements changes as the lag increases.
#
# The following code selects the second half of the results and corrects for the length of the overlap:

# %% jupyter={"outputs_hidden": false}
N = len(corrs2)
lengths = range(N, N//2, -1)

half = corrs2[N//2:].copy()
half /= lengths
half /= half[0]
plt.plot(half)
decorate(xlabel='Lag', ylabel='Correlation')

# %% [markdown]
# Now the result is similar to what we computed before.
#
# If we plot the results computed by NumPy and my implementation, they are visually similar.  They are not quite identical because my version and theirs are normalized differently.

# %% jupyter={"outputs_hidden": false}
plt.plot(half)
plt.plot(corrs)
decorate(xlabel='Lag', ylabel='Correlation')

# %% [markdown]
# The difference between the NumPy implementation and mine is less than 0.02 over most of the range.

# %% jupyter={"outputs_hidden": false}
diff = corrs - half[:-1]
plt.plot(diff)
decorate(xlabel='Lag', ylabel='Difference in correlation')

# %% jupyter={"outputs_hidden": false}
