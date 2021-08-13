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
# This notebook contains solutions to exercises in Chapter 3: Non-periodic signals
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
# ### Exercise 1
#
# Run and listen to the examples in chap03.ipynb.  In the leakage example, try replacing the Hamming window with one of the other windows provided by NumPy, and see what effect they have on leakage. 
#
# See http://docs.scipy.org/doc/numpy/reference/routines.window.html
#
#
# ### Solution
#
# Here's the leakage example:

# %%
from thinkdsp import SinSignal

signal = SinSignal(freq=440)
duration = signal.period * 30.25
wave = signal.make_wave(duration)
spectrum = wave.make_spectrum()

# %%
spectrum.plot(high=880)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# The following figure shows the effect of 4 different windows.

# %%
for window_func in [np.bartlett, np.blackman, np.hamming, np.hanning]:
    wave = signal.make_wave(duration)
    wave.ys *= window_func(len(wave.ys))

    spectrum = wave.make_spectrum()
    spectrum.plot(high=880, label=window_func.__name__)

decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# All four do a good job of reducing leakage.  The Bartlett filter leaves some residual "ringing".  The Hamming filter dissipates the least amount of energy. 

# %% [markdown]
# ### Exercise 2
#
# Write a class called `SawtoothChirp` that extends `Chirp` and overrides evaluate to generate a sawtooth waveform with frequency that increases (or decreases) linearly.

# %%
from thinkdsp import Chirp
from thinkdsp import normalize, unbias

PI2 = 2 * np.pi

class SawtoothChirp(Chirp):
    """Represents a sawtooth signal with varying frequency."""

    def evaluate(self, ts):
        """Helper function that evaluates the signal.

        ts: float array of times
        """
        freqs = np.linspace(self.start, self.end, len(ts))
        dts = np.diff(ts, prepend=0)
        dphis = PI2 * freqs * dts
        phases = np.cumsum(dphis)
        cycles = phases / PI2
        frac, _ = np.modf(cycles)
        ys =  normalize(unbias(frac), self.amp)
        return ys


# %% [markdown]
# Here's what it sounds like.

# %%
signal = SawtoothChirp(start=220, end=880)
wave = signal.make_wave(duration=1, framerate=4000)
wave.apodize()
wave.make_audio()

# %% [markdown]
# And here's the spectrogram.

# %%
sp = wave.make_spectrogram(256)
sp.plot()
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

# %% [markdown]
# At a relatively low frame rate, you can see the aliased harmonics bouncing off the folding frequency.  And you can hear them as a background hiss.  If you crank up the frame rate, they go away.
#
# By the way, if you are a fan of the original Star Trek series, you might recognize the sawtooth chirp as the red alert signal:

# %%
if not os.path.exists('tos-redalert.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/tos-redalert.wav

# %%
from thinkdsp import read_wave

read_wave('tos-redalert.wav').make_audio()

# %% [markdown]
# ### Exercise 3
#
# Make a sawtooth chirp that sweeps from 2500 to 3000 Hz, then
# make a wave with duration 1 and framerate 20 kHz.  Draw a sketch of
# what you think the spectrum will look like.  Then plot the
# spectrum and see if you got it right.
#
# ### Solution
#
# Since the fundamental sweeps from 2500 to 3000 Hz, I expect to see something like the
# Eye of Sauron in that range.  The first harmonic sweeps from 5000 to 6000 Hz, so I expect a shorter tower in that range, like the Outhouse of Sauron.  The second harmonic sweeps from 7500 to 9000 Hz, so I expect something even shorter in that range, like the Patio of Sauron.
#
# The other harmonics get aliased all over the place, so I expect to see some energy at all other frequencies.  This distributed energy creates some interesting sounds.

# %%
signal = SawtoothChirp(start=2500, end=3000)
wave = signal.make_wave(duration=1, framerate=20000)
wave.make_audio()

# %%
wave.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# ### Exercise 4
#
# In musical terminology, a “glissando” is a note that slides from one pitch to another, so it is similar to a chirp. Find or make a recording of a glissando and plot its spectrogram.  
#
# One suggestion: George Gershwin's *Rhapsody in Blue* starts with a famous clarinet glissando; you can download a recording from http://archive.org/details/rhapblue11924.

# %%
if not os.path.exists('72475__rockwehrmann__glissup02.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/72475__rockwehrmann__glissup02.wav

# %%
wave = read_wave('72475__rockwehrmann__glissup02.wav')
wave.make_audio()

# %%
wave.make_spectrogram(512).plot(high=5000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')


# %% [markdown]
# ### Exercise 5
#
# A trombone player can play a glissando by extending the trombone slide while blowing continuously. As the slide extends, the total length of the tube gets longer, and the resulting pitch is inversely proportional to length.
# Assuming that the player moves the slide at a constant speed, how
# does frequency vary with time?  
#
# Write a class called `TromboneGliss` that extends `Chirp` and
# provides `evaluate`.  Make a wave that simulates a trombone
# glissando from F3 down to C3 and back up to F3.  C3 is 262 Hz; F3 is
# 349 Hz.
#
# Plot a spectrogram of the resulting wave.  Is a trombone glissando
# more like a linear or exponential chirp?

# %%
class TromboneGliss(Chirp):
    """Represents a trombone-like signal with varying frequency."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        l1, l2 = 1.0 / self.start, 1.0 / self.end
        lengths = np.linspace(l1, l2, len(ts))
        freqs = 1 / lengths
        
        dts = np.diff(ts, prepend=0)
        dphis = PI2 * freqs * dts
        phases = np.cumsum(dphis)
        ys = self.amp * np.cos(phases)
        return ys


# %% [markdown]
# Here's the first part of the wave:

# %%
low = 262
high = 349
signal = TromboneGliss(high, low)
wave1 = signal.make_wave(duration=1)
wave1.apodize()
wave1.make_audio()

# %% [markdown]
# And the second part:

# %%
signal = TromboneGliss(low, high)
wave2 = signal.make_wave(duration=1)
wave2.apodize()
wave2.make_audio()

# %% [markdown]
# Putting them together:

# %%
wave = wave1 | wave2
wave.make_audio()

# %% [markdown]
# Here's the spectrogram:

# %%
sp = wave.make_spectrogram(1024)
sp.plot(high=1000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

# %% [markdown]
# ### Exercise 6
#
# Make or find a recording of a series of vowel sounds and look at the
# spectrogram.  Can you identify different vowels?

# %%
if not os.path.exists('87778__marcgascon7__vocals.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/87778__marcgascon7__vocals.wav

# %%
wave = read_wave('87778__marcgascon7__vocals.wav')
wave.make_audio()

# %%
wave.make_spectrogram(1024).plot(high=1000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

# %% [markdown]
# The stripe across the bottom is probably background noise.  The peaks in the spectrogram are called "formants".
#
# In general, vowel sounds are distinguished by the amplitude ratios of the first two formants relative to the fundamental.  For more, see https://en.wikipedia.org/wiki/Formant
#
# We can see the formats more clearly by selecting a segment during 'ah'.

# %%
high = 1000

segment = wave.segment(start=1, duration=0.25)
segment.make_spectrum().plot(high=high)

# %% [markdown]
# The fundamental is near 100 Hz.  The next highest peaks are at 200 Hz and 700 Hz.  People who know more about this than I do can identify vowels by looking at spectrums, but I can't.

# %% [markdown]
# The 'eh' segment has a high-amplitude formant near 500 Hz.

# %%
segment = wave.segment(start=2.2, duration=0.25)
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# The 'ih' segment has no high frequency components.

# %%
segment = wave.segment(start=3.5, duration=0.25)
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# The 'oh' segment has a high-amplitude formant near 500 Hz, even higher than the fundamental.

# %%
segment = wave.segment(start=5.1, duration=0.25)
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# The 'oo' segment has a high-amplitude formant near 300 Hz and no high-frequency components

# %%
segment = wave.segment(start=6.5, duration=0.25)
segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')

# %% jupyter={"outputs_hidden": true}
