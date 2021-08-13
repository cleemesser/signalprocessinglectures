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
# This notebook contains code examples from Chapter 3: Non-periodic signals
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
# ### Chirp
#
# Make a linear chirp from A3 to A5.

# %%
from thinkdsp import Chirp

signal = Chirp(start=220, end=880)
wave1 = signal.make_wave(duration=2)
wave1.make_audio()

# %% [markdown]
# Here's what the waveform looks like near the beginning.

# %%
wave1.segment(start=0, duration=0.01).plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# And near the end.

# %%
wave1.segment(start=0.9, duration=0.01).plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Here's an exponential chirp with the same frequency range and duration.

# %%
from thinkdsp import ExpoChirp

signal = ExpoChirp(start=220, end=880)
wave2 = signal.make_wave(duration=2)
wave2.make_audio()

# %% [markdown]
# ## Leakage
#
# Spectral leakage is when some of the energy at one frequency appears at another frequency (usually nearby).
#
# Let's look at the effect of leakage on a sine signal (which only contains one frequency component).

# %%
from thinkdsp import SinSignal

signal = SinSignal(freq=440)

# %% [markdown]
# If the duration is an integer multiple of the period, the beginning and end of the segment line up, and we get minimal leakage.

# %%
duration = signal.period * 30
wave = signal.make_wave(duration)
wave.plot()
decorate(xlabel='Time (s)')

# %%
spectrum = wave.make_spectrum()
spectrum.plot(high=880)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')

# %% [markdown]
# If the duration is not a multiple of a period, the leakage is pretty bad.

# %%
duration = signal.period * 30.25
wave = signal.make_wave(duration)
wave.plot()
decorate(xlabel='Time (s)')

# %%
spectrum = wave.make_spectrum()
spectrum.plot(high=880)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# Windowing helps (but notice that it reduces the total energy).

# %%
wave.hamming()
spectrum = wave.make_spectrum()
spectrum.plot(high=880)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# ## Spectrogram
#
# If you blindly compute the DFT of a non-periodic segment, you get "motion blur".

# %%
signal = Chirp(start=220, end=440)
wave = signal.make_wave(duration=1)
spectrum = wave.make_spectrum()
spectrum.plot(high=700)
decorate(xlabel='Frequency (Hz)')


# %% [markdown]
# A spectrogram is a visualization of a short-time DFT that lets you see how the spectrum varies over time.

# %%
def plot_spectrogram(wave, seg_length):
    """
    """
    spectrogram = wave.make_spectrogram(seg_length)
    print('Time resolution (s)', spectrogram.time_res)
    print('Frequency resolution (Hz)', spectrogram.freq_res)
    spectrogram.plot(high=700)
    decorate(xlabel='Time(s)', ylabel='Frequency (Hz)')


# %%
signal = Chirp(start=220, end=440)
wave = signal.make_wave(duration=1, framerate=11025)
plot_spectrogram(wave, 512)

# %% [markdown]
# If you increase the segment length, you get better frequency resolution, worse time resolution.

# %%
plot_spectrogram(wave, 1024)

# %% [markdown]
# If you decrease the segment length, you get better time resolution, worse frequency resolution.

# %%
plot_spectrogram(wave, 256)

# %%
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

slider = widgets.IntSlider(min=128, max=4096, value=100, step=128)
interact(plot_spectrogram, wave=fixed(wave), seg_length=slider);


# %% [markdown]
# ## Spectrum of a chirp
#
# The following interaction lets you customize the Eye of Sauron as you vary the start and end frequency of the chirp.

# %%
def eye_of_sauron(start, end):
    """Plots the spectrum of a chirp.
    
    start: initial frequency
    end: final frequency
    """
    signal =  Chirp(start=start, end=end)
    wave = signal.make_wave(duration=0.5)
    spectrum = wave.make_spectrum()
    
    spectrum.plot(high=1200)
    decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')


# %%
slider1 = widgets.FloatSlider(min=100, max=1000, value=100, step=50)
slider2 = widgets.FloatSlider(min=100, max=1000, value=200, step=50)
interact(eye_of_sauron, start=slider1, end=slider2);

# %% jupyter={"outputs_hidden": true}
