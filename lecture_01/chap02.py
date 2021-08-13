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
# This notebook contains code examples from Chapter 2: Harmonics
#
# Copyright 2015 Allen Downey
#
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)

# %%
# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py

# %% [markdown]
# ## Waveforms and harmonics
#
# Create a triangle signal and plot a 3 period segment.

# %%
from thinkdsp import TriangleSignal
from thinkdsp import decorate

signal = TriangleSignal(200)
duration = signal.period*3
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Make a wave and play it.

# %%
wave = signal.make_wave(duration=0.5, framerate=10000)
wave.apodize()
wave.make_audio()

# %% [markdown]
# Compute its spectrum and plot it.

# %%
spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# Make a square signal and plot a 3 period segment.

# %%
from thinkdsp import SquareSignal

signal = SquareSignal(200)
duration = signal.period*3
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Make a wave and play it.

# %%
wave = signal.make_wave(duration=0.5, framerate=10000)
wave.apodize()
wave.make_audio()

# %% [markdown]
# Compute its spectrum and plot it.

# %%
spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# Create a sawtooth signal and plot a 3 period segment.

# %%
from thinkdsp import SawtoothSignal

signal = SawtoothSignal(200)
duration = signal.period*3
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Make a wave and play it.

# %%
wave = signal.make_wave(duration=0.5, framerate=10000)
wave.apodize()
wave.make_audio()

# %% [markdown]
# Compute its spectrum and plot it.

# %%
spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# ### Aliasing
#
# Make a cosine signal at 4500 Hz, make a wave at framerate 10 kHz, and plot 5 periods.

# %%
from thinkdsp import CosSignal

signal = CosSignal(4500)
duration = signal.period*5
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Make a cosine signal at 5500 Hz, make a wave at framerate 10 kHz, and plot the same duration.
#
# With framerate 10 kHz, the folding frequency is 5 kHz, so a 4500 Hz signal and a 5500 Hz signal look exactly the same.

# %%
signal = CosSignal(5500)
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Make a triangle signal and plot the spectrum.  See how the harmonics get folded.

# %%
signal = TriangleSignal(1100)
segment = signal.make_wave(duration=0.5, framerate=10000)
spectrum = segment.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# ## Amplitude and phase
#
# Make a sawtooth wave.

# %%
signal = SawtoothSignal(500)
wave = signal.make_wave(duration=1, framerate=10000)
segment = wave.segment(duration=0.005)
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Play it.

# %%
wave.make_audio()

# %% [markdown]
# Extract the wave array and compute the real FFT (which is just an FFT optimized for real inputs).

# %%
import numpy as np

hs = np.fft.rfft(wave.ys)
hs

# %% [markdown]
# Compute the frequencies that match up with the elements of the FFT.

# %%
n = len(wave.ys)                 # number of samples
d = 1 / wave.framerate           # time between samples
fs = np.fft.rfftfreq(n, d)
fs

# %% [markdown]
# Plot the magnitudes vs the frequencies.

# %%
import matplotlib.pyplot as plt

magnitude = np.absolute(hs)
plt.plot(fs, magnitude)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# Plot the phases vs the frequencies.

# %%
angle = np.angle(hs)
plt.plot(fs, angle)
decorate(xlabel='Phase (radian)')

# %% [markdown]
# ## What does phase sound like?
#
# Shuffle the phases.

# %%
import random
random.shuffle(angle)
plt.plot(fs, angle)
decorate(xlabel='Phase (radian)')

# %% [markdown]
# Put the shuffled phases back into the spectrum.  Each element in `hs` is a complex number with magitude $A$ and phase $\phi$, with which we can compute $A e^{i \phi}$

# %%
i = complex(0, 1)
spectrum = wave.make_spectrum()
spectrum.hs = magnitude * np.exp(i * angle)

# %% [markdown]
# Convert the spectrum back to a wave (which uses irfft).

# %%
wave2 = spectrum.make_wave()
wave2.normalize()
segment = wave2.segment(duration=0.005)
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Play the wave with the shuffled phases.

# %%
wave2.make_audio()

# %% [markdown]
# For comparison, here's the original wave again.

# %%
wave.make_audio()


# %% [markdown]
# Although the two signals have different waveforms, they have the same frequency components with the same amplitudes.  They differ only in phase.

# %% [markdown]
# ## Aliasing interaction
#
# The following interaction explores the effect of aliasing on the harmonics of a sawtooth signal.

# %%
def view_harmonics(freq, framerate):
    """Plot the spectrum of a sawtooth signal.
    
    freq: frequency in Hz
    framerate: in frames/second
    """
    signal = SawtoothSignal(freq)
    wave = signal.make_wave(duration=0.5, framerate=framerate)
    spectrum = wave.make_spectrum()
    spectrum.plot(color='C0')
    decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
    display(wave.make_audio())


# %%
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

slider1 = widgets.FloatSlider(min=100, max=10000, value=100, step=100)
slider2 = widgets.FloatSlider(min=5000, max=40000, value=10000, step=1000)
interact(view_harmonics, freq=slider1, framerate=slider2);

# %%

# %%
