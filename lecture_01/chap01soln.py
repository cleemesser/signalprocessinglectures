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
# This notebook contains solutions to exercises in Chapter 1: Sounds and Signals
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
# ### Exercise 1
#
# Go to http://freesound.org and download a sound sample that
# includes music, speech, or other sounds that have a well-defined pitch.
# Select a roughly half-second segment where the pitch is
# constant.  Compute and plot the spectrum of the segment you selected.
# What connection can you make between the timbre of the sound and the
# harmonic structure you see in the spectrum?
#
# Use `high_pass`, `low_pass`, and `band_stop` to
# filter out some of the harmonics.  Then convert the spectrum back
# to a wave and listen to it.  How does the sound relate to the
# changes you made in the spectrum?

# %% [markdown]
# ### Solution
#
# I chose this recording (or synthesis?) of a trumpet section http://www.freesound.org/people/Dublie/sounds/170255/
#
# As always, thanks to the people who contributed these recordings!

# %%
if not os.path.exists('170255__dublie__trumpet.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/170255__dublie__trumpet.wav

# %%
from thinkdsp import read_wave

wave = read_wave('170255__dublie__trumpet.wav')
wave.normalize()
wave.make_audio()

# %% [markdown]
# Here's what the whole wave looks like:

# %%
wave.plot()

# %% [markdown]
# By trial and error, I selected a segment with a constant pitch (although I believe it is a chord played by at least two horns).

# %%
segment = wave.segment(start=1.1, duration=0.3)
segment.make_audio()

# %% [markdown]
# Here's what the segment looks like:
#

# %%
segment.plot()

# %% [markdown]
# And here's an even shorter segment so you can see the waveform:

# %%
segment.segment(start=1.1, duration=0.005).plot()

# %% [markdown]
# Here's what the spectrum looks like:

# %%
spectrum = segment.make_spectrum()
spectrum.plot(high=7000)

# %% [markdown]
# It has lots of frequency components.  Let's zoom in on the fundamental and dominant frequencies:

# %%
spectrum = segment.make_spectrum()
spectrum.plot(high=1000)

# %% [markdown]
# `peaks` prints the highest points in the spectrum and their frequencies, in descending order:

# %%
spectrum.peaks()[:30]

# %% [markdown]
# The dominant peak is at 870 Hz.  It's not easy to dig out the fundamental, but with peaks at 507, 347, and 253 Hz, we can infer a fundamental at roughly 85 Hz, with harmonics at 170, 255, 340, 425, and 510 Hz.
#
# 85 Hz is close to F2 at 87 Hz.  The pitch we perceive is usually the fundamental, even when it is not dominant.  When you listen to this segment, what pitch(es) do you perceive?
#
# Next we can filter out the high frequencies:

# %%
spectrum.low_pass(2000)

# %% [markdown]
# And here's what it sounds like:

# %%
spectrum.make_wave().make_audio()

# %% [markdown]
# The following interaction allows you to select a segment and apply different filters.  If you set the cutoff to 3400 Hz, you can simulate what the sample would sound like over an old (not digital) phone line.

# %%
from thinkdsp import decorate
from IPython.display import display

def filter_wave(wave, start, duration, cutoff):
    """Selects a segment from the wave and filters it.
    
    Plots the spectrum and displays an Audio widget.
    
    wave: Wave object
    start: time in s
    duration: time in s
    cutoff: frequency in Hz
    """
    segment = wave.segment(start, duration)
    spectrum = segment.make_spectrum()

    spectrum.plot(high=5000, color='0.7')
    spectrum.low_pass(cutoff)
    spectrum.plot(high=5000, color='#045a8d')
    decorate(xlabel='Frequency (Hz)')
    
    audio = spectrum.make_wave().make_audio()
    display(audio)


# %%
from ipywidgets import interact, fixed

interact(filter_wave, wave=fixed(wave), 
         start=(0, 5, 0.1), duration=(0, 5, 0.1), cutoff=(0, 5000, 100));

# %% [markdown]
# ### Exercise 2
#
# Synthesize a compound signal by creating SinSignal and CosSignal
# objects and adding them up.  Evaluate the signal to get a Wave,
# and listen to it.  Compute its Spectrum and plot it.
# What happens if you add frequency
# components that are not multiples of the fundamental?

# %% [markdown]
# ### Solution
#
# Here are some arbitrary components I chose.  It makes an interesting waveform!

# %%
from thinkdsp import SinSignal

signal = (SinSignal(freq=400, amp=1.0) +
          SinSignal(freq=600, amp=0.5) +
          SinSignal(freq=800, amp=0.25))
signal.plot()

# %% [markdown]
# We can use the signal to make a wave:

# %%
wave2 = signal.make_wave(duration=1)
wave2.apodize()

# %% [markdown]
# And here's what it sounds like:

# %%
wave2.make_audio()

# %% [markdown]
# The components are all multiples of 200 Hz, so they make a coherent sounding tone.
#
# Here's what the spectrum looks like:

# %%
spectrum = wave2.make_spectrum()
spectrum.plot(high=2000)

# %% [markdown]
# If we add a component that is not a multiple of 200 Hz, we hear it as a distinct pitch.

# %%
signal += SinSignal(freq=450)
signal.make_wave().make_audio()

# %% [markdown]
# ### Exercise 3
#
# Write a function called `stretch` that takes a Wave and a stretch factor and speeds up or slows down the wave by modifying `ts` and `framerate`.  Hint: it should only take two lines of code.

# %% [markdown]
# ### Solution
#
# I'll use the trumpet example again:

# %%
wave3 = read_wave('170255__dublie__trumpet.wav')
wave3.normalize()
wave3.make_audio()


# %% [markdown]
# Here's my implementation of `stretch`

# %%
def stretch(wave, factor):
    wave.ts *= factor
    wave.framerate /= factor


# %% [markdown]
# And here's what it sounds like if we speed it up by a factor of 2.

# %%
stretch(wave3, 0.5)
wave3.make_audio()

# %% [markdown]
# Here's what it looks like (to confirm that the `ts` got updated correctly).

# %%
wave3.plot()

# %% [markdown]
# I think it sounds better speeded up.  In fact, I wonder if we are playing the original at the right speed.

# %%
