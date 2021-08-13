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
# This notebook contains solutions to exercises in Chapter 11: Modulation and sampling
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
# As we have seen, if you sample a signal at too low a
# framerate, frequencies above the folding frequency get aliased.
# Once that happens, it is no longer possible to filter out
# these components, because they are indistinguishable from
# lower frequencies.
#
# It is a good idea to filter out these frequencies *before*
# sampling; a low-pass filter used for this purpose is called
# an ``anti-aliasing filter''.
#
# Returning to the drum solo example, apply a low-pass filter
# before sampling, then apply the low-pass filter again to remove
# the spectral copies introduced by sampling.  The result should
# be identical to the filtered signal.
#

# %% [markdown]
# *Solution:*  I'll load the drum solo again.

# %%
if not os.path.exists('263868__kevcio__amen-break-a-160-bpm.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/263868__kevcio__amen-break-a-160-bpm.wav

# %% jupyter={"outputs_hidden": false}
from thinkdsp import read_wave

wave = read_wave('263868__kevcio__amen-break-a-160-bpm.wav')
wave.normalize()
wave.plot()

# %% [markdown]
# This signal is sampled at 44100 Hz.  Here's what it sounds like.

# %% jupyter={"outputs_hidden": false}
wave.make_audio()

# %% [markdown]
# And here's the spectrum:

# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum(full=True)
spectrum.plot()

# %% [markdown]
# I'll reduce the sampling rate by a factor of 3 (but you can change this to try other values):

# %% jupyter={"outputs_hidden": false}
factor = 3
framerate = wave.framerate / factor
cutoff = framerate / 2 - 1

# %% [markdown]
# Before sampling we apply an anti-aliasing filter to remove frequencies above the new folding frequency, which is `framerate/2`:

# %% jupyter={"outputs_hidden": false}
spectrum.low_pass(cutoff)
spectrum.plot()

# %% [markdown]
# Here's what it sounds like after filtering (still pretty good).

# %% jupyter={"outputs_hidden": false}
filtered = spectrum.make_wave()
filtered.make_audio()

# %% [markdown]
# Here's the function that simulates the sampling process:

# %%
from thinkdsp import Wave

def sample(wave, factor):
    """Simulates sampling of a wave.
    
    wave: Wave object
    factor: ratio of the new framerate to the original
    """
    ys = np.zeros(len(wave))
    ys[::factor] = np.real(wave.ys[::factor])
    return Wave(ys, framerate=wave.framerate) 


# %% [markdown]
# The result contains copies of the spectrum near 20 kHz; they are not very noticeable:

# %% jupyter={"outputs_hidden": false}
sampled = sample(filtered, factor)
sampled.make_audio()

# %% [markdown]
# But they show up when we plot the spectrum:

# %% jupyter={"outputs_hidden": false}
sampled_spectrum = sampled.make_spectrum(full=True)
sampled_spectrum.plot()

# %% [markdown]
# We can get rid of the spectral copies by applying the anti-aliasing filter again:

# %% jupyter={"outputs_hidden": false}
sampled_spectrum.low_pass(cutoff)
sampled_spectrum.plot()

# %% [markdown]
# We just lost half the energy in the spectrum, but we can scale the result to get it back:

# %% jupyter={"outputs_hidden": false}
sampled_spectrum.scale(factor)
spectrum.plot()
sampled_spectrum.plot()

# %% [markdown]
# Now the difference between the spectrum before and after sampling should be small.

# %% jupyter={"outputs_hidden": false}
spectrum.max_diff(sampled_spectrum)

# %% [markdown]
# After filtering and scaling, we can convert back to a wave:

# %% jupyter={"outputs_hidden": false}
interpolated = sampled_spectrum.make_wave()
interpolated.make_audio()

# %% [markdown]
# And the difference between the interpolated wave and the filtered wave should be small.

# %% jupyter={"outputs_hidden": false}
filtered.plot()
interpolated.plot()

# %% jupyter={"outputs_hidden": false}
filtered.max_diff(interpolated)

# %% jupyter={"outputs_hidden": true}
