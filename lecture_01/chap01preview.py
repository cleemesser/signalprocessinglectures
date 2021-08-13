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
# This notebook contains code examples from Chapter 1: Sounds and Signals
#
# Copyright 2015 Allen Downey
#
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)
#

# %%
# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py

# %%
import matplotlib.pyplot as plt

from thinkdsp import decorate

# %% [markdown]
# ### Read a wave
#
# `read_wave` reads WAV files.  The WAV examples in the book are from freesound.org.  In the contributors section of the book, I list and thank the people who uploaded the sounds I use.

# %%
if not os.path.exists('92002__jcveliz__violin-origional.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/92002__jcveliz__violin-origional.wav

# %%
from thinkdsp import read_wave

wave = read_wave('92002__jcveliz__violin-origional.wav')

# %%
wave.make_audio()

# %% [markdown]
# I pulled out a segment of this recording where the pitch is constant.  When we plot the segment, we can't see the waveform clearly, but we can see the "envelope", which tracks the change in amplitude during the segment.

# %%
start = 1.2
duration = 0.6
segment = wave.segment(start, duration)
segment.plot()
plt.xlabel('Time (s)');

# %% [markdown]
# ### Spectrums
#
# Wave provides `make_spectrum`, which computes the spectrum of the wave.

# %%
spectrum = segment.make_spectrum()

# %% [markdown]
# Spectrum provides `plot`

# %%
spectrum.plot()
plt.xlabel('Frequency (Hz)');

# %% [markdown]
# The frequency components above 10 kHz are small.  We can see the lower frequencies more clearly by providing an upper bound:

# %%
spectrum.plot(high=10000)
plt.xlabel('Frequency (Hz)');

# %% [markdown]
# Spectrum provides `low_pass`, which applies a low pass filter; that is, it attenuates all frequency components above a cutoff frequency.

# %%
spectrum.low_pass(3000)

# %% [markdown]
# The result is a spectrum with fewer components.

# %%
spectrum.plot(high=10000)
plt.xlabel('Frequency (Hz)');

# %% [markdown]
# We can convert the filtered spectrum back to a wave:

# %%
filtered = spectrum.make_wave()

# %% [markdown]
# Now we can listen to the original segment and the filtered version.

# %%
segment.make_audio()

# %%
filtered.make_audio()


# %% [markdown]
# The original sounds more complex, with some high-frequency components that sound buzzy.
#
# The filtered version sounds more like a pure tone, with a more muffled quality.
#
# The cutoff frequency I chose, 3000 Hz, is similar to the quality of a telephone line, so this example simulates the sound of a violin recording played over a telephone.

# %% [markdown]
# ### Interaction
#
# The following shows the same example using interactive IPython widgets.

# %%
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

    spectrum.plot(color='0.7')
    spectrum.low_pass(cutoff)
    spectrum.plot()
    plt.xlabel('Frequency (Hz)');
    plt.show()
    
    audio = spectrum.make_wave().make_audio()
    display(audio)


# %% [markdown]
# Adjust the sliders to control the start and duration of the segment and the cutoff frequency applied to the spectrum.

# %%
from ipywidgets import interact, fixed
from IPython.display import display

wave = read_wave('92002__jcveliz__violin-origional.wav')
interact(filter_wave, wave=fixed(wave), 
         start=(0, 5, 0.1), duration=(0, 5, 0.1), cutoff=(0, 10000, 100));

# %%
