# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: pyt19
#     language: python
#     name: pyt19
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

# %% [markdown]
# ## Think DSP module
#
# `thinkdsp` is a module that accompanies _Think DSP_ and provides classes and functions for working with signals.
#
# [Documentation of the thinkdsp module is here](http://greenteapress.com/thinkdsp.html). 

# %%
# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py

# %% [markdown]
# ## Signals
#
# Instantiate cosine and sine signals.

# %%
from thinkdsp import CosSignal, SinSignal

cos_sig = CosSignal(freq=440, amp=1.0, offset=0)
sin_sig = SinSignal(freq=880, amp=0.5, offset=0)

# %% [markdown]
# Plot the sine and cosine signals.  By default, `plot` plots three periods.  

# %%
from thinkdsp import decorate

cos_sig.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Here's the sine signal.

# %%
sin_sig.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Notice that the frequency of the sine signal is doubled, so the period is halved.
#
# The sum of two signals is a SumSignal.

# %%
mix = sin_sig + cos_sig
mix

# %% [markdown]
# Here's what it looks like.

# %%
mix.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# ## Waves
#
# A Signal represents a mathematical function defined for all values of time.  If you evaluate a signal at a sequence of equally-spaced times, the result is a Wave.  `framerate` is the number of samples per second.

# %%
wave = mix.make_wave(duration=0.5, start=0, framerate=11025)
wave

# %% [markdown]
# IPython provides an Audio widget that can play a wave.

# %%
from IPython.display import Audio
audio = Audio(data=wave.ys, rate=wave.framerate)
audio

# %% [markdown]
# Wave also provides `make_audio()`, which does the same thing:

# %%
wave.make_audio()

# %% [markdown]
# The `ys` attribute is a NumPy array that contains the values from the signal.  The interval between samples is the inverse of the framerate.

# %%
print('Number of samples', len(wave.ys))
print('Timestep in ms', 1 / wave.framerate * 1000)

# %% [markdown]
# Signal objects that represent periodic signals have a `period` attribute.
#
# Wave provides `segment`, which creates a new wave.  So we can pull out a 3 period segment of this wave.

# %%
period = mix.period
segment = wave.segment(start=0, duration=period*3)
period

# %% [markdown]
# Wave provides `plot`

# %%
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# `normalize` scales a wave so the range doesn't exceed -1 to 1.
#
# `apodize` tapers the beginning and end of the wave so it doesn't click when you play it.

# %%
wave.normalize()
wave.apodize()
wave.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# You can write a wave to a WAV file.

# %%
wave.write('temp.wav')

# %% [markdown]
# `wave.write` writes the wave to a file so it can be used by an exernal player.

# %%
from thinkdsp import play_wave

play_wave(filename='temp.wav', player='aplay')

# %% [markdown]
# `read_wave` reads WAV files.  The WAV examples in the book are from freesound.org.  In the contributors section of the book, I list and thank the people who uploaded the sounds I use.

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
decorate(xlabel='Time (s)')

# %% [markdown]
# ## Spectrums
#
# Wave provides `make_spectrum`, which computes the spectrum of the wave.

# %%
spectrum = segment.make_spectrum()

# %% [markdown]
# Spectrum provides `plot`

# %%
spectrum.plot()
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# The frequency components above 10 kHz are small.  We can see the lower frequencies more clearly by providing an upper bound:

# %%
spectrum.plot(high=10000)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# Spectrum provides `low_pass`, which applies a low pass filter; that is, it attenuates all frequency components above a cutoff frequency.

# %%
spectrum.low_pass(3000)

# %% [markdown]
# The result is a spectrum with fewer components.

# %%
spectrum.plot(high=10000)
decorate(xlabel='Frequency (Hz)')

# %% [markdown]
# We can convert the filtered spectrum back to a wave:

# %%
filtered = spectrum.make_wave()

# %% [markdown]
# And then normalize it to the range -1 to 1.

# %%
filtered.normalize()

# %% [markdown]
# Before playing it back, I'll apodize it (to avoid clicks).

# %%
filtered.apodize()
filtered.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# And I'll do the same with the original segment.

# %%
segment.normalize()
segment.apodize()
segment.plot()
decorate(xlabel='Time (s)')

# %% [markdown]
# Finally, we can listen to the original segment and the filtered version.

# %%
segment.make_audio()

# %%
filtered.make_audio()

# %% [markdown]
# The original sounds more complex, with some high-frequency components that sound buzzy.
# The filtered version sounds more like a pure tone, with a more muffled quality.  The cutoff frequency I chose, 3000 Hz, is similar to the quality of a telephone line, so this example simulates the sound of a violin recording played over a telephone.

# %% [markdown]
# ## Interaction
#
# The following example shows how to use interactive IPython widgets.

# %%
import matplotlib.pyplot as plt
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

    spectrum.plot(color='0.7')
    spectrum.low_pass(cutoff)
    spectrum.plot(color='#045a8d')
    decorate(xlabel='Frequency (Hz)')
    plt.show()
    
    audio = spectrum.make_wave().make_audio()
    display(audio)


# %% [markdown]
# Adjust the sliders to control the start and duration of the segment and the cutoff frequency applied to the spectrum.

# %%
from ipywidgets import interact, fixed

wave = read_wave('92002__jcveliz__violin-origional.wav')
interact(filter_wave, wave=fixed(wave), 
         start=(0, 5, 0.1), duration=(0, 5, 0.1), cutoff=(0, 10000, 100));

# %%

# %%
