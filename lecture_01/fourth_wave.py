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

# %% [markdown] slideshow={"slide_type": "slide"}
# <img width="700" src="https://github.com/AllenDowney/ThinkDSP/raw/master/code/fourth_wave.png">

# %% [markdown] slideshow={"slide_type": "slide"}
# # In Search of the Fourth Wave
#
# ### Allen Downey
#
# Olin College
#
# [DSP Online Conference](https://www.dsponlineconference.com/)

# %% [markdown] slideshow={"slide_type": "skip"}
# Copyright 2020 Allen B. Downey
#
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)

# %% [markdown] slideshow={"slide_type": "slide"}
# Run this notebook
#
# [tinyurl.com/mysterywave](https://tinyurl.com/mysterywave)

# %% [markdown] slideshow={"slide_type": "slide"}
# When I was working on [*Think DSP*](https://greenteapress.com/thinkdsp), I encountered a small mystery.  
#
# As you might know:

# %% [markdown] slideshow={"slide_type": "slide"}
# * A sawtooth wave contains harmonics at integer multiples of the fundamental frequency, and
#
# * Their amplitudes drop off in proportion to $1/f$.  

# %% [markdown] slideshow={"slide_type": "slide"}
# * A square wave contains only odd multiples of the fundamental, but 
#
# * They also drop off like $1/f$.  

# %% [markdown] slideshow={"slide_type": "slide"}
# * A triangle wave also contains only odd multiples, 
#
# * But they drop off like $1/f^2$.

# %% [markdown] slideshow={"slide_type": "slide"}
# Which suggests that there's a simple waveform that 
#
# * Contains all integer multiples (like a sawtooth) and 
#
# * Drops off like $1/f^2$ (like a triangle wave).  
#
# Let's find out what it is.

# %% [markdown] slideshow={"slide_type": "slide"}
# In this talk, I'll 
#
# * Suggest four ways we can find the mysterious fourth wave.
#
# * Demonstrate using tools from SciPy, NumPy and Pandas, and
#   

# %% [markdown] slideshow={"slide_type": "slide"}
# And a tour of DSP and the topics in *Think DSP*.

# %% [markdown] slideshow={"slide_type": "slide"}
# I'm a professor of Computer Science at Olin College.
#
# Author of *Think Python*, *Think Bayes*, and *Think DSP*.
#
# And *Probably Overthinking It*, a blog about Bayesian probability and statistics.
#
# Web page: [allendowney.com](http://allendowney.com)
#
# Twitter: @allendowney

# %% [markdown] slideshow={"slide_type": "slide"}
# This talk is a Jupyter notebook.
#
# [You can read it here](https://nbviewer.jupyter.org/github/AllenDowney/ThinkDSP/blob/master/code/fourth_wave.ipynb).
#
# [And run it here](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/code/fourth_wave.ipynb).

# %% [markdown] slideshow={"slide_type": "slide"}
# Here are the libraries we'll use.

# %% slideshow={"slide_type": "-"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown] slideshow={"slide_type": "skip"}
# And a convenience function for decorating figures.

# %% slideshow={"slide_type": "skip"}
def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like

    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties

    https://matplotlib.org/api/axes_api.html

    In addition, you can use `legend=False` to suppress the legend.

    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')
    """
    plt.gca().set(**options)
    plt.tight_layout()


# %% slideshow={"slide_type": "skip"}
def legend(**options):
    """Draws a legend only if there is at least one labeled item.

    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.plt.legend.html

    """
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)


# %% [markdown] slideshow={"slide_type": "skip"}
# ## Basic waveforms

# %% [markdown] slideshow={"slide_type": "slide"}
# We'll start with the basic waveforms:
#
# * sawtooth, 
#
# * triangle, and 
#
# * square.

# %% [markdown] slideshow={"slide_type": "slide"}
# Sampled at CD audio frame rate.

# %% slideshow={"slide_type": "-"}
framerate = 44100        # samples per second
dt = 1 / framerate       # seconds per sample

# %% [markdown] slideshow={"slide_type": "slide"}
# At equally-spaced times from 0 to `duration`.

# %% slideshow={"slide_type": "-"}
duration = 0.005                    # seconds
ts = np.arange(0, duration, dt)     # seconds

# %% [markdown] slideshow={"slide_type": "slide"}
# We'll work with signals at 1000 Hz.  
#
# The number of complete cycles is $f t$.

# %%
freq = 1000              # cycles per second (Hz)
cycles = freq * ts       # cycles

# %% [markdown] slideshow={"slide_type": "slide"}
# In 0.005 seconds, a 1000 Hz signal completes 5 cycles.

# %%
np.max(cycles)

# %% slideshow={"slide_type": "skip"}
plt.plot(ts, cycles)

decorate(xlabel='Time (s)',
         ylabel='Cycles',
         title='Cycles vs time')


# %% [markdown] slideshow={"slide_type": "slide"}
# `wrap` uses `modf` to compute the fraction part of the number of cycles.

# %%
def wrap(cycles):
    frac, _ = np.modf(cycles)
    return frac


# %% [markdown] slideshow={"slide_type": "slide"}
# If we apply `wrap` to `cycles`, the result is a sawtooth wave.
#
# I subtract off `0.5` so the mean of the signal is 0.

# %%
ys = wrap(cycles) - 0.5
ys.mean()

# %% [markdown] slideshow={"slide_type": "skip"}
# Here's what it looks like.

# %% slideshow={"slide_type": "slide"}
plt.plot(ts, ys)

decorate(xlabel='Time (s)',
         ylabel='Amplitude',
         title='Sawtooth wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# If we take the absolute value of `ys`, the result is a triangle wave.

# %% slideshow={"slide_type": "slide"}
plt.plot(ts, np.abs(ys))

decorate(xlabel='Time (s)',
         ylabel='Amplitude',
         title='Triangle wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# And if we take just the sign of `ys`, the result is a square wave.

# %% slideshow={"slide_type": "slide"}
plt.plot(ts, np.sign(ys))

decorate(xlabel='Time (s)',
         ylabel='Amplitude',
         title='Square wave')


# %% [markdown] slideshow={"slide_type": "slide"}
# Think of `abs` as magnitude and `sign` as direction
#
# * The triangle is the magnitude of a sawtooth.
#
# * The square wave is the direction of a sawtooth.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## One function to make them all
#
# `make_wave` contains the parts these waves have in common.

# %% slideshow={"slide_type": "slide"}
def make_wave(func, duration, freq):
    """Make a signal.
    
    func: function that takes cycles and computes ys
    duration: length of the wave in time
    """
    ts = np.arange(0, duration, dt)
    cycles = freq * ts    
    ys = func(cycles)
    
    ys = unbias(normalize(ys))
    series = pd.Series(ys, ts)
    return series


# %% [markdown] slideshow={"slide_type": "slide"}
# `normalize` scales the wave between 0 and 1,
#
# `unbias` shifts it so the mean is 0.

# %%
def normalize(ys):
    low, high = np.min(ys), np.max(ys)
    return (ys - low) / (high - low)

def unbias(ys):
    return ys - ys.mean()


# %% [markdown] slideshow={"slide_type": "slide"}
# Why use a `Series`?
#
# `Series` is like two arrays:
#
# * An index, and
#
# * Corresponding value.

# %% [markdown] slideshow={"slide_type": "slide"}
# Natural representation of correspondence:
#
# * From time to amplitude.
#
# * From frequency to complex amplitude.

# %% [markdown] slideshow={"slide_type": "slide"}
# We'll use `plot_wave` to plot a short segment of a wave.

# %%
def plot_wave(wave, title=''):
    segment = wave[0:0.01]
    segment.plot()
    decorate(xlabel='Time (s)',
             ylabel='Amplitude',
             title=title)


# %% [markdown] slideshow={"slide_type": "slide"}
# Now we can define `sawtooth_func` to compute the sawtooth wave.

# %%
def sawtooth_func(cycles):
    ys = wrap(cycles) - 0.5
    return ys


# %% [markdown] slideshow={"slide_type": "slide"}
# And pass it as an argument to `make_wave`:

# %% slideshow={"slide_type": "slide"}
sawtooth_wave = make_wave(sawtooth_func, duration=0.01, freq=500)
plot_wave(sawtooth_wave, title='Sawtooth wave')


# %% [markdown] slideshow={"slide_type": "slide"}
# Same with `triangle_func`.

# %% slideshow={"slide_type": "-"}
def triangle_func(cycles):
    ys = wrap(cycles) - 0.5
    return np.abs(ys)


# %% slideshow={"slide_type": "slide"}
triangle_wave = make_wave(triangle_func, duration=0.01, freq=500)
plot_wave(triangle_wave, title='Triangle wave')


# %% [markdown] slideshow={"slide_type": "slide"}
# And `square_func`.

# %% slideshow={"slide_type": "-"}
def square_func(cycles):
    ys = wrap(cycles) - 0.5
    return np.sign(ys)


# %% slideshow={"slide_type": "slide"}
square_wave = make_wave(square_func, duration=0.01, freq=500)
plot_wave(square_wave, title='Square wave')


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Spectrum
#
# Now let's see what the spectrums look like for these waves.
#

# %% slideshow={"slide_type": "slide"}
def make_spectrum(wave):
    n = len(wave)
    hs = np.fft.rfft(wave)         # amplitudes
    fs = np.fft.rfftfreq(n, dt)    # frequencies

    series = pd.Series(hs, fs)
    return series


# %% [markdown] slideshow={"slide_type": "slide"}
# `make_spectrum` uses NumPy to compute the real FFT of the wave:
#
# * `hs` contains the complex amplitudes, and
#
# * `fs` contains the corresponding frequencies.

# %% [markdown] slideshow={"slide_type": "slide"}
# Pandas `Series` represents correspondence between frequencies and complex amplitudes.

# %% [markdown] slideshow={"slide_type": "slide"}
# Use `abs` to compute magnitude of complex amplitudes and 
# plot them as a function of `fs`:

# %%
def plot_spectrum(spectrum, title=''):
    spectrum.abs().plot()
    decorate(xlabel='Frequency (Hz)',
             ylabel='Magnitude',
             title=title)


# %% [markdown] slideshow={"slide_type": "slide"}
# I'll use a sinusoid to test `make_spectrum`.

# %%
def sinusoid_func(cycles):
    return np.cos(2 * np.pi * cycles)


# %% [markdown] slideshow={"slide_type": "slide"}
# Now we can use `make_wave` to make a sinusoid.

# %%
sinusoid_wave = make_wave(sinusoid_func, duration=0.5, freq=freq)

# %% [markdown] slideshow={"slide_type": "slide"}
# And `make_spectrum` to compute its spectrum.

# %% slideshow={"slide_type": "slide"}
sinusoid_spectrum = make_spectrum(sinusoid_wave)
plot_spectrum(sinusoid_spectrum, 
              title='Spectrum of a sinusoid wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# A sinusoid only contains one frequency component.
#
# As contrasted with the spectrum of a sawtooth wave, which looks like this.

# %% slideshow={"slide_type": "slide"}
sawtooth_wave = make_wave(sawtooth_func, duration=0.5, freq=freq)
sawtooth_spectrum = make_spectrum(sawtooth_wave)
plot_spectrum(sawtooth_spectrum, 
              title='Spectrum of a sawtooth wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# The largest magnitude is at 1000 Hz, but the signal also contains components at every integer multiple of the fundamental frequency.

# %% [markdown] slideshow={"slide_type": "slide"}
# Here's the spectrum of a square wave with the same fundamental frequency.  

# %% slideshow={"slide_type": "slide"}
square_wave = make_wave(square_func, duration=0.5, freq=freq)
square_spectrum = make_spectrum(square_wave)
plot_spectrum(square_spectrum, 
              title='Spectrum of a square wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# The spectrum of the square wave has only odd harmonics.

# %% slideshow={"slide_type": "slide"}
triangle_wave = make_wave(triangle_func, duration=0.5, freq=freq)
triangle_spectrum = make_spectrum(triangle_wave)
plot_spectrum(triangle_spectrum, 
              title='Spectrum of a triangle wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# The spectrum of the triangle wave has odd harmonics only.
#
# But they drop off more quickly.
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Sound
#
#
# `make_audio` makes an IPython `Audio` object we can use to play a wave.

# %% slideshow={"slide_type": "slide"}
from IPython.display import Audio

def make_audio(wave):
    """Makes an IPython Audio object.
    """
    return Audio(data=wave, rate=framerate)


# %% [markdown] slideshow={"slide_type": "slide"}
# Dropping to 500 Hz to spare your ears.

# %% slideshow={"slide_type": "-"}
freq = 500

# %% slideshow={"slide_type": "slide"}
sinusoid_wave = make_wave(sinusoid_func, duration=0.5, freq=freq)
make_audio(sinusoid_wave)

# %% slideshow={"slide_type": "slide"}
triangle_wave = make_wave(triangle_func, duration=0.5, freq=freq)
make_audio(triangle_wave)

# %% slideshow={"slide_type": "slide"}
sawtooth_wave = make_wave(sawtooth_func, duration=0.5, freq=freq)
make_audio(sawtooth_wave)

# %% slideshow={"slide_type": "slide"}
square_wave = make_wave(square_func, duration=0.5, freq=freq)
make_audio(square_wave)


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Dropoff
#
# Let's see how the spectrums depend on $f$.

# %% slideshow={"slide_type": "slide"}
def plot_over_f(spectrum, freq, exponent):
    fs = spectrum.index
    hs = 1 / fs**exponent

    over_f = pd.Series(hs, fs)
    over_f[fs<freq] = np.nan
    over_f *= np.abs(spectrum[freq]) / over_f[freq]
    over_f.plot(color='gray')


# %% slideshow={"slide_type": "skip"}
freq = 500

# %% slideshow={"slide_type": "skip"}
sawtooth_wave = make_wave(func=sawtooth_func, 
                          duration=0.5, freq=freq)
sawtooth_spectrum = make_spectrum(sawtooth_wave)

# %% slideshow={"slide_type": "slide"}
plot_over_f(sawtooth_spectrum, freq, 1)
plot_spectrum(sawtooth_spectrum, 
              title='Spectrum of a sawtooth wave')

# %% slideshow={"slide_type": "skip"}
square_wave = make_wave(func=square_func, 
                        duration=0.5, freq=freq)
square_spectrum = make_spectrum(square_wave)

# %% slideshow={"slide_type": "slide"}
plot_over_f(square_spectrum, freq, 1)
plot_spectrum(square_spectrum, 
              title='Spectrum of a square wave')

# %% slideshow={"slide_type": "skip"}
triangle_wave = make_wave(func=triangle_func, 
                          duration=0.5, freq=freq)
triangle_spectrum = make_spectrum(triangle_wave)

# %% slideshow={"slide_type": "slide"}
plot_over_f(triangle_spectrum, freq, 2)
plot_spectrum(triangle_spectrum, 
              title='Spectrum of a triangle wave')


# %% slideshow={"slide_type": "skip"}
def text(x, y, text):
    transform = plt.gca().transAxes
    plt.text(x, y, text, transform=transform)


# %% slideshow={"slide_type": "skip"}
def plot_four_spectrums(fourth=None):
    plt.figure(figsize=(9, 6))

    plt.subplot(2, 2, 1)
    plot_over_f(square_spectrum, freq, 1)
    plot_spectrum(square_spectrum, 
              title='Spectrum of a square wave')
    text(0.3, 0.5, 'Odd harmonics, $1/f$ dropoff.')

    plt.subplot(2, 2, 2)
    plot_over_f(triangle_spectrum, freq, 2)
    plot_spectrum(triangle_spectrum, 
              title='Spectrum of a triangle wave')
    text(0.3, 0.5, 'Odd harmonics, $1/f^2$ dropoff.')

    plt.subplot(2, 2, 3)
    plot_over_f(sawtooth_spectrum, freq, 1)
    plot_spectrum(sawtooth_spectrum, 
              title='Spectrum of a sawtooth wave')
    text(0.3, 0.5, 'All harmonics, $1/f$ dropoff.')


    plt.subplot(2, 2, 4)
    text(0.3, 0.5, 'All harmonics, $1/f^2$ dropoff.')
    if fourth is not None:
        plot_over_f(fourth, freq, 2)
        plot_spectrum(fourth, 
              title='Spectrum of a parabola wave')
    else: 
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

# %% slideshow={"slide_type": "slide"}
plot_four_spectrums()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Method 1: Filtering
#
# One option is to start with a sawtooth wave, which has all of the harmonics we need.

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "skip"}
sawtooth_wave = make_wave(sawtooth_func, duration=0.5, freq=freq)
sawtooth_spectrum = make_spectrum(sawtooth_wave)

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plot_over_f(sawtooth_spectrum, freq, 1)
plot_spectrum(sawtooth_spectrum, 
              title='Spectrum of a sawtooth wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# And filter it by dividing through by $f$.

# %% jupyter={"outputs_hidden": false}
fs = sawtooth_spectrum.index
filtered_spectrum = sawtooth_spectrum / fs
filtered_spectrum[0:400] = 0

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plot_over_f(filtered_spectrum, freq, 2)
plot_spectrum(filtered_spectrum, 
              title='Spectrum of the filtered wave')


# %% [markdown] slideshow={"slide_type": "slide"}
# Now we can convert from spectrum to wave.

# %% jupyter={"outputs_hidden": false}
def make_wave_from_spectrum(spectrum):
    ys = np.fft.irfft(spectrum)
    n = len(ys)
    ts = np.arange(n) * dt

    series = pd.Series(ys, ts)
    return series


# %% jupyter={"outputs_hidden": false}
filtered_wave = make_wave_from_spectrum(filtered_spectrum)

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plot_wave(filtered_wave, title='Filtered wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# It's an interesting shape, but not easy to see what its functional form is.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Method 2: Additive synthesis
#
# Another approach is to add up a series of cosine signals with the right frequencies and amplitudes.

# %% slideshow={"slide_type": "slide"}
fundamental = 500
nyquist = framerate / 2
freqs = np.arange(fundamental, nyquist, fundamental)
amps = 1 / freqs**2

# %% slideshow={"slide_type": "slide"}
total = 0
for f, amp in zip(freqs, amps):
    component = amp * make_wave(sinusoid_func, 0.5, f)
    total += component

# %% slideshow={"slide_type": "slide"}
synth_wave = unbias(normalize(total))
synth_spectrum = make_spectrum(synth_wave)

# %% [markdown] slideshow={"slide_type": "slide"}
# Here's what the spectrum looks like:

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plot_over_f(synth_spectrum, freq, 2)
plot_spectrum(synth_spectrum)

# %% [markdown] slideshow={"slide_type": "slide"}
# And here's what the waveform looks like.

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plot_wave(synth_wave, title='Synthesized wave')

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plt.figure(figsize=(9,4))
plt.subplot(1, 2, 1)
plot_wave(filtered_wave, title='Filtered wave')
plt.subplot(1, 2, 2)
plot_wave(synth_wave, title='Synthesized wave')

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plt.figure(figsize=(9,4))
plt.subplot(1, 2, 1)
plot_spectrum(filtered_spectrum, title='Filtered spectrum')
plt.subplot(1, 2, 2)
plot_spectrum(synth_spectrum, title='Synthesized spectrum')

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
make_audio(synth_wave)

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
make_audio(filtered_wave)

# %% [markdown] slideshow={"slide_type": "slide"}
# What we hear depends on the magnitudes, not their phase.
#
# Mostly.
#
# [Autocorrelation and the case of the missing fundamental](https://www.dsprelated.com/showarticle/909.php)

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
plot_wave(synth_wave, title='Synthesized wave')


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Method 3: Parabolas

# %% slideshow={"slide_type": "slide"}
def parabola_func(cycles):
    ys = wrap(cycles) - 0.5
    return ys**2


# %% slideshow={"slide_type": "slide"}
parabola_wave = make_wave(parabola_func, 0.5, freq)
plot_wave(parabola_wave, title='Parabola wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# Does it have the right harmonics?

# %% jupyter={"outputs_hidden": false} slideshow={"slide_type": "slide"}
parabola_spectrum = make_spectrum(parabola_wave)
plot_over_f(parabola_spectrum, freq, 2)
plot_spectrum(parabola_spectrum, 
              title='Spectrum of a parabola wave')

# %% [markdown] slideshow={"slide_type": "slide"}
# We looked at the waveform and guessed it's a parabola.
#
# So we made a parabola and it seems to work.
#
# Satisfied?

# %% [markdown] slideshow={"slide_type": "slide"}
# There's another way to get there:

# %% [markdown] slideshow={"slide_type": "fragment"}
# The integral property of the Fourier transform.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Method 4: Integration

# %% [markdown] slideshow={"slide_type": "slide"}
# The basis functions of the FT are the complex exponentials:
#
# $e^{2 \pi i f t}$
#
# And we know how to integrate them.
#

# %% [markdown] slideshow={"slide_type": "slide"}
#
# $\int e^{2 \pi i f t}~dt = \frac{1}{2 \pi i f}~e^{2 \pi i f t}$
#
#
# Integration in the time domain is a $1/f$ filter in the frequency domain.

# %% [markdown] slideshow={"slide_type": "slide"}
# When we applied a $1/f$ filter to the spectrum of the sawtooth,
#
# We were integrating in the time domain.

# %% [markdown] slideshow={"slide_type": "slide"}
# Which we can approximate with `cumsum`.

# %% slideshow={"slide_type": "-"}
integrated_wave = unbias(normalize(sawtooth_wave.cumsum()))

# %% slideshow={"slide_type": "slide"}
plot_wave(integrated_wave)

# %% slideshow={"slide_type": "slide"}
integrated_spectrum = make_spectrum(integrated_wave)

plot_over_f(integrated_spectrum, freq, 2)
plot_spectrum(integrated_spectrum, 
              title='Spectrum of an integrated sawtooth wave')


# %% [markdown] slideshow={"slide_type": "slide"}
# Summary

# %% slideshow={"slide_type": "skip"}
def plot_four_waves(fourth=None):

    plt.figure(figsize=(9, 6))

    plt.subplot(2, 2, 1)
    plot_wave(square_wave, 
          title='Square wave')

    plt.subplot(2, 2, 2)
    plot_wave(triangle_wave, 
          title='Triangle wave')

    plt.subplot(2, 2, 3)
    plot_wave(sawtooth_wave, 
          title='Sawtooth wave')
    
    if fourth is not None:
        plt.subplot(2, 2, 4)
        plot_wave(parabola_wave, 
          title='Parabola wave')


    plt.tight_layout()

# %% slideshow={"slide_type": "slide"}
plot_four_waves()

# %% slideshow={"slide_type": "slide"}
plot_four_spectrums()

# %% [markdown] slideshow={"slide_type": "slide"}
# Four ways:
#
# * $1/f$ filter
#
# * Additive synthesis
#
# * Parabola waveform
#
# * Integrated sawtooth

# %% slideshow={"slide_type": "slide"}
plot_four_spectrums(parabola_spectrum)

# %% slideshow={"slide_type": "slide"}
plot_four_waves(parabola_wave)

# %% [markdown] slideshow={"slide_type": "slide"}
# What have we learned?

# %% [markdown] slideshow={"slide_type": "fragment"}
# Most of [*Think DSP*](https://greenteapress.com/wp/think-dsp/).
#
# At least as an introduction.

# %% [markdown] slideshow={"slide_type": "slide"}
# Discrete Fourier transform as correspondence between a wave and its spectrum
#
# [Chapter 1](http://greenteapress.com/thinkdsp/html/thinkdsp002.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# Harmonic structure of sound
#
# [Chapter 2](http://greenteapress.com/thinkdsp/html/thinkdsp003.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# Lead in to chirps (variable frequency waves)
#
# [Chapter 3](http://greenteapress.com/thinkdsp/html/thinkdsp004.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# Lead in to pink noise, which drops off like $1/f^{\beta}$
#
# [Chapter 4](http://greenteapress.com/thinkdsp/html/thinkdsp005.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# Additive synthesis
#
# [Chapter 7](http://greenteapress.com/thinkdsp/html/thinkdsp008.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# Integral property of the Fourier transform
#
# [Chapter 9](http://greenteapress.com/thinkdsp/html/thinkdsp010.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# Convolution in the time domain corresponds to multiplication in the frequency domain.
#
# [Chapter 10](http://greenteapress.com/thinkdsp/html/thinkdsp011.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# Thank you!
#
# Web page: allendowney.com
#
# Twitter: @allendowney
#
# This notebook: [tinyurl.com/mysterywave](https://tinyurl.com/mysterywave)
