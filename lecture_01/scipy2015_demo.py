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
# ThinkDSP
# ========
# by Allen Downey (think-dsp.com)
#
# This notebook contains examples and demos for a SciPy 2015 talk.

# %% jupyter={"outputs_hidden": false}
import thinkdsp
from thinkdsp import decorate

import numpy as np

# %% [markdown]
# A Signal represents a function that can be evaluated at an point in time.

# %% jupyter={"outputs_hidden": false}
cos_sig = thinkdsp.CosSignal(freq=440)

# %% [markdown]
# A cosine signal at 440 Hz has a period of 2.3 ms.

# %% jupyter={"outputs_hidden": false}
cos_sig.plot()
decorate(xlabel='time (s)')

# %% [markdown]
# `make_wave` samples the signal at equally-space time steps.

# %% jupyter={"outputs_hidden": false}
wave = cos_sig.make_wave(duration=0.5, framerate=11025)

# %% [markdown]
# `make_audio` creates a widget that plays the Wave.

# %% jupyter={"outputs_hidden": false}
wave.apodize()
wave.make_audio()

# %% [markdown]
# `make_spectrum` returns a Spectrum object.

# %% jupyter={"outputs_hidden": false}
spectrum = wave.make_spectrum()

# %% [markdown]
# A cosine wave contains only one frequency component (no harmonics).

# %% jupyter={"outputs_hidden": false}
spectrum.plot()
decorate(xlabel='frequency (Hz)')

# %% [markdown]
# A SawTooth signal has a more complex harmonic structure.

# %% jupyter={"outputs_hidden": false}
saw_sig = thinkdsp.SawtoothSignal(freq=440)
saw_sig.plot()

# %% [markdown]
# Here's what it sounds like:

# %% jupyter={"outputs_hidden": false}
saw_wave = saw_sig.make_wave(duration=0.5)
saw_wave.make_audio()

# %% [markdown]
# And here's what the spectrum looks like:

# %% jupyter={"outputs_hidden": false}
saw_wave.make_spectrum().plot()

# %% [markdown]
# Here's a short violin performance from jcveliz on freesound.org:

# %% jupyter={"outputs_hidden": false}
violin = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')
violin.make_audio()

# %% [markdown]
# The spectrogram shows the spectrum over time:

# %% jupyter={"outputs_hidden": false}
spectrogram = violin.make_spectrogram(seg_length=1024)
spectrogram.plot(high=5000)

# %% [markdown]
# We can select a segment where the pitch is constant:

# %% jupyter={"outputs_hidden": false}
start = 1.2
duration = 0.6
segment = violin.segment(start, duration)

# %% [markdown]
# And compute the spectrum of the segment:

# %% jupyter={"outputs_hidden": false}
spectrum = segment.make_spectrum()
spectrum.plot()

# %% [markdown]
# The dominant and fundamental peak is at 438.3 Hz, which is a slightly flat A4 (about 7 cents). 

# %% jupyter={"outputs_hidden": false}
spectrum.peaks()[:5]

# %% [markdown]
# As an aside, you can use the spectrogram to help extract the Parson's code and then identify the song.

# %% [markdown]
# Parson's code: DUUDDUURDR
#
# Send it off to http://www.musipedia.org

# %% [markdown]
# A chirp is a signal whose frequency varies continuously over time (like a trombone).

# %% jupyter={"outputs_hidden": false}
import math
PI2 = 2 * math.pi

class SawtoothChirp(thinkdsp.Chirp):
    """Represents a sawtooth signal with varying frequency."""

    def _evaluate(self, ts, freqs):
        """Helper function that evaluates the signal.

        ts: float array of times
        freqs: float array of frequencies during each interval
        """
        dts = np.diff(ts)
        dps = PI2 * freqs * dts
        phases = np.cumsum(dps)
        phases = np.insert(phases, 0, 0)
        cycles = phases / PI2
        frac, _ = np.modf(cycles)
        ys = thinkdsp.normalize(thinkdsp.unbias(frac), self.amp)
        return ys


# %% [markdown]
# Here's what it looks like:

# %% jupyter={"outputs_hidden": false}
signal = SawtoothChirp(start=220, end=880)
wave = signal.make_wave(duration=2, framerate=10000)
segment = wave.segment(duration=0.06)
segment.plot()

# %% [markdown]
# Here's the spectrogram.

# %% jupyter={"outputs_hidden": false}
spectrogram = wave.make_spectrogram(1024)
spectrogram.plot()
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

# %% [markdown]
# What do you think it sounds like?

# %% jupyter={"outputs_hidden": false}
wave.apodize()
wave.make_audio()

# %% [markdown]
# Up next is one of the coolest examples in Think DSP.  It uses LTI system theory to characterize the acoustics of a recording space and simulate the effect this space would have on the sound of a violin performance.
#
# I'll start with a recording of a gunshot:

# %% jupyter={"outputs_hidden": false}
response = thinkdsp.read_wave('180960__kleeb__gunshot.wav')

start = 0.12
response = response.segment(start=start)
response.shift(-start)

response.normalize()
response.plot()
decorate(xlabel='Time (s)', ylabel='amplitude')

# %% [markdown]
# If you play this recording, you can hear the initial shot and several seconds of echos.

# %% jupyter={"outputs_hidden": false}
response.make_audio()

# %% [markdown]
# This wave records the "impulse response" of the room where the gun was fired.

# %% [markdown]
# Now let's load a recording of a violin performance:

# %% jupyter={"outputs_hidden": false}
wave = thinkdsp.read_wave('92002__jcveliz__violin-origional.wav')
start = 0.11
wave = wave.segment(start=start)
wave.shift(-start)

wave.truncate(len(response))
wave.normalize()
wave.plot()
decorate(xlabel='Time (s)', ylabel='Amplitude')

# %% [markdown]
# And listen to it:

# %% jupyter={"outputs_hidden": false}
wave.make_audio()

# %% [markdown]
# Now we can figure out what the violin would sound like if it was played in the room where the gun was fired.  All we have to do is convolve the two waves:

# %% jupyter={"outputs_hidden": false}
output = wave.convolve(response)
output.normalize()

# %% [markdown]
# Here's what it looks like:

# %% jupyter={"outputs_hidden": false}
wave.plot(label='original')
output.plot(label='convolved')
decorate(xlabel='Time (s)', ylabel='Amplitude')

# %% [markdown]
# And here's what it sounds like:

# %% jupyter={"outputs_hidden": false}
output.make_audio()

# %% [markdown]
# If you think this example is black magic, you are not alone.   But there is a good reason why this works, and I do my best to explain it in Chapter 9.  So stay tuned.
#
# I'd like to thanks jcveliz and kleeb for making these recordings available from freesound.org.

# %% jupyter={"outputs_hidden": false}
