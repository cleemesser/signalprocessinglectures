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
# This notebook contains an example from Chapter 10: Signals and Systems
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
# ## LTI System Theory
#
# This notebook contains one of the coolest examples in Think DSP.  It uses LTI system theory to characterize the acoustics of a recording space and simulate the effect this space would have on the sound of a violin performance.
#
# I'll start with a recording of a gunshot:

# %%
if not os.path.exists('180960__kleeb__gunshot.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/180960__kleeb__gunshot.wav

# %%
from thinkdsp import read_wave

response = read_wave('180960__kleeb__gunshot.wav')

start = 0.12
response = response.segment(start=start)
response.shift(-start)

response.plot()
decorate(xlabel='Time (s)', ylabel='Amplitude')

# %% [markdown]
# If you play this recording, you can hear the initial shot and several seconds of echos.

# %%
response.make_audio()

# %% [markdown]
# This wave records the "impulse response" of the room where the gun was fired.

# %% [markdown]
# Now let's load a recording of a violin performance:

# %%
if not os.path.exists('92002__jcveliz__violin-origional.wav'):
    # !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/92002__jcveliz__violin-origional.wav

# %%
wave = read_wave('92002__jcveliz__violin-origional.wav')
wave.truncate(len(response))
wave.normalize()
wave.plot()
decorate(xlabel='Time (s)', ylabel='Amplitude')

# %% [markdown]
# And listen to it:

# %%
wave.make_audio()

# %% [markdown]
# Now we can figure out what the violin would sound like if it was played in the room where the gun was fired.  All we have to do is convolve the two waves:

# %%
output = wave.convolve(response)
output.normalize()

# %% [markdown]
# Here's what it looks like:

# %%
wave.plot(label='original')
output.plot(label='convolved')
decorate(xlabel='Time (s)', ylabel='Amplitude')

# %% [markdown]
# And here's what it sounds like:

# %%
output.make_audio()

# %% [markdown]
# If you think this example is black magic, you are not alone.   But there is a good reason why this works, and I do my best to explain it in Chapter 10.  So stay tuned.
#
# I'd like to thank jcveliz and kleeb for making these recordings available from freesound.org.

# %% jupyter={"outputs_hidden": true}
