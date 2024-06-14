# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: pyt181
#     language: python
#     name: pyt181
# ---

# %% [markdown] slideshow={"slide_type": "skip"}
# Notes:
# - eegml_signal esfilters
# - 

# %% slideshow={"slide_type": "skip"}
import pathlib

# %% slideshow={"slide_type": "skip"}
import numpy as np
import bokeh
from bokeh.io import output_notebook
import matplotlib.pyplot as plt
output_notebook()

# %% slideshow={"slide_type": "skip"}
import eegvis
import eegml_signal
import eeghdf

# %% slideshow={"slide_type": "skip"}
import thinkdsp

# %%
NEO_DIR  = '/home/clee/code/eegml/data/stevenson_neonatalsz_2019/hdf'
NEO_PATH = pathlib.Path(NEO_DIR)


# %% [markdown]
# - example of signal 
#
#
#

# %%
eeg1 = eeghdf.Eeghdf('/home/clee/code/eegml/data/stevenson_neonatalsz_2019/hdf/eeg11.eeg.h5')

# %%
eeg1.signal_labels

# %%
import eegvis.nb_eegview

# %%
brow1 = eegvis.nb_eegview.EeghdfBrowser(eeg1,montage='double banana', page_width_seconds=20)
brow1.show()

# %%
w0  = thinkdsp.Wave(eeg1.phys_signals[7,0:1000*256], framerate=256*70)
w0.make_audio()

# %%
eegneosz5 = eeghdf.Eeghdf(NEO_DIR + '/eeg5.eeg.h5')

# %%
neosz5_sf, L_sec = eegneosz5.sample_frequency, eegneosz5.phys_signals.shape[1]/eegneosz5.sample_frequency
fs = neosz5_sf # default sammple refrequency
speedup = 70
neosz5_sf, L_sec 

# %%
brow1 = eegvis.nb_eegview.EeghdfBrowser(eegneosz5, montage='neonatal', page_width_seconds=20)
brow1.show()

# %%
sz2interval_sec = (150, 840.0)
sz2interval_sample = int(sz2interval_sec[0]*neosz5_sf), int(sz2interval_sec[1]*neosz5_sf)
sz2interval_sample

# %%
[(ii, lab) for (ii, lab) in enumerate(eegneosz5.electrode_labels) 
if 'T3' in lab ]


# %%
eeg5t3 = eegneosz5.phys_signals[7,:]

# %%
w1 =thinkdsp.Wave(eeg5t3[:10000*3],framerate=fs * 70)
w1.make_audio()

# %%
w2  = thinkdsp.Wave(eeg5t3[sz2interval_sample[0]:sz2interval_sample[1]], framerate=fs*70)
w2.make_audio()

# %%
w3 = np.mean(eegneosz5.phys_signals[0:19, :], axis=0) # try averaging all the channels together

# %%
w3w = thinkdsp.Wave(w3, framerate=fs*70)
w3w.make_audio()

# %%
