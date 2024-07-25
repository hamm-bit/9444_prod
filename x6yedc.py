import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal
from gammatone.filters import make_erb_filters, erb_filterbank

# Load the audio signal
audio_file = 'path_to_audio_file.wav'
y, sr = librosa.load(audio_file, sr=None)

# Gammatone filterbank
def apply_gammatone_filterbank(y, sr, num_bands=32, low_freq=100, high_freq=None):
    if high_freq is None:
        high_freq = sr / 2
    centre_freqs = np.linspace(low_freq, high_freq, num_bands)
    erb_filters = make_erb_filters(sr, centre_freqs)
    y_filtered, _ = erb_filterbank(y, erb_filters)
    return y_filtered

# Half Wave Rectifier (HWR)
def half_wave_rectifier(y):
    return np.maximum(y, 0)

# Teager Energy Operator (TEO)
def teager_energy_operator(y):
    return y[:-2]**2 - y[1:-1] * y[2:]

# Short-term Averaging
def short_term_averaging(y, frame_size=1024, hop_size=512):
    return librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_size)

# Logarithmic Compression
def logarithmic_compression(y):
    return np.log1p(y)

# Apply Gammatone filterbank
y_filtered = apply_gammatone_filterbank(y, sr)

# Apply Half Wave Rectifier
y_hwr = half_wave_rectifier(y_filtered)

# Apply Teager Energy Operator
y_teo = np.apply_along_axis(teager_energy_operator, 0, y_hwr)

# Apply Short-term Averaging
y_sta = np.apply_along_axis(short_term_averaging, 0, y_teo)

# Apply Logarithmic Compression
y_log = np.apply_along_axis(logarithmic_compression, 0, y_sta)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.plot(y)
plt.title('Audio Signal')

plt.subplot(6, 1, 2)
plt.imshow(y_filtered, aspect='auto', origin='lower')
plt.title('Gammatone Filterbank')

plt.subplot(6, 1, 3)
plt.imshow(y_hwr, aspect='auto', origin='lower')
plt.title('Half Wave Rectifier (HWR)')

plt.subplot(6, 1, 4)
plt.imshow(y_teo, aspect='auto', origin='lower')
plt.title('TEO (Teager Energy Operator)')

plt.subplot(6, 1, 5)
plt.imshow(y_sta, aspect='auto', origin='lower')
plt.title('Short-term Averaging')

plt.subplot(6, 1, 6)
plt.imshow(y_log, aspect='auto', origin='lower')
plt.title('Logarithmic Compression')

plt.tight_layout()
plt.show()
