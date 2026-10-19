import numpy as np
from scipy.io import wavfile

sample_rate = 44100   # samples per second
duration = 1          # seconds
frequency = 440       # Hz (the note A)

# 44,100 evenly spaced time points between 0 and 1 second
t = np.linspace(0, duration, sample_rate * duration, endpoint=False)

# the sine wave: one value between -1 and 1 at each time point
wave = np.sin(2 * np.pi * frequency * t)

# .wav files want whole numbers between -32767 and 32767
# (0.5 keeps the volume at half so it's not harsh)
wave = (wave * 32767 * 0.5).astype(np.int16)

wavfile.write("beep.wav", sample_rate, wave)
print("saved beep.wav")