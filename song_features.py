import numpy as np
from song_spectrogram import load_mono
import pandas as pd
import matplotlib.pyplot as plt


SONG = "song2.mp3"

data, rate = load_mono(SONG)
seconds = len(data) // rate               # how many whole seconds

loudness = []
pitch = []

for s in range(seconds):
    chunk = data[s * rate : (s + 1) * rate]

    rms = np.sqrt(np.mean(chunk ** 2))
    loudness.append(rms)

    spectrum = np.abs(np.fft.rfft(chunk))              # strength of every frequency
    freqs = np.fft.rfftfreq(len(chunk), 1 / rate)      # which frequency each one is
    pitch.append(freqs[np.argmax(spectrum)])           # the strongest one

df = pd.DataFrame({
    "second":   range(seconds),
    "loudness": loudness,
    "pitch_hz": pitch,
})

print(df.head(20))                          # first 20 rows

df.to_csv(SONG + "_features.csv", index=False)  # save the table as a file

df.plot(x="second", y="loudness", figsize=(12, 4), title="Loudness over time")
plt.show()

loudest = df.sort_values("loudness", ascending=False).head(10)
print(loudest)                              # the 10 loudest seconds

quietest = df.sort_values("loudness").head(10)
print(quietest)                              # the 10 QUIETEST seconds