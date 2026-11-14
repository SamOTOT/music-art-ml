import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from song_spectrogram import load_mono

def measure_seconds(data, rate):
    seconds = len(data) // rate
    loudness, pitch, brightness = [], [], []
    for s in range(seconds):
        chunk = data[s * rate : (s + 1) * rate]
        loudness.append(np.sqrt(np.mean(chunk ** 2)))           # RMS, from #7
        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / rate)
        pitch.append(freqs[np.argmax(spectrum)])                # strongest frequency, from #7
        total = spectrum.sum()
        brightness.append((freqs * spectrum).sum() / total if total > 0 else 0)   # NEW
    return pd.DataFrame({
        "second": range(seconds),
        "loudness": loudness,
        "pitch_hz": pitch,
        "brightness": brightness,
    })

if __name__ == "__main__":
    SONG = "song.mp3"
    data, rate = load_mono(SONG)
    df = measure_seconds(data, rate)
    print(df.head(20))
    df.to_csv(SONG + "_features.csv", index=False)
    df.plot(x="second", y="loudness", figsize=(12, 4), title="Loudness over time")
    plt.show()
    print(df.sort_values("loudness", ascending=False).head(10))
    print(df.sort_values("loudness").head(10))