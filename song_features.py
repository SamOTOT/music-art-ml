import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from song_spectrogram import load_mono

BANDS = {
    "bass":    (20, 250),
    "lowmid":  (250, 1000),
    "mid":     (1000, 4000),
    "highmid": (4000, 8000),
    "high":    (8000, 16000),
}

def measure_seconds(data, rate):
    seconds = len(data) // rate
    loudness, pitch, brightness = [], [], []
    bands = {name: [] for name in BANDS}          # one empty list per band

    for s in range(seconds):
        chunk = data[s * rate : (s + 1) * rate]
        loudness.append(np.sqrt(np.mean(chunk ** 2)))
        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / rate)
        pitch.append(freqs[np.argmax(spectrum)])
        total = spectrum.sum()
        brightness.append((freqs * spectrum).sum() / total if total > 0 else 0)

        power = spectrum ** 2                       # energy of each frequency
        total_power = power.sum()
        for name, (low, high) in BANDS.items():
            in_band = (freqs >= low) & (freqs < high)        # True for frequencies inside this band
            share = power[in_band].sum() / total_power if total_power > 0 else 0
            bands[name].append(share)

    df = pd.DataFrame({
        "second": range(seconds),
        "loudness": loudness,
        "pitch_hz": pitch,
        "brightness": brightness,
    })
    for name in BANDS:
        df[name] = bands[name]                     # add one column per band
    return df

if __name__ == "__main__":
    SONG = "song2.mp3"
    data, rate = load_mono(SONG)
    df = measure_seconds(data, rate)
    print(df.head(20))
    df.to_csv(SONG + "_features.csv", index=False)
    df.plot(x="second", y="loudness", figsize=(12, 4), title="Loudness over time")
    plt.show()