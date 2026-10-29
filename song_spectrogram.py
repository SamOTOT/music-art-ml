import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def load_mono(path):                      # open any file, always get one list back
    data, rate = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, rate

def plot_spectrogram(path, out_png):
    data, rate = load_mono(path)
    plt.figure(figsize=(12, 5))
    plt.specgram(data, Fs=rate, NFFT=2048, noverlap=1024, cmap="magma", vmin=-130)
    plt.ylim(0, 8000)                     # show 0–8,000 Hz, where most of the music is
    plt.title("Spectrogram: " + path)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Loudness (dB)")
    plt.savefig(out_png, dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_spectrogram("song.mp3", "spectrogram_song.png")
    plot_spectrogram("track.wav", "spectrogram_track.png")