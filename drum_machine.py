
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

SAMPLE_RATE = 44100

def show(wave, title):
    t = np.arange(len(wave)) / SAMPLE_RATE

    plt.plot(t, wave)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def make_kick(duration=0.3):
    t = np.linspace(
        0, duration,
        int(SAMPLE_RATE * duration),
        endpoint=False
    )

    tone = np.sin(2 * np.pi * 100 * t)
    envelope = np.exp(-40 * t)

    return tone * envelope

def make_snare(duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    noise = np.random.uniform(-1, 1, len(t))
    envelope = np.exp(-20 * t)

    return noise * envelope

def make_hihat(duration=0.1):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    hihat = np.random.uniform(-1, 1, len(t))
    envelope = np.exp(-90 * t)

    return hihat * envelope

BPM = 120
STEPS = 16
step_samples = int(SAMPLE_RATE * 60 / BPM / 4)    # samples in one step
bar = np.zeros(STEPS * step_samples)              # a silent bar

kick_pattern  = [1,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,0]
snare_pattern = [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0]
hihat_pattern = [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0]

def add_track(bar, sound, pattern, step_samples):
    for i, hit in enumerate(pattern):
        if hit == 1:
            start = i * step_samples                 # where step i begins, in samples
            piece = sound[:len(bar) - start]        # trim if it runs past the end
            bar[start:start + len(piece)] += piece

add_track(bar, make_kick(), kick_pattern, step_samples)
add_track(bar, make_snare(), snare_pattern, step_samples)
add_track(bar, make_hihat() * 0.8 , hihat_pattern, step_samples)

bar = bar / np.max(np.abs(bar))                   # squash into -1..1 so it can't clip
show(bar, "One bar")
wavfile.write("beat.wav", SAMPLE_RATE, (bar * 32767 * 0.8).astype(np.int16))