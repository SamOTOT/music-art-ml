import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tone_generator import note_to_freq, make_tone   # borrow the tools

SAMPLE_RATE = 44100

def show(wave, title):
    t = np.arange(len(wave)) / SAMPLE_RATE
    plt.plot(t, wave)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def make_kick(duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
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
    envelope = np.exp(-50 * t)
    return hihat * envelope

def add_track(bar, sound, pattern, step_samples):
    for i, hit in enumerate(pattern):
        if hit == 1:
            start = i * step_samples                 # where step i begins, in samples
            piece = sound[:len(bar) - start]         # trim if it runs past the end
            bar[start:start + len(piece)] += piece

def make_bar(kick_p, snare_p, hihat_p, bpm):         # NEW: the bar machine
    step_samples = int(SAMPLE_RATE * 60 / bpm / 4)
    bar = np.zeros(16 * step_samples)
    add_track(bar, make_kick(), kick_p, step_samples)
    add_track(bar, make_snare(), snare_p, step_samples)
    add_track(bar, make_hihat(), hihat_p, step_samples)
    return bar

if __name__ == "__main__":
    show(make_kick(), "Kick")
    show(make_snare(), "Snare")
    show(make_hihat(), "Hi-hat")

kick_pattern  = [1,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,0]
snare_pattern = [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0]
hihat_pattern = [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0]

fill_kick  = [1,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,0]
fill_snare = [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,1,1,1]
fill_hihat = [1,1,1,1, 1,1,1,1, 1,1,1,1, 0,0,0,0]

BPM = 90
song = ["A", "A", "A", "B"] * 4      # the plan above, 16 bars

bars = []
for name in song:
    if name == "A":
        bars.append(make_bar(kick_pattern, snare_pattern, hihat_pattern, BPM))
    else:
        bars.append(make_bar(fill_kick, fill_snare, fill_hihat, BPM))  # build a B bar

track = np.concatenate(bars)         # join all bars into one long track



beat_len = 60 / BPM          # one note per beat 

riff = ["C4","E4","G4","E4",  "A3","C4","E4","C4",
        "F3","A3","C4","A3",  "G3","B3","D4","B3"]   # 4 bars of tune

tune_notes = []
for note in riff * 4:        # repeat it 4 times → 16 bars, same as the drums
    tune_notes.append(make_tone(note_to_freq(note), beat_len))
tune = np.concatenate(tune_notes)

tune = tune[:len(track)]     # cut to the exact length of the drums
track = track + tune * 0.6   # add the tune on top, a bit quieter

peak = np.max(np.abs(track))
if peak > 0:                         # the safety check from #3
    track = track / peak

show(track, "Full track")
wavfile.write("track.wav", SAMPLE_RATE, (track * 32767 * 0.8).astype(np.int16))

