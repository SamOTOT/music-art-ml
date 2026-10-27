import numpy as np
from scipy.io import wavfile

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_to_freq(note):
    name = note[:-1]          # "C#4" -> "C#"
    octave = int(note[-1])    # "C#4" -> 4
    position = NOTES.index(name)
    n = (octave - 4) * 12 + (position - 9)  # semitones away from A4
    return 440 * (2 ** (n / 12))  # the frequency formula

#frequency to sound

SAMPLE_RATE = 44100

def make_tone(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2 * np.pi * freq * t)
    fade_len = int(SAMPLE_RATE * 0.01)       # 10 ms = 441 samples
    ramp = np.linspace(0, 1, fade_len)       # 0 → 1, smoothly
    wave[:fade_len] *= ramp                  # fade in the first 441 samples
    wave[-fade_len:] *= ramp[::-1]          # fade out the last 441 samples
    return wave

text = input("Enter notes separated by spaces (e.g. C4 E4 G4): ")
melody = text.split()

waves = []
for note in melody:
    freq = note_to_freq(note)
    waves.append(make_tone(freq, 0.4))

song = np.concatenate(waves)
song = (song * 32767 * 0.5).astype(np.int16)
wavfile.write("melody.wav", SAMPLE_RATE, song)
print("saved melody.wav")