import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ---------- the tools ----------

def note_to_freq(note):
    name = note[:-1]
    octave = int(note[-1])
    position = NOTES.index(name)
    n = (octave - 4) * 12 + (position - 9)
    return 440 * (2 ** (n / 12))

def make_tone(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    fade_len = int(SAMPLE_RATE * 0.01)
    ramp = np.linspace(0, 1, fade_len)
    wave[:fade_len] *= ramp
    wave[-fade_len:] *= ramp[::-1]
    return wave

# ---------- the doing part (only when run directly) ----------

if __name__ == "__main__":
    text = input("Enter notes separated by spaces (e.g. C4 E4 G4): ")
    melody = text.split()

    waves = []
    for note in melody:
        waves.append(make_tone(note_to_freq(note), 0.4))

    song = np.concatenate(waves)
    song = (song * 32767 * 0.5).astype(np.int16)
    wavfile.write("melody.wav", SAMPLE_RATE, song)
    print("saved melody.wav")