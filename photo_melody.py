import numpy as np
from scipy.io import wavfile
from photo_palette import get_palette                               # your k-means tool
from tone_generator import note_to_freq, make_tone, SAMPLE_RATE     # your tone tools from #2

PHOTO = "photo.jpg"
SCALE = ["C3","D3","E3","F3","G3","A3","B3",
         "C4","D4","E4","F4","G4","A4","B4","C5"]   # white keys, low → high

palette, share = get_palette(PHOTO)

notes = []
waves = []
for colour, part in zip(palette, share):
    brightness = colour.mean()                               # 0 = black, 255 = white
    note = SCALE[int(brightness / 256 * len(SCALE))]         # dark → low note, bright → high note
    duration = 0.2 + part * 3                                # bigger share → longer note
    notes.append(note)
    waves.append(make_tone(note_to_freq(note), duration))

print(notes)
tune = np.concatenate(waves * 3)                             # play the 5 notes 3 times
wavfile.write(PHOTO + "_melody.wav", SAMPLE_RATE, (tune * 32767 * 0.5).astype(np.int16))