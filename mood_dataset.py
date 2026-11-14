import os
import pandas as pd
from song_spectrogram import load_mono
from song_features import measure_seconds

CLIP_SECONDS = 15
rows = []

for label in ["calm", "energetic"]:
    folder = os.path.join("mood_songs", label)
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".mp3"):
            continue                                        # skip anything that isn't an mp3

        data, rate = load_mono(os.path.join(folder, name))
        middle = len(data) // 2
        half = CLIP_SECONDS * rate // 2
        clip = data[middle - half : middle + half]          # 15 seconds from the middle

        per_second = measure_seconds(clip, rate)            # your #7 code
        rows.append({
            "song": name,
            "loudness": per_second["loudness"].mean(),
            "loudness_change": per_second["loudness"].std(),
            "pitch": per_second["pitch_hz"].mean(),
            "brightness": per_second["brightness"].mean(),
            "label": label,                                 # the folder name is the label
        })
        print("measured:", label, name)

df = pd.DataFrame(rows)
print(df)
print(df["label"].value_counts())                           # how balanced is it?
df.to_csv("mood_dataset.csv", index=False)