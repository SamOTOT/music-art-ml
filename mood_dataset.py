import os
import pandas as pd
from song_spectrogram import load_mono
from song_features import measure_seconds

CLIP_SECONDS = 15

def clip_features(path):                     # measure ANY mp3, the same way every time
    data, rate = load_mono(path)
    middle = len(data) // 2
    half = CLIP_SECONDS * rate // 2
    clip = data[middle - half : middle + half]
    per_second = measure_seconds(clip, rate)
    return {
        "loudness": per_second["loudness"].mean(),
        "loudness_change": per_second["loudness"].std(),
        "pitch": per_second["pitch_hz"].mean(),
        "brightness": per_second["brightness"].mean(),
    }

if __name__ == "__main__":
    rows = []
    for label in ["calm", "energetic"]:
        folder = os.path.join("mood_songs", label)
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".mp3"):
                continue
            row = clip_features(os.path.join(folder, name))
            row["song"] = name
            row["label"] = label
            rows.append(row)
            print("measured:", label, name)

    df = pd.DataFrame(rows)
    print(df)
    print(df["label"].value_counts())
    df.to_csv("mood_dataset.csv", index=False)