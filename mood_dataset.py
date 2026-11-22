import os
import pandas as pd
from song_spectrogram import load_mono
from song_features import measure_seconds



def clip_features(path):                       # now measures the WHOLE song
    data, rate = load_mono(path)
    per_second = measure_seconds(data, rate)
    per_second = per_second.iloc[5:-5]         # skip the first and last 5 seconds
    return {
        "loudness": per_second["loudness"].mean(),
        "loudness_change": per_second["loudness"].std(),
        "loudness_peak": per_second["loudness"].quantile(0.9),   # NEW: how loud the loudest moments get
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