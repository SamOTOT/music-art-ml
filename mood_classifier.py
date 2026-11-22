import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mood_dataset import clip_features

FEATURES = ["loudness", "loudness_change", "loudness_peak", "brightness"]

class MoodGuesser:
    def __init__(self, k=3):
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k)

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

df = pd.read_csv("mood_dataset.csv")
X = df[FEATURES]
y = df["label"]

# ---------- Exp 0: one test of 5 hidden songs ----------
X_train, X_test, y_train, y_test, songs_train, songs_test = train_test_split(
    X, y, df["song"], test_size=0.25, stratify=y, random_state=0)
guesser = MoodGuesser(k=3)
guesser.train(X_train, y_train)
guesses = guesser.predict(X_test)
print("Exp 0 single test accuracy:", accuracy_score(y_test, guesses))

# ---------- Exp 1: cross-validation ----------
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = []
wrong = []
for train_idx, test_idx in folds.split(X, y):
    g = MoodGuesser(k=3)
    g.train(X.iloc[train_idx], y.iloc[train_idx])
    guesses = g.predict(X.iloc[test_idx])
    real = y.iloc[test_idx].values
    scores.append(accuracy_score(real, guesses))
    for song, r, gs in zip(df["song"].iloc[test_idx], real, guesses):
        if r != gs:
            wrong.append(song)

print("Scores per round:", np.round(scores, 2))
print("Cross-validation accuracy:", round(np.mean(scores), 2))
print("Songs it got wrong:", wrong)

# ---------- Final model: guess a new song ----------
final = MoodGuesser(k=3)
final.train(X, y)
NEW_SONG = "new_song.mp3"
new = pd.DataFrame([clip_features(NEW_SONG)])[FEATURES]
print(NEW_SONG, "->", final.predict(new)[0])