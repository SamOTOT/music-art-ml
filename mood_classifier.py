import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from song_features import BANDS
from mood_dataset import clip_features

FEATURES_4 = ["loudness", "loudness_change", "loudness_peak", "brightness"]
FEATURES_14 = list(FEATURES_4)
for band in BANDS:
    FEATURES_14 += [band, band + "_change"]

class MoodGuesser:
    def __init__(self, model):                  # now you choose which model goes inside
        self.scaler = StandardScaler()
        self.model = model

    def train(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

df = pd.read_csv("mood_dataset.csv")
y = df["label"]

def cv_score(make_model, features):             # the Exp 1 scoreboard, as a reusable function
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for train_idx, test_idx in folds.split(df, y):
        g = MoodGuesser(make_model())            # a fresh model every round
        g.train(df[features].iloc[train_idx], y.iloc[train_idx])
        guesses = g.predict(df[features].iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], guesses))
    return np.mean(scores)

# ---------- Exp 4: compare models ----------
models = {
    "KNN k=3":       lambda: KNeighborsClassifier(n_neighbors=3),
    "KNN k=5":       lambda: KNeighborsClassifier(n_neighbors=5),
    "KNN k=7":       lambda: KNeighborsClassifier(n_neighbors=7),
    "Decision tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
}

results = []
for name, make in models.items():
    results.append({
        "model": name,
        "4 features": cv_score(make, FEATURES_4),
        "14 features": cv_score(make, FEATURES_14),
    })
print(pd.DataFrame(results).round(2))

# ---------- Which features does the tree use? ----------
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(df[FEATURES_14], y)
importance = pd.Series(tree.feature_importances_, index=FEATURES_14)
print(importance.sort_values(ascending=False).round(2).head(6))

# ---------- Final model: guess the new song ----------
final = MoodGuesser(KNeighborsClassifier(n_neighbors=3))
final.train(df[FEATURES_4], y)
new = pd.DataFrame([clip_features("new_song.mp3")])[FEATURES_4]
print("new_song.mp3 ->", final.predict(new)[0])