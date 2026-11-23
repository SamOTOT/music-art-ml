import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile
from tone_generator import note_to_freq, make_tone, SAMPLE_RATE
from drum_track import make_bar
import soundfile as sf
from song_features import measure_seconds
from song_painting import make_painting
from photo_palette import get_palette
from photo_melody import palette_to_melody
import pandas as pd
from mood_classifier import train_final, FEATURES_4
from mood_dataset import clip_features
import librosa
import librosa.display

def to_wav(wave, rate=SAMPLE_RATE):
    """Turn a wave (numbers) into .wav bytes the app can play."""
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    buffer = io.BytesIO()
    wavfile.write(buffer, rate, (wave * 32767 * 0.8).astype(np.int16))
    return buffer.getvalue()

def plot_wave(wave, rate, title):
    """Draw a waveform and return the figure."""
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(len(wave)) / rate
    ax.plot(t, wave, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

@st.cache_data
def load_song(file_bytes):
    """Open an uploaded song once, and remember it between reruns."""
    data, rate = sf.read(io.BytesIO(file_bytes))
    channels = 1 if data.ndim == 1 else data.shape[1]
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, rate, channels

@st.cache_data
def mel_db_of(file_bytes):
    """Mel spectrogram of the uploaded song in decibels, computed once."""
    data, rate, _ = load_song(file_bytes)
    mel = librosa.feature.melspectrogram(y=data.astype(np.float32), sr=rate, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max), rate

def plot_mel(mel_db, rate, title):
    """Draw a mel spectrogram and return the figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    image = librosa.display.specshow(mel_db, sr=rate, x_axis="time", y_axis="mel",
                                     fmax=8000, cmap="magma", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz, mel scale)")
    fig.colorbar(image, ax=ax, format="%+2.0f dB", label="Loudness (dB below the loudest moment)")
    return fig

def plot_spec(data, rate, title):
    """Draw a spectrogram and return the figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _, _, _, image = ax.specgram(data, Fs=rate, NFFT=2048, noverlap=1024, cmap="magma", vmin=-130)
    ax.set_ylim(0, 8000)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, label="Loudness (dB)")
    return fig

@st.cache_data
def song_table(file_bytes):
    """Measure the uploaded song second by second (the #7 table), once."""
    data, rate, _ = load_song(file_bytes)
    return measure_seconds(data, rate)

@st.cache_data
def photo_palette(file_bytes):
    """Run k-means on the uploaded photo once."""
    return get_palette(io.BytesIO(file_bytes))

@st.cache_resource
def mood_model():
    """Train the final mood model once, when the app starts."""
    return train_final()

@st.cache_data
def song_mood_features(file_bytes):
    """Measure the uploaded song the same way as the 20 training songs."""
    return clip_features(io.BytesIO(file_bytes))

st.set_page_config(page_title="Music × Art × ML", layout="wide")
st.title("Music × Art × ML")
st.caption("Sound, pictures and machine learning, all made from numbers.")

st.sidebar.header("Your files")
song_file = st.sidebar.file_uploader("Song (MP3 or WAV)", type=["mp3", "wav"])
photo_file = st.sidebar.file_uploader("Photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

tabs = st.tabs(["Tone generator", "Drum machine", "Song visualizer",
                "Painting", "Photo melody", "Mood guesser"])

# ---------- Tab 1: tone generator ----------
with tabs[0]:
    st.header("Tone generator")
    text = st.text_input("Notes, separated by spaces", "C4 E4 G4 E4 C4")
    length = st.slider("Note length (seconds)", 0.1, 1.0, 0.4, 0.05)

    if st.button("Make melody"):
        try:
            waves = []
            for note in text.split():
                waves.append(make_tone(note_to_freq(note), length))
            melody = np.concatenate(waves)
        except (ValueError, IndexError):
            st.error("Couldn't read those notes. Use names like C4, F#3 or A5, separated by spaces.")
        else:
            st.audio(to_wav(melody), format="audio/wav")
            st.pyplot(plot_wave(melody, SAMPLE_RATE, "Your melody"))


# ---------- Tab 2: drum machine ----------
with tabs[1]:
    st.header("Drum machine")
    bpm = st.slider("Tempo (BPM)", 60, 180, 120)
    repeats = st.slider("How many bars", 1, 8, 2)
    st.write("Tick the steps where each drum plays. 16 steps = 1 bar.")

    defaults = {
        "Kick":   [1,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,0],
        "Snare":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "Hi-hat": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
    }
    patterns = {}
    for drum, default in defaults.items():
        cols = st.columns(17)                       # 1 column for the name + 16 for the steps
        cols[0].write(f"**{drum}**")
        steps = []
        for i in range(16):
            on = cols[i + 1].checkbox(f"{drum} step {i + 1}", value=bool(default[i]),
                                      key=f"{drum}-{i}", label_visibility="collapsed")
            steps.append(1 if on else 0)
        patterns[drum] = steps

    if st.button("Make beat"):
        bar = make_bar(patterns["Kick"], patterns["Snare"], patterns["Hi-hat"], bpm)
        beat = np.concatenate([bar] * repeats)
        st.audio(to_wav(beat), format="audio/wav")
        st.pyplot(plot_wave(beat, SAMPLE_RATE, f"Your beat: {repeats} bar(s) at {bpm} BPM"))


# ---------- Tab 3: song visualizer ----------
with tabs[2]:
    st.header("Song visualizer")
    if song_file is None:
        st.info("Upload a song in the sidebar to see it here.")
    else:
        data, rate, channels = load_song(song_file.getvalue())
        st.audio(song_file)

        seconds = len(data) / rate
        c1, c2, c3 = st.columns(3)
        c1.metric("Length", f"{int(seconds // 60)} min {int(seconds % 60)} s")
        c2.metric("Sample rate", f"{rate:,} Hz")
        c3.metric("Channels", "stereo" if channels == 2 else "mono")

        step = max(1, len(data) // 200_000)          # draw at most ~200,000 points
        st.pyplot(plot_wave(data[::step], rate / step, "Waveform"))
        
        mel_db, mel_rate = mel_db_of(song_file.getvalue())
        st.pyplot(plot_mel(mel_db, mel_rate, "Mel spectrogram"))
        st.caption("Same song, with the pitch axis redrawn the way ears hear it: low notes get more room, "
                   "high notes are squeezed together. This is the kind of picture audio AI models learn from.")

# ---------- Tab 4: painting ----------
with tabs[3]:
    st.header("Painting")
    if song_file is None:
        st.info("Upload a song in the sidebar to paint it.")
    else:
        table = song_table(song_file.getvalue())
        cols = st.slider("Circles per row", 8, 32, 16)
        fig = make_painting(table, cols)
        st.pyplot(fig)
        st.caption("One circle per second. Size = loudness, colour = pitch (dark purple = low, yellow = high).")

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, facecolor="black", bbox_inches="tight")
        st.download_button("Download painting (PNG)", buffer.getvalue(),
                           file_name="painting.png", mime="image/png")

# ---------- Tab 5: photo melody ----------
with tabs[4]:
    st.header("Photo melody")
    if photo_file is None:
        st.info("Upload a photo in the sidebar to hear it.")
    else:
        palette, share = photo_palette(photo_file.getvalue())
        notes, tune = palette_to_melody(palette, share)

        left, right = st.columns([1, 2])
        left.image(photo_file, caption="Your photo")
        with right:
            st.write("Its 5 main colours (k-means) and the note each one becomes:")
            for colour, part, note in zip(palette, share, notes):
                hex_code = "#{:02x}{:02x}{:02x}".format(*colour)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px;margin:6px 0'>"
                    f"<div style='width:56px;height:32px;border-radius:6px;background:{hex_code}'></div>"
                    f"<b>{note}</b> &nbsp;·&nbsp; {part * 100:.0f}% of the photo</div>",
                    unsafe_allow_html=True)

        repeats = st.slider("Play it how many times", 1, 5, 3)
        st.audio(to_wav(np.concatenate([tune] * repeats)), format="audio/wav")
        st.caption("Darker colour = lower note. Bigger share of the photo = longer note.")

# ---------- Tab 6: mood guesser ----------
with tabs[5]:
    st.header("Mood guesser")
    if song_file is None:
        st.info("Upload a song in the sidebar and the model will guess its mood.")
    else:
        model, data = mood_model()
        features = song_mood_features(song_file.getvalue())
        row = pd.DataFrame([features])[FEATURES_4]
        guess = model.predict(row)[0]

        st.metric("The model thinks this song is", guess.upper())

        st.subheader("How it decided")
        st.write("Its 3 most similar songs from the training set. They voted:")
        _, idx = model.model.kneighbors(model.scaler.transform(row))
        st.table(data.iloc[idx[0]][["song", "label"]])

        st.write("This song's numbers next to the calm and energetic averages:")
        compare = data.groupby("label")[FEATURES_4].mean().T
        compare["this song"] = row.iloc[0]
        st.dataframe(compare.round(3))

        st.caption("Model: KNN (k = 3) on 4 whole-song features. 85% accuracy in cross-validation "
                   "on 20 labelled songs. It can't hear rhythm, so some songs fool it.")