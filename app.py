import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile
from tone_generator import note_to_freq, make_tone, SAMPLE_RATE

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

st.set_page_config(page_title="Music × Art × ML", layout="wide")
st.title("Music × Art × ML")
st.caption("Sound, pictures and machine learning, all made from numbers.")

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

# ---------- Tabs still to build ----------
for i, ticket in [(1, "18.2"), (2, "18.3"), (3, "18.4"), (4, "18.5"), (5, "18.6")]:
    with tabs[i]:
        st.info(f"Coming in ticket {ticket}.")