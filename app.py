import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile
from tone_generator import note_to_freq, make_tone, SAMPLE_RATE
from drum_track import make_bar

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

# ---------- Tabs still to build ----------
for i, ticket in [(2, "18.3"), (3, "18.4"), (4, "18.5"), (5, "18.6")]:
    with tabs[i]:
        st.info(f"Coming in ticket {ticket}.")