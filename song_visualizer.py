import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

#data, rate = sf.read("track.wav")      # your own track first
#print(rate, data.shape, data.dtype)

#data, rate = sf.read("song.mp3")       # now the real song
#print(rate, data.shape, data.dtype)

#------------------------------------------------------------------------------------------------

data, rate = sf.read("song.mp3")

if data.ndim == 2:              # 2 columns = stereo
    data = data.mean(axis=1)    # average left + right in each row → mono

t = np.arange(len(data)) / rate         # sample number → seconds (uses the file's own rate)

plt.figure(figsize=(12, 4))             # a wide picture, better for a 4-minute song
plt.plot(t, data, linewidth=0.3)
plt.title("Waveform: song.mp3")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("waveform.png", dpi=150)    # save the picture first...
plt.show()                              # ...then open the window