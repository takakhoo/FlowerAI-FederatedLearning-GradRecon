#!/usr/bin/env python3
import os, csv, numpy as np, soundfile as sf

root = "/scratch2/f004h1v/flower-ASR/flower/baselines/fedwav2vec2"
datadir = os.path.join(root, "data", "client_0")
os.makedirs(datadir, exist_ok=True)

# Make a 2-second 16kHz sine wave
sr = 16000
t = np.linspace(0, 2.0, int(sr*2.0), endpoint=False)
y = 0.1*np.sin(2*np.pi*440.0*t).astype("float32")
wav_path = os.path.join(datadir, "tiny.wav")
sf.write(wav_path, y, sr)

# Minimal CSVs expected by dataset.py
rows = [
    {"ID": "utt0", "wav": wav_path, "start_seg":"0.0", "end_seg":"2.0", "char":"HELLO WORLD", "duration":"2.0"}
]
for split in ["ted_train.csv","ted_dev.csv","ted_test.csv"]:
    with open(os.path.join(datadir, split), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID","wav","start_seg","end_seg","char","duration"])
        w.writeheader(); w.writerows(rows)

print("Tiny client created under", datadir)


