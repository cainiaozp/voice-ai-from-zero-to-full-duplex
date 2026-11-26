"""Run a REAL pretrained VAD on a real voice recording.
Install: pip install torch silero-vad soundfile numpy scipy
Run: python silero_real_vad.py your_16k_or_48k_mono_audio.wav
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from silero_vad import load_silero_vad, get_speech_timestamps

if len(sys.argv) < 2:
    raise SystemExit('usage: python silero_real_vad.py YOUR_RECORDING.wav')
path = Path(sys.argv[1])
y, sr = sf.read(str(path),dtype='float32')
if y.ndim == 2:
    y = y.mean(axis=1)
if sr != 16000:
    g = gcd(sr, 16000)
    y = resample_poly(y,16000//g,sr//g)
model = load_silero_vad()
stamps = get_speech_timestamps(torch.from_numpy(np.asarray(y,dtype=np.float32)),model,
                              sampling_rate=16000,return_seconds=True)
print('Speech segments:',stamps)
print('Reminder: these are speech intervals, NOT end-of-turn decisions.')
