"""Lesson 1: PCM, FFT, STFT, log-Mel, MFCC. CPU only.
Usage: python lesson1.py [voice.wav]
Outputs are saved next to this script under results/.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
N_MFCC = 13
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

# 1. Generate three simple experimental signals.
t = np.arange(SR, dtype=np.float64) / SR
pure = 0.4 * np.sin(2 * np.pi * 440 * t)
mixed = (0.4 * np.sin(2 * np.pi * 220 * t)
         + 0.2 * np.sin(2 * np.pi * 440 * t)
         + 0.1 * np.sin(2 * np.pi * 880 * t))
# Consecutive tones, with a short fade at segment boundaries.
fade = np.ones(SR)
ramp = np.linspace(0, 1, 160)
fade[:160], fade[-160:] = ramp, ramp[::-1]
sequence = np.concatenate([0.4 * np.sin(2*np.pi*f*t) * fade
                           for f in (220, 440, 880)])
for name, x in [('pure_440', pure), ('mixed', mixed), ('sequence', sequence)]:
    sf.write(OUT / f'{name}.wav', x, SR, subtype='PCM_16')

# 2. PCM: read the actual stored integers.
pcm, sr = sf.read(OUT / 'pure_440.wav', dtype='int16')
print('PCM:', pcm.shape, pcm.dtype, 'sample rate:', sr)
print('First 10 samples:', pcm[:10])
print('Normalized:', pcm[:10].astype(np.float32) / 32768)

# 3. FFT: compare single- and multi-frequency tones.
fig, ax = plt.subplots(2, 2, figsize=(11, 5), layout='constrained')
for i, (name, x) in enumerate([('pure_440', pure), ('mixed', mixed)]):
    x, _ = sf.read(OUT / f'{name}.wav')
    f = np.fft.rfftfreq(len(x), d=1/SR)
    a = 2 * np.abs(np.fft.rfft(x)) / len(x)
    a[0] /= 2
    if len(x) % 2 == 0:
        a[-1] /= 2
    ids = np.argsort(a)[-3:][::-1]
    print(name, 'top FFT peaks:', [(round(f[j],1),round(a[j],4)) for j in ids])
    n = int(SR * 0.02)
    ax[i,0].plot(np.arange(n)/SR*1000, x[:n], linewidth=1.3)
    ax[i,0].set(title=f'{name}: first 20 ms', xlabel='Time (ms)', ylabel='Amplitude', ylim=(-.75,.75))
    ax[i,1].plot(f, a, linewidth=1.3)
    ax[i,1].set(title='One-sided amplitude spectrum', xlabel='Frequency (Hz)', ylabel='Amplitude', xlim=(0,1300), ylim=(0,.46))
    for j in ids:
        if a[j] > .01:
            ax[i,1].annotate(f'{f[j]:.0f} Hz', (f[j], a[j]), xytext=(7,3), textcoords='offset points')
fig.savefig(OUT / 'fft.png', dpi=150)
plt.close(fig)

# 4. STFT -> Mel -> MFCC using the SAME frames for all plots.
def analyze(x, name):
    if len(x) < N_FFT:
        raise ValueError('Audio must be at least 25 ms long.')
    D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP,
                     win_length=N_FFT, window='hann', center=False)
    power = np.abs(D) ** 2
    mel_filters = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    mel_power = mel_filters @ power
    log_mel = librosa.power_to_db(mel_power, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    print(f'{name}: waveform={x.shape}, STFT={D.shape}, mel_filters={mel_filters.shape}, log_mel={log_mel.shape}, MFCC={mfcc.shape}')
    np.savez_compressed(OUT / f'{name}_features.npz', waveform=x, sample_rate=SR, stft=D, log_mel=log_mel, mfcc=mfcc)
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), layout='constrained')
    ax[0].plot(np.arange(len(x))/SR, x, linewidth=.65)
    ax[0].set(title=f'{name}: waveform', ylabel='Amplitude', xlim=(0,len(x)/SR))
    opts = dict(sr=SR, hop_length=HOP, x_axis='time')
    # center=False: centers of frames occur 12.5ms after the frame starts.
    tc = (np.arange(D.shape[1])*HOP+N_FFT/2)/SR
    im = librosa.display.specshow(librosa.power_to_db(power, ref=np.max), y_axis='linear', x_coords=tc, ax=ax[1], **opts)
    ax[1].set(title='STFT power spectrum (dB)', ylim=(0,4000))
    fig.colorbar(im, ax=ax[1], format='%+2.0f dB')
    im = librosa.display.specshow(log_mel, y_axis='mel', x_coords=tc, ax=ax[2], **opts)
    ax[2].set(title='Log-Mel spectrum (80 filters)')
    fig.colorbar(im, ax=ax[2], format='%+2.0f dB')
    im = librosa.display.specshow(mfcc, x_coords=tc, ax=ax[3], **opts)
    ax[3].set(title='MFCC (13 coefficients)', ylabel='Coefficient')
    fig.colorbar(im, ax=ax[3])
    fig.savefig(OUT / f'{name}_analysis.png', dpi=150)
    plt.close(fig)

analyze(pure, 'pure_440')
analyze(sequence, 'sequence')
if len(sys.argv)>1:
    voice, _ = librosa.load(sys.argv[1], sr=SR, mono=True)
    analyze(voice, 'voice')
print('Saved to', OUT)
