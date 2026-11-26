"""Lesson 5: a waveform, STFT phase, magnitude and Mel inversion.

Run: pip install numpy librosa soundfile matplotlib
     python lesson5_vocoder_lab.py --input voice_example.wav

This file demonstrates classical signal inversion only. It does NOT run or
train a neural vocoder (HiFi-GAN).
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_mono(path, wave, sr):
    sf.write(str(path), np.clip(np.asarray(wave), -1, 1), sr, subtype='PCM_16')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='WAV/FLAC etc. speech recording')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--iterations', type=int, default=32)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sr = 16000
    y, _ = librosa.load(args.input, sr=sr, mono=True)
    y = y.astype(np.float32)
    # The audio waveform is the exact target. No text or class label is needed.
    n_fft, hop, n_mels = 1024, 256, 80
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True)
    magnitude = np.abs(stft)
    power = magnitude**2
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Exact complex STFT inversion: magnitude AND phase are retained.
    exact = librosa.istft(stft, hop_length=hop, win_length=n_fft, center=True, length=len(y))

    # Only magnitude is retained. Phase is estimated iteratively with Griffin-Lim.
    no_phase = librosa.griffinlim(magnitude, n_iter=args.iterations,
                                  hop_length=hop, win_length=n_fft,
                                  length=len(y))

    # Mel inversion: approximates linear-frequency magnitude before Griffin-Lim.
    from_mel = librosa.feature.inverse.mel_to_audio(
        mel, sr=sr, n_fft=n_fft, hop_length=hop,
        win_length=n_fft, fmax=8000, n_iter=args.iterations,
        length=len(y))

    outputs = {
        '01_original.wav': y,
        '02_complex_stft_istft.wav': exact,
        '03_magnitude_griffinlim.wav': no_phase,
        '04_mel_griffinlim.wav': from_mel,
    }
    for filename, audio in outputs.items():
        save_mono(out/filename, audio, sr)
        print(f'{filename}: {len(audio)/sr:.2f}s, {sr}Hz, RMS={np.sqrt(np.mean(audio**2)):.4f}')
    print('PCM:', y.shape, 'complex STFT:', stft.shape, 'Mel:', mel.shape)
    print('Complex STFT inverse max absolute error:', np.max(np.abs(y-exact)))

    fig, ax = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    t = np.arange(len(y))/sr
    for i, (name, a) in enumerate(outputs.items()):
        ax[i].plot(t, a, lw=.5)
        ax[i].set_title(name.replace('.wav','').replace('_',' '))
        ax[i].set_ylim(-1.05,1.05)
        ax[i].set_ylabel('Amplitude')
        ax[i].grid(alpha=.18)
    ax[-1].set_xlabel('Time (seconds)')
    fig.savefig(out/'waveforms_comparison.png', dpi=145)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13,4), constrained_layout=True)
    im1=ax[0].imshow(librosa.amplitude_to_db(magnitude,ref=np.max),origin='lower',aspect='auto',
                     extent=[0,len(y)/sr,0,sr/2],cmap='magma',vmin=-80,vmax=0)
    ax[0].set(title='STFT magnitude (linear frequency)', xlabel='Time (s)',ylabel='Hz')
    fig.colorbar(im1,ax=ax[0],label='dB relative')
    im2=ax[1].imshow(log_mel,origin='lower',aspect='auto',
                     extent=[0,len(y)/sr,0,n_mels],cmap='magma',vmin=-80,vmax=0)
    ax[1].set(title='Log-Mel spectrogram (80 bands)',xlabel='Time (s)',ylabel='Mel bin')
    fig.colorbar(im2,ax=ax[1],label='dB relative')
    fig.savefig(out/'stft_vs_logmel.png', dpi=145)
    plt.close(fig)

if __name__=='__main__':
    main()
