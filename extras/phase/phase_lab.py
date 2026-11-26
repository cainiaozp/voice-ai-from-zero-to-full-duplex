"""Phase listening experiment: same 440 Hz sine, shifted phase, and sums.

Outputs synthetic WAV and comparison PNG. No model weights or private audio.
"""
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='outputs/phase')
    p.add_argument('--hz', type=float, default=440)
    p.add_argument('--duration', type=float, default=2.0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.arange(int(sr * args.duration)) / sr
    fade = np.minimum(1, np.minimum(t / .025, (args.duration-t)/.025)).clip(0,1)
    amp = .22  # even same-phase sum won't clip
    a = amp * np.sin(2 * np.pi * args.hz * t) * fade
    sf.write(out/'single_0.wav', a, sr)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    shown = int(sr * 5 / args.hz)
    for deg in (0, 90, 180, 360):
        b = amp * np.sin(2 * np.pi * args.hz * t + np.deg2rad(deg)) * fade
        # ensure same envelope, avoid apparent clicks at ends
        sf.write(out/f'single_{deg}.wav', b, sr)
        total = a+b
        sf.write(out/f'sum_{deg}.wav', total, sr)
        rms = float(np.sqrt(np.mean(total[round(.2*sr):round(1.5*sr)]**2)))
        print(f'phase={deg:3d}°, sum RMS={rms:.8f}, relative amplitude={2*abs(np.cos(np.deg2rad(deg)/2)):.3f}×')
        ax[0].plot(t[:shown]*1000,b[:shown],label=f'{deg}°')
        ax[1].plot(t[:shown]*1000,total[:shown],label=f'{deg}°')
    ax[0].plot(t[:shown]*1000,a[:shown],color='k',lw=1.5,label='reference A')
    ax[0].set_ylabel('Individual amplitude')
    ax[1].set_ylabel('Summed amplitude')
    ax[1].set_xlabel('Time (ms)')
    for x in ax: x.legend(ncol=5,fontsize=8); x.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'phase_comparison.png',dpi=150)
    print('Saved', out.resolve())

if __name__ == '__main__':
    main()
