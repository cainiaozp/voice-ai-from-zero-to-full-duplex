"""Reproduce per-segment peak normalization artifact with synthetic audio.

Default 5s signal: quiet noise for most segments, 1s obvious event.
Optional --input WAV/FLAC (resampled to 16k). This does NOT load ESC-50 files.
"""
from pathlib import Path
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80


def make_synthetic(seed=5):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 3e-5, 5*SR).astype(np.float32)
    t = np.arange(SR)/SR
    e = .28*(np.sin(2*np.pi*250*t)+.32*np.sin(2*np.pi*500*t))
    fade = np.minimum(1,np.minimum(t/.05,(1-t)/.05)).clip(0,1)
    y[2*SR:3*SR] += (e*fade).astype(np.float32)
    return y


def dB_segment(y):
    power = librosa.feature.melspectrogram(
        y=np.asarray(y,dtype=np.float32),sr=SR, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS,center=False,power=2.0)
    return 10*np.log10(np.maximum(power, 1e-8))


def norm(db, peak):
    return np.clip(db-peak,-80,0)/80+1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='',help='Optional authorized audio clip, >=5sec')
    ap.add_argument('--out',default='outputs/esc50_compare.png')
    args=ap.parse_args()
    if args.input:
        y,_=librosa.load(args.input,sr=SR,mono=True)
        if len(y)<5*SR: y=np.pad(y,(0,5*SR-len(y)))
        y=y[:5*SR]
    else:
        y=make_synthetic()
    chunks=[y[k*SR:(k+1)*SR] for k in range(5)]
    db=[dB_segment(c) for c in chunks]
    peak=max(float(np.max(s)) for s in db)
    local=np.concatenate([norm(d,float(np.max(d))) for d in db],axis=1)
    global_ref=np.concatenate([norm(d,peak) for d in db],axis=1)
    print('local ≥0.9 ratio:',f'{(local>=.9).mean():.2%}', 'p50:',round(float(np.median(local)),3))
    print('global ≥0.9 ratio:',f'{(global_ref>=.9).mean():.2%}', 'p50:',round(float(np.median(global_ref)),3))
    for k,(loc,glob) in enumerate(zip(np.split(local,5,axis=1),np.split(global_ref,5,axis=1))):
        print(f'{k}-{k+1}s: RMS {np.sqrt(np.mean(chunks[k]**2)):.7f}, local med {np.median(loc):.3f}, full-clip med {np.median(glob):.3f}')
    fig,ax=plt.subplots(2,1,figsize=(11,6.2),sharex=True,sharey=True)
    for one,title,a in [(local,'Per-1s local peak: quiet clips can be bright',ax[0]),(global_ref,'Same 5s peak: event/background contrast preserved',ax[1])]:
        im=a.imshow(one,origin='lower',aspect='auto',extent=(0,5,0,80),vmin=0,vmax=1,cmap='magma')
        a.set_title(title);a.set_ylabel('Mel bin')
        for sec in (1,2,3,4):a.axvline(sec,ls='--',c='cyan',alpha=.7,lw=.7)
    ax[-1].set_xlabel('Time (s)');fig.colorbar(im,ax=ax,label='normalized Log-Mel',fraction=.03)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160,bbox_inches='tight')
    print('saved',out.resolve())

if __name__=='__main__': main()
