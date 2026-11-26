"""Lesson 4: inspect residual vector quantization on raw PCM frames.

This is an intentionally small, NON-NEURAL VQ demonstration, not a replacement
for EnCodec/Qwen3-TTS or their learned audio encoder/decoder. Train the toy
codebooks on short waveform blocks so each RVQ stage is easy to inspect.

Install: pip install numpy scipy soundfile matplotlib
Run:     python lesson4_rvq_lab.py --input source.wav --out results
"""
import argparse
from pathlib import Path
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_audio(path, sample_rate=24000, max_seconds=6):
    audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        divisor = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, sr // divisor)
    return np.asarray(audio[:sample_rate*max_seconds], dtype=np.float32), sample_rate


def kmeans_train(vectors, k=64, iterations=10, seed=42):
    rng = np.random.default_rng(seed)
    n, d = vectors.shape
    centroids = vectors[rng.choice(n, size=k, replace=n<k)].copy()
    for _ in range(iterations):
        dist = np.sum(vectors* vectors, axis=1, keepdims=True) \
             + np.sum(centroids*centroids, axis=1)[None, :] \
             - 2 * vectors @ centroids.T
        ids = np.argmin(dist, axis=1)
        counts = np.bincount(ids, minlength=k)
        sums = np.zeros((k,d),dtype=np.float32)
        np.add.at(sums, ids, vectors)
        nonempty = counts>0
        centroids[nonempty] = sums[nonempty] / counts[nonempty,None]
        empty = ~nonempty
        if np.any(empty):
            centroids[empty] = vectors[rng.integers(n,size=np.sum(empty))]
    return centroids


def snr_db(target, estimate):
    err=np.sum(np.square(target-estimate))
    return 10*np.log10((np.sum(np.square(target)) + 1e-12)/(err + 1e-12))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path(__file__).with_name('source.wav'))
    ap.add_argument('--out',type=Path,default=Path(__file__).with_name('results'))
    ap.add_argument('--frame-size',type=int,default=32)
    ap.add_argument('--codebook-size',type=int,default=64)
    ap.add_argument('--stages',type=int,default=4)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    audio,sr=load_audio(args.input)
    sf.write(args.out/'original_24k.wav',audio,sr,subtype='PCM_16')
    frame_size=args.frame_size
    n_frames=int(np.ceil(len(audio)/frame_size))
    padded=np.pad(audio,(0,n_frames*frame_size-len(audio)))
    original_frames=padded.reshape(-1,frame_size).copy()
    residual=original_frames.copy()
    reconstructed=np.zeros_like(original_frames)
    all_codes=[]
    history=[]
    for stage in range(args.stages):
        codebook=kmeans_train(residual,k=args.codebook_size,iterations=12,seed=2026+stage)
        dist=np.sum(residual*residual,axis=1,keepdims=True)+np.sum(codebook*codebook,axis=1)[None,:]-2*residual@codebook.T
        ids=np.argmin(dist,axis=1)
        picked=codebook[ids]
        reconstructed += picked
        residual -= picked
        all_codes.append(ids.astype(np.uint16))
        output=reconstructed.reshape(-1)[:len(audio)]
        path=args.out/f'after_{stage+1}_codebooks.wav'
        sf.write(path,np.clip(output,-1,1),sr,subtype='PCM_16')
        score=snr_db(audio,output)
        history.append(score)
        print(f'RVQ {stage+1}: codes shape=[{stage+1}, {n_frames}], SNR={score:.2f} dB, WAV={path}')
    codes=np.stack(all_codes)
    np.save(args.out/'rvq_token_ids.npy',codes)
    print('First 10 timeframes (rows=codebooks):\n',codes[:,:10])
    ideal_bps=sr/frame_size * args.stages * np.ceil(np.log2(args.codebook_size))
    print(f'Ideal token indices rate: {ideal_bps:.0f} bit/s; excludes codebooks and file overhead')

    fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    # Show first 0.06 sec waveform to see local continuity
    t=np.arange(min(1440,len(audio)))/sr
    ax[0,0].plot(t,audio[:len(t)],label='original',linewidth=.9)
    ax[0,0].plot(t,reconstructed.reshape(-1)[:len(t)],label=f'{args.stages} codebooks',linewidth=.7,alpha=.7)
    ax[0,0].set(title='Waveform (first 60ms)',xlabel='Time (s)',ylabel='Amplitude');ax[0,0].legend()
    ax[0,1].plot(np.arange(1,args.stages+1),history,marker='o')
    ax[0,1].set(title='Reconstruction improves with RVQ stages',xlabel='Number of codebooks',ylabel='SNR (dB)')
    ax[0,1].set_xticks(np.arange(1,args.stages+1));ax[0,1].grid(alpha=.2)
    # codes rows plot
    ax[1,0].imshow(codes[:,:min(60,codes.shape[1])],aspect='auto',cmap='viridis',interpolation='nearest')
    ax[1,0].set(title='Discretized code IDs (first 60 frames)',xlabel='Time frame',ylabel='Codebook index')
    f,tt,z = stft(audio,fs=sr,nperseg=512,noverlap=384)
    mask=(f<=4000)
    ax[1,1].pcolormesh(tt,f[mask],20*np.log10(np.maximum(abs(z[mask]),1e-6)),shading='auto',cmap='magma',vmin=-85,vmax=-10)
    ax[1,1].set(title='Original audio log-frequency magnitude',xlabel='Time (s)',ylabel='Frequency (Hz)')
    fig.savefig(args.out/'rvq_comparison.png',dpi=165)
    plt.close(fig)
    print('Plot:',args.out/'rvq_comparison.png')

if __name__=='__main__':
    main()
