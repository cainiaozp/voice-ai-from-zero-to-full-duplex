"""Lesson 8: VAD != end-of-turn. Run: python lesson8_turn_lab.py

Requires numpy, matplotlib, soundfile.
Produces a *synthetic harmonic signal*, NOT understandable Chinese speech.
Transcript fragments below are manual annotations, not STT model outputs.
The semantic turn policy is a teaching heuristic, not a trained turn detector.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
SR, DURATION = 16000, 7.5
rng = np.random.default_rng(2026)
t = np.arange(int(SR * DURATION)) / SR
wave = rng.normal(0, 0.0005, size=len(t))
# The user pauses twice mid-sentence. Audio is illustrative non-speech harmonics.
segments = [
    (0.40, 1.45, 'I want to ask... / 我想问一下……', 'continue'),
    (2.35, 3.35, 'about Whisper... / 关于 Whisper……', 'continue'),
    (4.05, 5.25, 'Why cross attention? / 为什么使用交叉注意力？', 'finished'),
    (6.35, 6.78, 'Wait, not that / 等一下，不是那个', 'barge_in'),
]
for k, (a, b, _, _) in enumerate(segments):
    sel = (t >= a) & (t < b)
    local = t[sel] - a
    length = b - a
    f0 = (140 + 32*k) + 10*np.sin(2*np.pi*2.6*local)
    phase = 2*np.pi * np.cumsum(f0)/SR
    envelope = np.minimum(1, local/.025) * np.minimum(1, (length-local)/.035)
    syllable = .6 + .4*np.sin(2*np.pi*4.4*local)**2
    sound = .19 * envelope * syllable * (
        .72*np.sin(phase) + .23*np.sin(2*phase) + .1*np.sin(3*phase))
    wave[sel] += sound
wave = np.clip(wave, -1, 1)
sf.write(OUT / 'synthetic_turns.wav', wave, SR, subtype='PCM_16')

# 20-ms frames, energy gating (NOT learned VAD, and may trigger on other sounds)
frame_ms = 20
n = SR * frame_ms // 1000
frames = wave[:len(wave)//n*n].reshape(-1, n)
rms = np.sqrt(np.mean(frames**2, axis=1))
frame_centers = (np.arange(len(frames)) + .5) * frame_ms/1000
threshold = 0.014
vad = rms > threshold

def crossings(binary):
    padded = np.r_[False, binary, False].astype('int')
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(round(i*.02, 2), round(j*.02, 2)) for i, j in zip(starts, ends)]

raw_speech_segments = crossings(vad)
# Fixed endpointing policy: wait 450 ms after speech activity ends.
# Only include pauses before a next utterance, not at the end of the recording.
speech_intervals = [(a,b) for a,b,_,_ in segments]
naive = [(round(b + .45, 2), 'turn_commit', 'naive fixed-silence threshold') for a,b in speech_intervals[:3]]
# Semantic timeline is MANUALLY annotated to demonstrate what the learned model would aim to predict.
semantic = [(1.45,'continue_listening','sentence unfinished'),
            (3.35,'continue_listening','question unfinished'),
            (5.51,'turn_commit','question complete; 260-ms debounce'),
            (6.40,'interrupt_assistant','new user request detected during playback')]

with (OUT/'events.csv').open('w', newline='', encoding='utf-8') as fh:
    wr = csv.writer(fh)
    wr.writerow(['time_s','policy','decision','reason'])
    for tm,event,reason in naive:
        wr.writerow([tm,'fixed_silence',event,reason])
    for tm,event,reason in semantic:
        wr.writerow([tm,'semantic_demo',event,reason])

fig, axes = plt.subplots(4,1,figsize=(12,8.7),sharex=True,
                          gridspec_kw={'height_ratios':[2,1.2,1.0,1.2]})
fig.suptitle('Lesson 8 - VAD detects activity, turn detection decides when to respond', fontsize=13)
axes[0].plot(t[::8], wave[::8],lw=.55)
axes[0].set_ylabel('PCM amplitude')
for a,b,txt,tag in segments:
    axes[0].axvspan(a,b,color='tab:blue',alpha=.12)
    axes[0].text((a+b)/2,.32,txt.split(' / ')[0],ha='center',fontsize=8)
axes[0].set_ylim(-.29,.39)
axes[1].plot(frame_centers,rms,lw=1.2,label='frame RMS')
axes[1].axhline(threshold,ls='--',color='tab:red',label='energy threshold')
axes[1].fill_between(frame_centers,0,rms,where=vad,alpha=.22)
axes[1].set_ylabel('Energy / VAD')
axes[1].legend(fontsize=8,loc='upper right')
axes[2].set_ylim(-.2,1.1)
axes[2].set_yticks([.45],['naive'])
axes[2].hlines(.45,0,DURATION,color='0.7',lw=2)
for tm,_,_ in naive:
    axes[2].plot(tm,.45,marker='x',ms=11,mew=3,c='tab:red')
    axes[2].text(tm,.75,f'{tm:.2f}s',ha='center',color='tab:red',fontsize=9)
axes[2].set_title('Naive: fixed 450-ms silence -> false cutoffs in mid-turn pauses', fontsize=10)
axes[3].set_ylim(0,1.5)
axes[3].set_yticks([.55],['context-aware'])
axes[3].hlines(.55,0,DURATION,color='0.7',lw=2)
for tm,decision,_ in semantic:
    color = 'tab:green' if decision=='turn_commit' else 'tab:orange' if decision=='interrupt_assistant' else 'tab:blue'
    axes[3].plot(tm,.55,marker='o',ms=9,color=color)
    axes[3].text(tm, .90 if decision!='turn_commit' else .20,decision.replace('_','\n'),ha='center',va='bottom',fontsize=8,color=color)
axes[3].axvspan(5.51,6.4,color='tab:green',alpha=.08)
axes[3].axvspan(6.4,7.2,color='tab:red',alpha=.07)
axes[3].set_xlabel('Time (seconds)')
axes[3].set_xlim(0,DURATION)
fig.tight_layout()
fig.savefig(OUT / 'turn_timeline.png',dpi=155,bbox_inches='tight')
plt.close(fig)

print('Audio:',OUT/'synthetic_turns.wav')
print('Plot:',OUT/'turn_timeline.png')
print('Frame size:',n,'samples /',frame_ms,'ms; total',len(frames),'frames')
print('Detected speech intervals (illustrative energy VAD):',raw_speech_segments)
print('\nNaive 450ms silence policy:')
for r in naive: print(' ',r)
print('\nContext-aware teaching policy (MANUAL transcript states; not an ML model):')
for r in semantic: print(' ',r)
