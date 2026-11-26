"""Lesson 10: reproducible full-duplex session & KV-cache laboratory.

This is a control-plane/dataflow simulation, NOT a speech understanding model.
Only NumPy and matplotlib are needed. 20ms IO frames; 80ms synthesized audio chunks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 16000
FRAME_MS = 20
N = RATE * FRAME_MS // 1000
TOTAL_MS = 6000
AUDIO_CHUNK_MS = 80
AUDIO_CHUNK_FRAMES = AUDIO_CHUNK_MS // FRAME_MS


class CausalKV:
    """A two-layer, two-head toy causal attention model with real K/V tensors."""
    def __init__(self, max_frames: int, seed: int = 8):
        rng = np.random.default_rng(seed)
        self.dim = 16
        self.heads = 2
        self.dh = 8
        self.n_layers = 2
        self.weights = [tuple(rng.normal(0, 0.1, (16, 16)) for _ in range(4)) for _ in range(self.n_layers)]
        self.k = np.zeros((self.n_layers, self.heads, max_frames, self.dh))
        self.v = np.zeros_like(self.k)
        self.length = 0

    def one_frame(self, x: np.ndarray) -> np.ndarray:
        pos = self.length
        for layer, (wq, wk, wv, wo) in enumerate(self.weights):
            q = (x @ wq).reshape(self.heads, self.dh)
            self.k[layer, :, pos, :] = (x @ wk).reshape(self.heads, self.dh)
            self.v[layer, :, pos, :] = (x @ wv).reshape(self.heads, self.dh)
            scores = np.einsum('hd,htd->ht', q, self.k[layer, :, :pos + 1, :]) / math.sqrt(self.dh)
            scores -= scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn /= attn.sum(axis=-1, keepdims=True)
            z = np.einsum('ht,htd->hd', attn, self.v[layer, :, :pos + 1, :]).reshape(self.dim)
            x = x + z @ wo
        self.length += 1
        return x

    def full_causal(self, xseq: np.ndarray) -> np.ndarray:
        """Slow reference: recompute causal attention for all sequence positions."""
        x = xseq.copy()
        for (wq, wk, wv, wo) in self.weights:
            q = (x @ wq).reshape(len(x), self.heads, self.dh)
            k = (x @ wk).reshape(len(x), self.heads, self.dh)
            v = (x @ wv).reshape(len(x), self.heads, self.dh)
            outputs = []
            for t in range(len(x)):
                scores = np.einsum('hd,thd->ht', q[t], k[:t+1]) / math.sqrt(self.dh)
                scores -= scores.max(axis=-1, keepdims=True)
                a = np.exp(scores)
                a /= a.sum(axis=-1, keepdims=True)
                z = np.einsum('ht,thd->hd', a, v[:t+1]).reshape(self.dim)
                outputs.append(x[t] + z @ wo)
            x = np.asarray(outputs)
        return x


def input_kind(t_ms: int) -> str:
    # Two-part user question (mid-sentence pause), acknowledgement, real interruption.
    if 320 <= t_ms < 1360: return 'question_part_1'
    if 1620 <= t_ms < 2200: return 'question_part_2'
    if 2880 <= t_ms < 3040: return 'backchannel'
    if 3240 <= t_ms < 3560: return 'interrupt'
    return 'silence'


def input_signal(t_ms: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    sec = (t_ms + np.arange(N) * 1000 / RATE) / 1000
    if kind == 'silence': return np.zeros(N)
    hz = {'question_part_1': 190, 'question_part_2': 240,
          'backchannel': 290, 'interrupt': 360}[kind]
    return np.clip(0.19*np.sin(2*np.pi*hz*sec) + 0.07*np.sin(2*np.pi*(2*hz)*sec)
                   + rng.normal(0, 0.003, N), -1, 1)


def output_signal(t_ms: int, generation: int) -> np.ndarray:
    sec = (t_ms + np.arange(N) * 1000 / RATE) / 1000
    hz = 460 if generation == 1 else 570
    return 0.15*np.sin(2*np.pi*hz*sec) + 0.04*np.sin(2*np.pi*hz*2*sec)


def features(frame: np.ndarray, t_ms: int) -> np.ndarray:
    rms = float(np.sqrt(np.mean(frame * frame)))
    zero_cross = float(np.mean(frame[:-1]*frame[1:] < 0)) if len(frame) > 1 else 0
    return np.asarray([
        rms, zero_cross, t_ms/1000, np.sin(t_ms*0.001), np.cos(t_ms*.001),
        rms**2, np.log1p(rms), rms*2, zero_cross**2, 1., 0., 0., 0., 0., 0., 0.
    ], dtype=np.float64)


def write_wav(out: Path, left: np.ndarray, right: np.ndarray):
    pcm = np.stack([left, right], axis=1)
    pcm = np.int16(np.round(np.clip(pcm, -1, 1) * 32767))
    with wave.open(str(out), 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())


def run(out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(7)
    model = CausalKV(max_frames=TOTAL_MS//FRAME_MS)
    input_vecs = []
    incremental = []
    user_audio = []
    bot_audio = []
    rows = []
    events = []
    play_buffer = deque()  # 20ms PCM frames; bounded at 160ms.
    generation = 0
    active = False
    first_audio_after_turn = {}
    last_turn_finished = {}
    actual_barge_ms = None
    stop_ms = None
    dropped_ms = 0
    max_buffer_ms = 0
    output_frames_during_user_speech = 0
    state = 'LISTEN'
    next_chunk_at = None
    chunks_generated = 0

    for t_ms in range(0, TOTAL_MS, FRAME_MS):
        kind = input_kind(t_ms)
        user_frame = input_signal(t_ms, kind, rng)
        user_audio.append(user_frame)
        x = features(user_frame, t_ms)
        input_vecs.append(x)
        incremental.append(model.one_frame(x))
        event = ''

        # This is a deliberately scripted teacher/oracle, not a learned turn detector.
        # It avoids false early cutoff during the first 260ms pause.
        if t_ms == 2400:
            generation += 1
            last_turn_finished[generation] = 2200
            active = True
            state = 'SPEAK'
            next_chunk_at = 2480  # 80ms artificial first-chunk computation
            chunks_generated = 0
            event = 'TURN_END_1'
        if kind == 'backchannel' and t_ms == 2880:
            event = 'BACKCHANNEL_IGNORED'
        if kind == 'interrupt' and t_ms == 3240:
            actual_barge_ms = t_ms
            stop_ms = t_ms
            dropped_ms += FRAME_MS * len(play_buffer)
            play_buffer.clear()
            active = False
            state = 'LISTEN'
            next_chunk_at = None
            event = 'BARGE_IN_CANCEL_1'
        if t_ms == 3760:
            generation += 1
            last_turn_finished[generation] = 3560
            active = True
            state = 'SPEAK'
            next_chunk_at = 3840
            chunks_generated = 0
            event = 'TURN_END_2'

        # Simulate an 80ms codec block becoming available every 80ms.
        # Never allow more than 160ms of audio queued for playback.
        if (active and next_chunk_at is not None and t_ms >= next_chunk_at
                and len(play_buffer) <= 4 and chunks_generated < 23):
            for j in range(AUDIO_CHUNK_FRAMES):
                play_buffer.append((generation, output_signal(t_ms + j*FRAME_MS, generation)))
            chunks_generated += 1
            next_chunk_at += AUDIO_CHUNK_MS
        max_buffer_ms = max(max_buffer_ms, FRAME_MS * len(play_buffer))

        if play_buffer:
            frame_gen, bot_frame = play_buffer.popleft()
            if frame_gen not in first_audio_after_turn:
                first_audio_after_turn[frame_gen] = t_ms
            if kind != 'silence': output_frames_during_user_speech += 1
        else:
            bot_frame = np.zeros(N)

        bot_audio.append(bot_frame)
        if event:
            events.append({'t_ms': t_ms, 'event': event})
        rows.append({'t_ms': t_ms, 'user_state': kind, 'session_state': state,
                     'speech_active': int(kind != 'silence'),
                     'playback_active': int(bool(np.any(bot_frame))),
                     'generation_id': generation, 'kv_frames_per_layer': model.length,
                     'playback_buffer_ms': FRAME_MS * len(play_buffer), 'event': event})

    xseq = np.asarray(input_vecs)
    ref = model.full_causal(xseq)
    cache_error = float(np.max(np.abs(ref - np.asarray(incremental))))
    assert cache_error < 1e-10, f'Cache did not match: {cache_error}'
    assert first_audio_after_turn[1] == 2480, first_audio_after_turn
    assert first_audio_after_turn[2] == 3840, first_audio_after_turn
    assert stop_ms == 3240, stop_ms
    assert any(r['event'] == 'BACKCHANNEL_IGNORED' for r in rows)
    assert output_frames_during_user_speech > 0

    write_wav(out_dir/'simulated_duplex_stereo.wav', np.concatenate(user_audio), np.concatenate(bot_audio))
    with open(out_dir/'frames.csv', 'w', newline='', encoding='utf8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out_dir/'events.json', 'w', encoding='utf8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    summary = {
        'frame_ms': FRAME_MS, 'audio_chunk_ms': AUDIO_CHUNK_MS,
        'frames_processed': model.length,
        'kv_shape_each_layer': [1, model.heads, model.length, model.dh],
        'cached_vs_full_attention_max_abs_error': cache_error,
        'first_audio_latency_from_scripted_user_end_ms': {str(g): first_audio_after_turn[g] - last_turn_finished[g] for g in sorted(first_audio_after_turn)},
        'interrupt_to_audio_stop_ms': stop_ms - actual_barge_ms,
        'max_queued_playback_ms': max_buffer_ms,
        'stale_buffer_cleared_ms': dropped_ms,
        'overlap_frames_playback_and_user_audio': output_frames_during_user_speech,
        'notice': 'Synthetic PCM, scripted turn/interruption labels and virtual time; not measured model or network latency.'
    }
    with open(out_dir/'metrics.json', 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = np.asarray([r['t_ms']/1000 for r in rows])
        u = np.asarray([r['speech_active'] for r in rows])
        a = np.asarray([r['playback_active'] for r in rows])
        buff = np.asarray([r['playback_buffer_ms'] for r in rows])
        fig, axes = plt.subplots(3,1,figsize=(11,7),sharex=True,constrained_layout=True)
        axes[0].step(t,u,where='post',label='User microphone speech',lw=2)
        axes[0].step(t,a,where='post',label='Assistant playback',lw=2)
        axes[0].set_ylim(-.15,1.35)
        axes[0].set_ylabel('Audio activity')
        axes[0].legend(loc='upper right')
        axes[1].step(t,buff,where='post')
        axes[1].set_ylabel('Play queue (ms)')
        axes[1].set_ylim(0,200)
        axes[2].plot(t,np.asarray([r['kv_frames_per_layer'] for r in rows]),lw=2)
        axes[2].set_ylabel('KV length / layer')
        axes[2].set_xlabel('Virtual time (s)')
        for e in events:
            for ax in axes:
                ax.axvline(e['t_ms']/1000, ls=':', alpha=.35)
        fig.suptitle('Lesson 10: duplex streaming, barge-in, playback queue and causal KV')
        fig.savefig(out_dir/'timeline.png',dpi=160)
        plt.close(fig)
    except ImportError:
        print('matplotlib unavailable: CSV/WAV/JSON are still generated.')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Files written to:', out_dir)
    return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'results')
    args = p.parse_args()
    run(args.out)
