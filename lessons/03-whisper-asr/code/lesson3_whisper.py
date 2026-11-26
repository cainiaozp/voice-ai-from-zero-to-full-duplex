"""Lesson 3: Whisper encoder/decoder/cross attention experiment.

Install: pip install -U torch transformers librosa soundfile matplotlib numpy
Run: python lesson3_whisper.py --audio your_chinese_speech.wav --model openai/whisper-tiny
Options: --model openai/whisper-base, --language zh, --skip-attention, --reference 'text'

The first run downloads the Hugging Face model weights. A short speech clip (<30s) is recommended.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Whisper layer and attention inspection")
    parser.add_argument('--audio', type=Path, required=True, help='A spoken-language audio file')
    parser.add_argument('--model', default='openai/whisper-tiny', help='Multilingual Whisper checkpoint')
    parser.add_argument('--language', default='zh', help='Whisper language code')
    parser.add_argument('--reference', default=None, help='Optional ground-truth transcription')
    parser.add_argument('--skip-attention', action='store_true', help='Skip expensive attention visualization')
    parser.add_argument('--out', type=Path, default=Path('lesson3_output'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.audio.exists():
        raise FileNotFoundError(f'Input audio not found: {args.audio}')

    sr = 16000
    wave, _ = librosa.load(str(args.audio), sr=sr, mono=True)
    if len(wave) == 0:
        raise ValueError('Empty audio input')
    if len(wave) > 30 * sr:
        print('Warning: using the first 30 seconds only for this lesson')
        wave = wave[:30 * sr]
    duration = len(wave) / sr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('PCM shape:', wave.shape, 'duration:', round(duration, 3), 's')
    processor = WhisperProcessor.from_pretrained(args.model)
    # Eager mode makes it possible to inspect explicit attention matrices.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, attn_implementation='eager'
    ).to(device).eval()
    feats = processor(wave, sampling_rate=sr, return_tensors='pt').input_features.to(device)
    print('Log-Mel input:', tuple(feats.shape))

    with torch.inference_mode():
        encoder_output = model.model.encoder(input_features=feats, return_dict=True)
    audio_hidden = encoder_output.last_hidden_state
    print('Encoder output:', tuple(audio_hidden.shape))
    print('Encoder model width:', model.config.d_model)
    print('Encoder layers:', model.config.encoder_layers)
    print('Decoder layers:', model.config.decoder_layers)

    # Actual transcribed text, not a predetermined answer.
    with torch.inference_mode():
        ids = model.generate(feats, language=args.language, task='transcribe', max_new_tokens=128)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print('Generated token IDs:', ids[0].tolist())
    print('Transcription:', text)
    (args.out / 'transcript.txt').write_text(text, encoding='utf-8')

    mel = feats[0].detach().cpu().numpy()
    plt.figure(figsize=(11, 3.8))
    plt.imshow(mel, origin='lower', aspect='auto', cmap='magma', extent=[0, 30, 0, mel.shape[0]])
    plt.xlim(0, max(1, duration))
    plt.xlabel('Audio time (seconds)')
    plt.ylabel('Mel bin')
    plt.title('Whisper model input: Log-Mel')
    plt.tight_layout()
    plt.savefig(args.out / 'logmel.png', dpi=160)
    plt.close()

    if args.reference:
        with torch.inference_mode():
            targets = processor.tokenizer(args.reference, return_tensors='pt').input_ids.to(device)
            loss = model(input_features=feats, labels=targets, return_dict=True).loss
        print('Teacher-forcing reference CE loss:', float(loss))

    if not args.skip_attention:
        # Restrict text sequence length for a small visualizable matrix.
        dec_ids = ids[:, :min(ids.shape[1], 20)]
        with torch.inference_mode():
            output = model(input_features=feats,
                           decoder_input_ids=dec_ids,
                           output_attentions=True,
                           use_cache=False,
                           return_dict=True)
        cross = output.cross_attentions
        if cross is None or cross[-1] is None:
            print('Attention unavailable; check model and transformers version')
        else:
            # [batch, heads, text_positions, audio_positions]
            attention = cross[-1][0, 0].detach().float().cpu().numpy()
            print('Cross Attention last layer shape:', tuple(cross[-1].shape))
            plt.figure(figsize=(11, 4.5))
            plt.imshow(attention, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis', extent=[0, 30, -0.5, dec_ids.shape[1]-.5])
            plt.xlim(0, max(1, duration))
            plt.xlabel('Audio time (seconds)')
            plt.ylabel('Decoder token position')
            plt.title('Whisper cross attention (last decoder layer, head 0)')
            plt.colorbar(label='Attention weight')
            plt.tight_layout()
            plt.savefig(args.out / 'cross_attention.png', dpi=160)
            plt.close()
    print('Output directory:', args.out.resolve())


if __name__ == '__main__':
    main()
