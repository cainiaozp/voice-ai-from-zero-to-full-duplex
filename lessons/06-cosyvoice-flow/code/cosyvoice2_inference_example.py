"""Run INSIDE the official QwenAudio/CosyVoice repository with pretrained CosyVoice2.

Dependencies and weights: see https://github.com/QwenAudio/CosyVoice
A reference WAV and its exact transcript are required for zero-shot voice cloning.
This script has not been exercised in this offline environment.

Example:
  python /path/to/cosyvoice_lesson6/cosyvoice2_inference_example.py \
    --model pretrained_models/CosyVoice2-0.5B \
    --prompt-wav asset/zero_shot_prompt.wav \
    --prompt-text '希望你以后能够做的比我还好呦。' \
    --text '你好，我正在学习语音合成模型。'
"""
import argparse
import sys
from pathlib import Path

# Necessary when run from the official repo root (submodule checkout)
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='pretrained_models/CosyVoice2-0.5B')
    p.add_argument('--prompt-wav',required=True)
    p.add_argument('--prompt-text',required=True)
    p.add_argument('--text',default='你好，我正在学习语音合成模型。')
    p.add_argument('--stream',action='store_true')
    p.add_argument('--out',default='cosyvoice2_results')
    a=p.parse_args()
    model=AutoModel(model_dir=a.model)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for i,res in enumerate(model.inference_zero_shot(
        a.text,a.prompt_text,a.prompt_wav,stream=a.stream
    )):
        speech=res['tts_speech'].cpu()
        filename=out/f'piece_{i:03d}.wav'
        torchaudio.save(str(filename),speech,model.sample_rate)
        print('saved',filename,'sample_rate',model.sample_rate,'shape',tuple(speech.shape))

if __name__=='__main__':main()
