"""Lesson 4: pretrained REAL EnCodec 24kHz neural codec, optional experiment.

Install (ensure torchaudio matches your torch):
    pip install -U encodec soundfile torchaudio torch matplotlib
Run:
    python lesson4_encodec_real.py --input source.wav --bandwidth 6

The first run downloads Meta's pretrained model weights via Torch Hub.
Requires network and may encounter platform-specific torchaudio dependencies.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from encodec import EncodecModel
from encodec.utils import convert_audio


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path(__file__).with_name('source.wav'))
    parser.add_argument('--bandwidth',type=float,choices=[1.5,3.0,6.0,12.0,24.0],default=6.0)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('encodec_reconstructed.wav'))
    args=parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=EncodecModel.encodec_model_24khz().to(device)
    model.set_target_bandwidth(args.bandwidth)
    model.eval()
    wav,sr=torchaudio.load(str(args.input))
    wav=convert_audio(wav,sr,model.sample_rate,model.channels)
    wav=wav.unsqueeze(0).to(device) # [B=1, C=1, samples]
    with torch.inference_mode():
        frames=model.encode(wav)
        codes=torch.cat([frame[0] for frame in frames],dim=-1)
        decoded=model.decode(frames)[...,:wav.shape[-1]]
    codes_cpu=codes.cpu()
    print('Input waveform:',tuple(wav.shape))
    print('Discrete codes:',tuple(codes_cpu.shape),'[B, num_codebooks, timesteps]')
    print('First 12 frames, first codebook:', codes_cpu[0,0,:12].tolist())
    print('Reconstructed:',tuple(decoded.shape))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(args.output, decoded[0,0].cpu().numpy(),model.sample_rate,subtype='PCM_16')
    np.save(args.output.with_suffix('.npy'),codes_cpu.numpy())
    print('Reconstructed wav:', args.output)
    print('Codebook token IDs:',args.output.with_suffix('.npy'))

if __name__=='__main__':
    main()
