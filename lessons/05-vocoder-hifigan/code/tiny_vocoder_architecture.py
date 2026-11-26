"""Minimal educational neural vocoder architecture (NOT a pretrained model).

Goal: demonstrate how [B,80,T] Mel frames are upsampled into [B,1,T*160] waveform samples.
This untrained network cannot synthesize meaningful speech. Do not play its random output at volume.
"""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.short = nn.Conv1d(channels, channels, 3, padding=1)
        self.long = nn.Conv1d(channels, channels, 7, padding=3)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return x + 0.5*(self.short(self.activation(x)) + self.long(self.activation(x)))

class TinyVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv1d(80, 128, 7, padding=3)
        self.upsample1 = nn.ConvTranspose1d(128, 64, kernel_size=20, stride=10, padding=5)
        self.residual1 = ResidualBlock(64)
        self.upsample2 = nn.ConvTranspose1d(64, 32, kernel_size=32, stride=16, padding=8)
        self.residual2 = ResidualBlock(32)
        self.output_projection = nn.Conv1d(32, 1, 7, padding=3)

    def forward(self, mel, verbose=False):
        x = self.input_projection(mel)
        if verbose: print('input conv: ', tuple(x.shape))
        x = self.residual1(torch.nn.functional.leaky_relu(self.upsample1(x), .2))
        if verbose: print('upsample x10:', tuple(x.shape))
        x = self.residual2(torch.nn.functional.leaky_relu(self.upsample2(x), .2))
        if verbose: print('upsample x16:', tuple(x.shape))
        wave = torch.tanh(self.output_projection(x))
        if verbose: print('PCM output: ', tuple(wave.shape))
        return wave

if __name__ == '__main__':
    model = TinyVocoder()
    example = torch.randn(1,80,100)
    print('Mel input:  ', tuple(example.shape))
    generated = model(example, verbose=True)
    print('Parameter count:', sum(p.numel() for p in model.parameters()))
    print('NOTE: random weights; output is NOT meaningful speech.')
    # Test that network weights do receive gradients.
    loss = generated.abs().mean()
    loss.backward()
    print('Backward pass OK.')
