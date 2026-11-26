"""Minimal VQ-VAE training demonstration.

Runs offline on synthetic RGB shapes; optionally trains on torchvision CIFAR-10.

Usage:
  python train_vqvae.py --epochs 3 --dataset synthetic
  python train_vqvae.py --epochs 10 --dataset cifar10
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


class SyntheticRGB(Dataset):
    """Unlabeled 32x32 RGB blocks/stripes. No downloads needed."""
    def __init__(self, count=512, seed=42):
        rng = np.random.default_rng(seed)
        data = []
        for _ in range(count):
            img = np.zeros((3, 32, 32), dtype=np.float32)
            img[:] = rng.uniform(0.05, 0.25, (3, 1, 1))
            # colored square in different positions and sizes
            sz = int(rng.integers(7, 19))
            y = int(rng.integers(0, 33 - sz))
            x = int(rng.integers(0, 33 - sz))
            img[:, y:y+sz, x:x+sz] = rng.uniform(.5, 1., (3, 1, 1))
            # horizontal or vertical line pattern
            if rng.integers(2):
                img[:, ::6, :] = rng.uniform(.4, 1., (3, 1, 1))
            else:
                img[:, :, ::6] = rng.uniform(.4, 1., (3, 1, 1))
            data.append(torch.from_numpy(img.copy()))
        self.data = torch.stack(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class VQVAE(nn.Module):
    def __init__(self, num_codes=128, latent_dim=32, beta=0.25):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, latent_dim, 3, padding=1),
        )
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)  # [B, D, 8, 8]
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, d)
        e = self.codebook.weight  # [K, D]
        # squared distances [B*H*W, K], no full broadcasting [N,K,D]
        dist = flat.square().sum(-1, keepdim=True) - 2*flat @ e.t() + e.square().sum(-1)[None, :]
        ids = dist.argmin(-1)  # [B*H*W]
        z_q = self.codebook(ids).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        # reconstruction gradient bypasses argmin and goes to encoder
        z_st = z + (z_q - z).detach()
        x_hat = self.decoder(z_st)
        # codebook loss updates e; commitment loss updates encoder
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        return x_hat, ids.view(b, h, w), codebook_loss, commitment_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['synthetic', 'cifar10'], default='synthetic')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-codes', type=int, default=128)
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--outdir', default='results_vqvae')
    args = p.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == 'synthetic':
        data = SyntheticRGB(512)
    else:
        from torchvision import datasets, transforms
        data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    net = VQVAE(args.num_codes, args.latent_dim)
    first_codebook = net.codebook.weight.detach().clone()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    for epoch in range(args.epochs):
        net.train()
        sums = torch.zeros(4)
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x_hat, ids, l_code, l_commit = net(x)
            l_recon = F.mse_loss(x_hat, x)
            loss = l_recon + l_code + .25*l_commit
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += torch.tensor([loss.item(), l_recon.item(), l_code.item(), l_commit.item()])
        net.eval()
        with torch.no_grad():
            demo = data[0]
            demo = demo[0] if isinstance(demo, (tuple, list)) else demo
            recovered, ids, _, _ = net(demo.unsqueeze(0))
            save_image(torch.stack([demo, recovered[0]]), out / 'original_vs_reconstruction.png', nrow=2)
        print(f'epoch {epoch+1}: total={sums[0]/len(loader):.4f} '
              f'recon={sums[1]/len(loader):.4f} codebook={sums[2]/len(loader):.4f} '
              f'commit={sums[3]/len(loader):.4f} | ids={tuple(ids.shape)} '
              f'used_in_one_image={ids.unique().numel()}/{args.num_codes}')
    drift = (net.codebook.weight.detach() - first_codebook).abs().mean().item()
    print('Codebook mean absolute weight change:', round(drift, 6))
    print('Example image token IDs:\n', ids[0])
    torch.save({'state_dict': net.state_dict(), 'num_codes': args.num_codes,
                'latent_dim': args.latent_dim}, out / 'vqvae.pt')
    print('Saved:', out / 'vqvae.pt', out / 'original_vs_reconstruction.png')


if __name__ == '__main__':
    main()
