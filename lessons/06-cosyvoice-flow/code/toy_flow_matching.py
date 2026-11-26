"""Lesson 6: a tiny conditional Flow Matching demo on synthetic spectrograms.

This is NOT CosyVoice and does not synthesize speech. It trains a real PyTorch
velocity model on simple synthetic time-frequency patterns and integrates the
learned velocity with Euler's method to generate a spectrogram from noise.

Usage:
  pip install torch numpy matplotlib
  python toy_flow_matching.py --steps 450
  python toy_flow_matching.py --steps 120 --device cpu  # quick smoke test
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FREQ, TIME = 24, 36
NAMES = ["rising", "falling", "two_flat_tones"]

def make_batch(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Target 'Mel-like' spectrograms, shape [B,1,24,36], value approx [-0.75,1]."""
    batch = labels.numel()
    f = torch.arange(FREQ, device=device, dtype=torch.float32).view(1, FREQ, 1)
    t = torch.arange(TIME, device=device, dtype=torch.float32).view(1, 1, TIME)
    lbl = labels.view(-1,1,1)
    # Simple spectral trajectories, and slight shift variability.
    shift = torch.empty(batch,1,1,device=device).uniform_(-.7,.7)
    rising = 5.0 + 14.0 * t / (TIME-1) + shift
    falling = 19.0 - 14.0 * t / (TIME-1) + shift
    two = 0.95 * torch.exp(-0.5*((f-(6.0+shift))/.9)**2) + 0.82 * torch.exp(-0.5*((f-(17.0+shift))/.9)**2)
    single = torch.exp(-0.5*((f-torch.where(lbl==0, rising, falling))/.95)**2)
    signal = torch.where(lbl==2, two.expand(-1,-1,TIME), single)
    # Background -0.65, ridge +0.85
    spect = -.65 + 1.55 * signal + .025*torch.randn(batch,FREQ,TIME,device=device)
    return spect.unsqueeze(1)

class VelocityNet(nn.Module):
    """Tiny conditional CNN: noise/mixed-Mel + time + 3-class condition -> velocity."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 32, 5, padding=2), nn.SiLU(),
            nn.Conv2d(32, 48, 5, padding=2), nn.SiLU(),
            nn.Conv2d(48, 48, 5, padding=2), nn.SiLU(),
            nn.Conv2d(48, 1, 3, padding=1),
        )

    def forward(self, x, t, labels):
        b, _, h, w = x.shape
        tmap = t.view(b,1,1,1).expand(b,1,h,w)
        class_map = torch.nn.functional.one_hot(labels, num_classes=3).to(x.dtype).view(b,3,1,1).expand(b,3,h,w)
        freq_map = torch.linspace(-1,1,h,device=x.device).view(1,1,h,1).expand(b,1,h,w)
        time_map = torch.linspace(-1,1,w,device=x.device).view(1,1,1,w).expand(b,1,h,w)
        return self.net(torch.cat([x,tmap,class_map,freq_map,time_map],dim=1))

@torch.no_grad()
def sample_flow(net, label, device, seed=123, n_steps=20):
    torch.manual_seed(seed)
    x = torch.randn(1,1,FREQ,TIME,device=device)
    frames = {0:x.squeeze().cpu().numpy().copy()}
    for k in range(n_steps):
        t = torch.tensor([k/n_steps],device=device,dtype=torch.float32)
        cls = torch.tensor([label],device=device,dtype=torch.long)
        v = net(x,t,cls)
        x = x + v/n_steps
        if k+1 in [round(.25*n_steps),round(.5*n_steps),round(.75*n_steps),n_steps]:
            frames[k+1] = x.squeeze().cpu().numpy().copy()
    return frames

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=450)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default=str(Path(__file__).parent/'results'))
    args=p.parse_args()
    torch.manual_seed(23)
    torch.set_num_threads(min(3,torch.get_num_threads()))
    device=torch.device(args.device)
    net=VelocityNet().to(device)
    opt=torch.optim.Adam(net.parameters(),lr=1.5e-3)
    hist=[]
    for step in range(args.steps):
        labels=torch.randint(0,3,(24,),device=device)
        x1=make_batch(labels,device)
        x0=torch.randn_like(x1)
        t=torch.rand(24,device=device)
        mix=(1-t[:,None,None,None])*x0+t[:,None,None,None]*x1
        target_v=x1-x0
        pred=net(mix,t,labels)
        loss=(pred-target_v).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        hist.append(loss.item())
        if (step+1)%max(1,args.steps//5)==0 or step==0:
            print(f'step={step+1:4d} flow_mse={loss.item():.5f}',flush=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'model':net.state_dict(),'steps':args.steps},out/'toy_velocity_net.pt')
    net.eval()
    plot_steps=[0,5,10,15,20]
    fig,axes=plt.subplots(3,6,figsize=(16,8),sharex=True,sharey=True)
    for j in range(3):
        frames=sample_flow(net,j,device,n_steps=20)
        for k,s in enumerate(plot_steps):
            axes[j,k].imshow(frames[s],origin='lower',aspect='auto',vmin=-1.3,vmax=1.3,cmap='magma')
            if j==0:axes[j,k].set_title(f'Flow t={s/20:.2f}')
            if k==0:axes[j,k].set_ylabel(NAMES[j]+'\nfrequency bins')
            if j==2:axes[j,k].set_xlabel('time frames')
        truth=make_batch(torch.tensor([j],device=device),device).squeeze().cpu().numpy()
        axes[j,5].imshow(truth,origin='lower',aspect='auto',vmin=-1.3,vmax=1.3,cmap='magma')
        if j==0:axes[j,5].set_title('Example target')
        if j==2:axes[j,5].set_xlabel('time frames')
    fig.suptitle('Toy Conditional Flow Matching: noise -> Mel-like features (not speech)',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(out/'flow_steps.png',dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot(hist)
    ax.set_xlabel('Training step');ax.set_ylabel('Velocity MSE');ax.set_title('Flow Matching training loss')
    fig.tight_layout();fig.savefig(out/'loss.png',dpi=150);plt.close(fig)
    print('Saved:',out/'flow_steps.png',out/'loss.png',out/'toy_velocity_net.pt',flush=True)

if __name__=='__main__':main()
