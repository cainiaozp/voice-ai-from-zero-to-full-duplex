"""Audio lesson 2: reproducible three-class synthetic sound classification.
Run: python train_audio_cnn.py --epochs 6
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.io.wavfile import write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 80
NAMES = ['low_to_high', 'high_to_low', 'simultaneous']

def mel_bank():
    """Fixed triangular filters on the HTK mel scale (not learned)."""
    hz = torch.linspace(0, SR / 2, N_FFT // 2 + 1)
    max_mel = 2595 * np.log10(1 + SR / 2 / 700)
    edges = 700 * (10 ** (torch.linspace(0, float(max_mel), N_MELS + 2) / 2595) - 1)
    left, middle, right = edges[:-2,None], edges[1:-1,None], edges[2:,None]
    return torch.minimum((hz-left)/(middle-left), (right-hz)/(right-middle)).clamp(min=0)

MEL = mel_bank()
WINDOW = torch.hann_window(N_FFT)

def log_mel(wave):
    # wave [N] or [B,N] -> [...,1,80,98] for 1 second
    spec = torch.stft(wave, N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=WINDOW.to(wave.device), center=False, return_complex=True)
    power = spec.abs().square()
    mel = MEL.to(wave.device) @ power
    db = 10 * torch.log10(mel.clamp_min(1e-8))
    db = db - db.amax(dim=(-2,-1), keepdim=True)
    return ((db.clamp(-80, 0) + 80) / 80).unsqueeze(-3)


def make_wave(label, rng):
    """Different frequencies, phases, gains, transition times and noise per sample."""
    t = np.arange(SR) / SR
    low, high = rng.uniform(200, 320), rng.uniform(650, 1000)
    a = np.sin(2*np.pi*low*t + rng.uniform(0, 2*np.pi))
    b = np.sin(2*np.pi*high*t + rng.uniform(0, 2*np.pi))
    split = rng.uniform(.4, .6)
    # 20ms crossfade makes the switch smooth.
    w = np.clip((t - split + .01) / .02, 0, 1)
    if label == 0:
        wave = (1-w)*a + w*b
    elif label == 1:
        wave = (1-w)*b + w*a
    else:
        wave = .5*a + .5*b
    wave = rng.uniform(.25, .65)*wave
    fade = np.minimum(np.minimum(t/.02, (1-t)/.02), 1)
    wave = wave*fade + rng.normal(0, rng.uniform(.005,.06), SR)
    return torch.tensor(np.clip(wave, -1, 1), dtype=torch.float32)


def make_dataset(per_class, seed):
    rng = np.random.default_rng(seed)
    waves = torch.stack([make_wave(k,rng) for k in range(3) for _ in range(per_class)])
    labels = torch.arange(3).repeat_interleave(per_class)
    idx = torch.from_numpy(rng.permutation(len(labels)))
    waves, labels = waves[idx], labels[idx]
    # Batch extraction to keep memory modest.
    features = torch.cat([log_mel(x) for x in waves.split(64)])
    return TensorDataset(features, labels), waves


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.classifier = nn.Linear(32*4*4, 3)

    def forward(self, x):
        return self.classifier(self.encoder(x))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    cm = torch.zeros(3,3,dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += nn.functional.cross_entropy(logits,y,reduction='sum').item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += len(y)
        cm += torch.bincount((3*y+pred).cpu(),minlength=9).reshape(3,3)
    return total_loss/n, correct/n, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int, default=6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default=str(Path(__file__).parent/'results'))
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    device = torch.device(args.device)
    train, _ = make_dataset(120, 100)
    val, _ = make_dataset(30, 200)
    test, test_waves = make_dataset(30, 300)
    loaders = [DataLoader(ds,batch_size=32,shuffle=i==0, generator=torch.Generator().manual_seed(123)) for i,ds in enumerate([train,val,test])]
    model = AudioCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print('Train/val/test:',len(train),len(val),len(test))
    print('One input:',train[0][0].shape,'Batch:',next(iter(loaders[0]))[0].shape)
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; correct=0
        for x,y in loaders[0]:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            logits=model(x)
            loss=loss_fn(logits,y)
            loss.backward(); optimizer.step()
            total+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item()
        vl,va,_=evaluate(model,loaders[1],device)
        row=(total/len(train),correct/len(train),vl,va); history.append(row)
        print(f'Epoch {epoch:02d}: train_loss={row[0]:.4f} train_acc={row[1]:.2%} val_loss={vl:.4f} val_acc={va:.2%}')
    tl,ta,cm=evaluate(model,loaders[2],device)
    print(f'Test: loss={tl:.4f} acc={ta:.2%}\nConfusion matrix:\n{cm}')
    torch.save(model.state_dict(),out/'audio_cnn.pt')
    np.savetxt(out/'history.csv',np.asarray(history),delimiter=',',header='train_loss,train_acc,val_loss,val_acc',comments='')
    fig,ax=plt.subplots(2,1,figsize=(8,6),layout='constrained')
    h=np.array(history); e=np.arange(1,len(h)+1)
    ax[0].plot(e,h[:,0],marker='o',label='train'); ax[0].plot(e,h[:,2],marker='o',label='validation'); ax[0].set_ylabel('Cross entropy'); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].plot(e,h[:,1]*100,marker='o',label='train'); ax[1].plot(e,h[:,3]*100,marker='o',label='validation'); ax[1].set_ylabel('Accuracy (%)'); ax[1].set_xlabel('Epoch'); ax[1].set_ylim(0,105); ax[1].legend(); ax[1].grid(alpha=.2)
    fig.savefig(out/'training_curves.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(3,1,figsize=(8,6.5),layout='constrained')
    for k in range(3):
        idx=int(torch.where(test.tensors[1]==k)[0][0]); wave=test_waves[idx]
        feat=test.tensors[0][idx]
        with torch.no_grad():
            pred=int(model(feat[None].to(device)).argmax())
        write(out/f'{NAMES[k]}.wav',SR,np.rint(wave.numpy()*32767).astype(np.int16))
        ax[k].imshow(feat[0].numpy(),origin='lower',aspect='auto',extent=[0,.98,0,80],vmin=0,vmax=1,cmap='magma')
        ax[k].set_title(f'True: {NAMES[k]} | Pred: {NAMES[pred]}'); ax[k].set_ylabel('Mel bin')
    ax[-1].set_xlabel('Time (s, frame starts)'); fig.savefig(out/'samples.png',dpi=150); plt.close(fig)
    x=test.tensors[0][:1].to(device)
    print('Encoder embedding:',model.encoder(x).shape,'Logits:',model(x).shape)
    print('Saved to:',out)

if __name__=='__main__':
    main()
