"""第七课：两级多码本预测（教学演示，不是 Qwen3-TTS 权重/声码器）

运行: python toy_multicodebook.py --steps 400
依赖: pip install torch numpy matplotlib soundfile

本脚本把『输入文字序列 → 第一组语义 ID → 其余15组声学 ID →
16组码本 → 教学用波形合成』拆开。

注意：本示例的声音由固定数学谐波合成，不是真人语音。
真正的 Qwen3-TTS 使用训练好的神经音频 Codec Decoder。
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import soundfile as sf
import matplotlib.pyplot as plt

V = 32     # 教学：每组码本32个 ID（官方为2048）
Q = 16     # 官方12.5Hz版本的码本层数
NTEXT = 6  # 教学：6个抽象『文字 Token』
NSPK = 2


def make_codes(text, speaker):
    """由抽象文字+说话人决定一个可复现的教学标签 [B,T,16]。"""
    b,t = text.shape
    out = torch.zeros(b,t,Q,dtype=torch.long, device=text.device)
    out[:,:,0] = (3*text + 5*speaker.unsqueeze(1) + 7)%V
    for j in range(1,Q):
        out[:,:,j] = (out[:,:,j-1] + j + speaker.unsqueeze(1)) % V
    return out

class TinyMulticodebookTTS(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.text_embedding = nn.Embedding(NTEXT, dim)
        self.spk_embedding = nn.Embedding(NSPK, dim)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.semantic_head = nn.Linear(dim, V)
        self.prev_code_embedding = nn.Embedding(V, dim)
        self.depth_embedding = nn.Embedding(Q, dim)
        self.mtp = nn.Sequential(nn.Linear(dim*3, dim*2), nn.ReLU(), nn.Linear(dim*2,V))

    def encode_text(self, text, speaker):
        x = self.text_embedding(text) + self.spk_embedding(speaker)[:, None, :]
        h, _ = self.temporal(x)
        return h

    def loss_and_accuracy(self, text, speaker, labels):
        h = self.encode_text(text, speaker)
        b,t,d = h.shape
        logits0 = self.semantic_head(h)
        loss_sem = F.cross_entropy(logits0.reshape(-1,V), labels[:,:,0].reshape(-1))
        # MTP teacher forcing: 当前声学码本查询其上一组真实码本ID
        prev = labels[:,:,:-1]
        depth_ids = torch.arange(1,Q,device=text.device)
        mtp_input = torch.cat([
            h[:,:,None,:].expand(-1,-1,Q-1,-1),
            self.prev_code_embedding(prev),
            self.depth_embedding(depth_ids)[None,None,:,:].expand(b,t,-1,-1)
        ],dim=-1)
        logits = self.mtp(mtp_input)  # [B,T,15,32]
        loss_mtp = F.cross_entropy(logits.reshape(-1,V), labels[:,:,1:].reshape(-1))
        accuracy0 = (logits0.argmax(-1)==labels[:,:,0]).float().mean().item()
        accuracy_mtp = (logits.argmax(-1)==labels[:,:,1:]).float().mean().item()
        return loss_sem+loss_mtp, accuracy0, accuracy_mtp

    @torch.no_grad()
    def generate(self, text, speaker):
        h=self.encode_text(text,speaker)
        b,t,_=h.shape
        codes=torch.zeros(b,t,Q,dtype=torch.long,device=text.device)
        codes[:,:,0]=self.semantic_head(h).argmax(-1)
        for j in range(1,Q):
            depth=torch.full((b,t),j,device=text.device,dtype=torch.long)
            mtp_input=torch.cat([h, self.prev_code_embedding(codes[:,:,j-1]),self.depth_embedding(depth)],dim=-1)
            codes[:,:,j]=self.mtp(mtp_input).argmax(-1)
        return codes


def synthesize_tones(codes, used_codebooks=16, sr=24000, fps=12.5):
    """教学专用：把 ID 映射为不同正弦频率及谐波；绝不表示真实 Codec 解码。"""
    codes=np.asarray(codes,dtype=np.int64)
    assert codes.ndim==2 and codes.shape[1]==Q
    frame_len=int(sr/fps)  # 1920 samples = 80ms
    result=[]
    for row in codes:
        f0=155+int(row[0]%16)*13
        x=np.arange(frame_len,dtype=np.float64)/sr
        wave=np.sin(2*np.pi*f0*x)
        for j in range(1,used_codebooks):
            freq=(1+j)*f0+int(row[j]%5)*7
            if freq < sr/2:
                wave += (0.16*(0.86**(j-1)))*np.sin(2*np.pi*freq*x)
        wave=wave/(1+sum(0.16*(0.86**(j-1)) for j in range(1,used_codebooks)))
        # 每个帧左右的淡入淡出，避免相位跳变处强烈的 click
        fade=min(int(sr*.005),frame_len//4)
        envelope=np.ones(frame_len)
        envelope[:fade]=np.linspace(0,1,fade)
        envelope[-fade:]=np.linspace(1,0,fade)
        result.append(wave*envelope*.55)
    return np.concatenate(result).astype(np.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--out',default=str(Path(__file__).parent/'results'))
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(),2))
    torch.manual_seed(7);np.random.seed(7)
    model=TinyMulticodebookTTS()
    optim=torch.optim.AdamW(model.parameters(),lr=.003)
    history=[]
    model.train()
    for step in range(args.steps):
        text=torch.randint(0,NTEXT,(96,5))
        speaker=torch.randint(0,NSPK,(96,))
        labels=make_codes(text,speaker)
        loss,a0,am=model.loss_and_accuracy(text,speaker,labels)
        optim.zero_grad();loss.backward();optim.step()
        if step%40==0 or step==args.steps-1:
            print(f'step {step:4d}  loss={loss.item():.4f}  semantic_acc={a0:.1%}  mtp_acc={am:.1%}',flush=True)
        history.append((step,loss.item(),a0,am))
    model.eval()
    # 每个数字这里表示『一个抽象文字位置』，不是真实的中文字分词 ID
    txt=torch.tensor([[0,1,2,3,4]],dtype=torch.long)
    spk=torch.tensor([0],dtype=torch.long)
    with torch.no_grad():
        pred=model.generate(txt,spk)[0].numpy()
        target=make_codes(txt,spk)[0].numpy()
    print('target codes [T,Q]:\n',target)
    print('generated codes [T,Q]:\n',pred)
    print('full frame-code accuracy:',np.mean(pred==target))
    np.save(out/'generated_codes_T16.npy',pred)
    np.savetxt(out/'generated_codes_T16.csv',pred,fmt='%d',delimiter=',')
    torch.save({'state_dict':model.state_dict(),'note':'illustrative synthetic labels; not actual Qwen3-TTS weights'},out/'toy_mtp.pt')

    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    axes[0].plot([h[0] for h in history],[h[1] for h in history],linewidth=1.5)
    axes[0].set(title='Training loss',xlabel='Step')
    axes[1].plot([h[0] for h in history],[h[2] for h in history],label='codebook 0')
    axes[1].plot([h[0] for h in history],[h[3] for h in history],label='codebooks 1-15')
    axes[1].legend();axes[1].set(title='Teacher-forcing accuracy',xlabel='Step',ylim=(0,1.02))
    axes[2].imshow(pred.T,aspect='auto',cmap='viridis',vmin=0,vmax=V-1,interpolation='nearest')
    axes[2].set(title='Generated token IDs',xlabel='Audio frame (80ms)',ylabel='Codebook level',xticks=np.arange(len(pred)),yticks=[0,1,4,8,12,15])
    fig.savefig(out/'training_and_codes.png',dpi=145)
    plt.close(fig)
    for nq in [1,4,8,16]:
        wav=synthesize_tones(np.tile(pred, (5, 1)),used_codebooks=nq)
        sf.write(out/f'synthetic_{nq:02d}_codebooks.wav',wav,24000)
    print('saved:',out)

if __name__=='__main__':
    main()
