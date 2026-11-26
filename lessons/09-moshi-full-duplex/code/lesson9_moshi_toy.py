"""Lesson 9: Moshi-inspired full-duplex multi-stream teaching experiment.

NOT a reproduction of Moshi's trained weights or sound quality.
Dependencies: torch numpy matplotlib soundfile

Three independent experiments:
  1) Tiny causal Temporal Transformer learns a synthetic speech/listen/interrupt policy.
  2) Tiny Depth Transformer learns predictable toy 8-codebook ID patterns.
  3) Streaming inference uses per-layer KV cache, logs frame-by-frame decisions,
     and writes two-channel synthesized tones to show overlapping input/output.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

FRAME_MS = 80
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000) # 1920
CODEBOOKS = 8
CODE_VOCAB = 32
ACTIONS = ['LISTEN', 'SPEAK', 'STOP']


def scripted_policy(u):
    """Synthetic oracle policy, used ONLY to make training labels, not test predictions."""
    y = []
    seen_speech = False
    responding = False
    pause_frames = 0
    for user_now in u:
        if user_now:
            if responding:
                action = 2 # interrupt ongoing answer
                responding = False
                seen_speech = True
            else:
                action = 0
                seen_speech = True
            pause_frames = 0
        else:
            if responding:
                action = 1
            elif seen_speech:
                pause_frames += 1
                if pause_frames >= 3:
                    responding = True
                    seen_speech = False
                    action = 1
                else:
                    action = 0
            else:
                action = 0
        y.append(action)
    return np.asarray(y, dtype=np.int64)


def sample_batch(batch_size=32, length=48, rng=None):
    rng = rng or np.random.default_rng(1)
    users = np.zeros((batch_size, length), dtype=np.int64)
    for b in range(batch_size):
        first = int(rng.integers(1, 5))
        n1 = int(rng.integers(8, 16))
        users[b, first:min(first+n1,length)] = 1
        if rng.random() < .85:
            # Interrupt while the system may be answering.
            second = min(length-7, first+n1+int(rng.integers(4, 14)))
            users[b, second:min(length,second+int(rng.integers(3,9)))] = 1
    actions = np.stack([scripted_policy(row) for row in users])
    return torch.from_numpy(users), torch.from_numpy(actions)


class CausalBlock(nn.Module):
    def __init__(self, dim=48, heads=4):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim//heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim,3*dim)
        self.proj = nn.Linear(dim,dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))

    def forward(self,x,cache=None):
        B,T,D = x.shape
        qkv=self.qkv(self.norm1(x)).reshape(B,T,3,self.heads,self.head_dim)
        q,k,v=[qkv[:,:,j].transpose(1,2) for j in range(3)]
        if cache is None:
            z=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            new_cache=(k,v)
        else:
            old_k,old_v=cache
            new_k=torch.cat((old_k,k),dim=2)
            new_v=torch.cat((old_v,v),dim=2)
            # For a single NEW query, all keys are past/current, hence no extra mask.
            z=F.scaled_dot_product_attention(q,new_k,new_v,is_causal=False)
            new_cache=(new_k,new_v)
        z=z.transpose(1,2).contiguous().reshape(B,T,D)
        x=x+self.proj(z)
        x=x+self.ffn(self.norm2(x))
        return x,new_cache


class TinyTemporal(nn.Module):
    def __init__(self,dim=48,layers=2,max_length=128):
        super().__init__()
        self.user_embed=nn.Embedding(2,dim)
        self.prev_action_embed=nn.Embedding(4,dim)  # 0,1,2 + 3: start
        self.position=nn.Embedding(max_length,dim)
        self.blocks=nn.ModuleList([CausalBlock(dim) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim,3)

    def forward(self,user,prev_actions):
        B,T=user.shape
        pos=torch.arange(T,device=user.device)[None,:]
        h=self.user_embed(user)+self.prev_action_embed(prev_actions)+self.position(pos)
        for block in self.blocks:
            h,_=block(h)
        return self.head(self.norm(h))

    def step(self,user_frame,prev_action,cache=None):
        # user_frame and prev_action have shape [B,1].
        if cache is None:
            cache=[None]*len(self.blocks)
        past=0 if cache[0] is None else cache[0][0].shape[2]
        pos=torch.tensor([[past]],device=user_frame.device)
        h=self.user_embed(user_frame)+self.prev_action_embed(prev_action)+self.position(pos)
        new=[]
        for block,layer_cache in zip(self.blocks,cache):
            h,c=block(h,layer_cache)
            new.append(c)
        return self.head(self.norm(h)),new


class TinyDepth(nn.Module):
    """Within-frame causal Transformer that predicts 8 toy speech codebooks."""
    def __init__(self,dim=48):
        super().__init__()
        self.state_embed=nn.Embedding(3,dim)
        self.code_embed=nn.Embedding(CODE_VOCAB,dim)
        self.pos_embed=nn.Embedding(CODEBOOKS,dim)
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=4,dim_feedforward=96,
                                           dropout=0,batch_first=True,activation='gelu')
        self.block=nn.TransformerEncoder(layer,num_layers=1)
        self.out=nn.Linear(dim,CODE_VOCAB)

    def forward(self,actions,prev_codes):
        # actions [B], prev_codes [B,8]: prev_codes[:,k] is code[k-1]
        B,K=prev_codes.shape
        x=self.state_embed(actions).unsqueeze(1)+self.code_embed(prev_codes)+self.pos_embed(torch.arange(K,device=actions.device))[None]
        mask=torch.triu(torch.ones((K,K),device=x.device,dtype=torch.bool),diagonal=1)
        return self.out(self.block(x,mask=mask))

    @torch.no_grad()
    def generate(self,action):
        codes=[]
        for k in range(CODEBOOKS):
            prefix=torch.zeros((1,CODEBOOKS),dtype=torch.long)
            if codes: prefix[0,1:len(codes)+1]=torch.tensor(codes)
            prediction=self.forward(torch.tensor([action]),prefix)
            codes.append(int(prediction[0,k].argmax()))
        return codes


def depth_targets(actions):
    actions=actions.long()
    codes=torch.zeros((actions.shape[0],CODEBOOKS),dtype=torch.long)
    base=torch.where(actions==1,torch.tensor(5),torch.where(actions==2,torch.tensor(10),torch.tensor(0)))
    for k in range(CODEBOOKS):
        codes[:,k]=torch.where(actions==0,0,(base+3*k)%CODE_VOCAB)
    prev=torch.zeros_like(codes)
    prev[:,1:]=codes[:,:-1]
    return prev,codes


def synthesize_demo(user,pred_actions,output):
    # Purely synthetic NON-SPEECH tones to demonstrate simultaneous streams.
    N=len(user)*SAMPLES_PER_FRAME
    tracks=np.zeros((N,2),dtype=np.float32)
    for t in range(len(user)):
        a=t*SAMPLES_PER_FRAME; b=a+SAMPLES_PER_FRAME
        tm=np.arange(SAMPLES_PER_FRAME)/SAMPLE_RATE
        envelope=np.sin(np.pi*(np.arange(SAMPLES_PER_FRAME)+.5)/SAMPLES_PER_FRAME)**.18
        if user[t]:
            tracks[a:b,0]=.26*envelope*(.75*np.sin(2*np.pi*260*tm)+.25*np.sin(2*np.pi*520*tm))
        # One-frame playback buffer means when interruption begins the previous
        # speech packet may already be in-flight (example, NOT empirical latency).
        play= pred_actions[t-1]==1 if t>0 else False
        if play:
            tracks[a:b,1]=.18*envelope*(.75*np.sin(2*np.pi*430*tm)+.25*np.sin(2*np.pi*645*tm))
    sf.write(output,tracks,SAMPLE_RATE,subtype='PCM_16')
    return tracks


def plot_results(user,pred,oracle,tracks,out):
    N=len(user)
    xs=np.arange(N)*.08
    fig,ax=plt.subplots(4,1,figsize=(12,7.5),sharex=True)
    ax[0].step(xs,user,where='post',label='user speaking',linewidth=2)
    ax[0].set_ylim(-.2,1.3);ax[0].set_ylabel('User PCM\n activity');ax[0].legend(loc='upper right')
    ax[1].step(xs,oracle,where='post',label='oracle',alpha=.7,linewidth=2)
    ax[1].step(xs,pred,where='post',label='model',alpha=.85,linestyle='--')
    ax[1].set_yticks([0,1,2],ACTIONS);ax[1].set_ylabel('Decision');ax[1].legend(loc='upper right')
    ax[2].plot(np.arange(len(tracks))/SAMPLE_RATE,tracks[:,0],linewidth=.3,label='left = user')
    ax[2].plot(np.arange(len(tracks))/SAMPLE_RATE,tracks[:,1],linewidth=.3,alpha=.65,label='right = assistant')
    ax[2].set_ylabel('Stereo PCM');ax[2].legend(loc='upper right')
    ax[3].step(xs,np.arange(1,N+1),where='post',label='KV length per layer')
    ax[3].set_ylabel('Cached\n frames');ax[3].set_xlabel('Time (seconds)');ax[3].legend(loc='upper left')
    for a in ax: a.grid(alpha=.2)
    fig.suptitle('Moshi-inspired full-duplex toy: audio frame = 80 ms | NOT Moshi weights',fontsize=13)
    fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=350)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='results')
    args=p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    rng=np.random.default_rng(args.seed)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)

    temporal=TinyTemporal()
    optimizer=torch.optim.AdamW(temporal.parameters(),lr=.003)
    history=[]
    for step in range(args.steps):
        u,y=sample_batch(32,48,rng)
        prev=torch.cat((torch.full((u.shape[0],1),3,dtype=torch.long),y[:,:-1]),dim=1)
        logits=temporal(u,prev)
        loss=F.cross_entropy(logits.reshape(-1,3),y.reshape(-1))
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step in (0,49,99,199,args.steps-1):
            with torch.no_grad(): accuracy=(logits.argmax(-1)==y).float().mean().item()
            history.append((step+1,float(loss),accuracy))
            print(f'temporal step {step+1:3d}: loss={loss.item():.4f}, teacher-forced train acc={accuracy:.1%}')
    temporal.eval()

    depth=TinyDepth()
    depth_optim=torch.optim.AdamW(depth.parameters(),lr=.003)
    for i in range(180):
        action=torch.randint(0,3,(128,))
        prev,codes=depth_targets(action)
        logits=depth(action,prev)
        loss=F.cross_entropy(logits.reshape(-1,CODE_VOCAB),codes.reshape(-1))
        depth_optim.zero_grad();loss.backward();depth_optim.step()
    depth.eval()

    demo=np.zeros(55,dtype=np.int64)
    demo[2:17]=1 # user speaking 1.2s
    demo[30:35]=1 # user interrupts while assistant is responding
    oracle=scripted_policy(demo)
    pred=[];caches=None;prev_action=3
    step_log=[]
    with torch.no_grad():
        for t,u_t in enumerate(demo):
            logits,caches=temporal.step(torch.tensor([[int(u_t)]]),torch.tensor([[prev_action]]),caches)
            action=int(logits[0,-1].argmax())
            pred.append(action)
            if action==1:
                ids=depth.generate(action)
            else:
                ids=[]
            sizes=[c[0].shape[2] for c in caches]
            step_log.append((t,round(t*.08,2),int(u_t),ACTIONS[action],','.join(map(str,ids)),','.join(map(str,sizes))))
            prev_action=action
    pred=np.asarray(pred,dtype=np.int64)
    accuracy=np.mean(pred==oracle)
    print('Streaming DEMO policy match vs synthetic oracle:',f'{accuracy:.1%}')
    print('Temporal KV cache sizes after last 80ms frame:',[c[0].shape for c in caches])
    print('Example Depth tokens for an SPEAK frame:',depth.generate(1))
    # Strict check: prefill and incremental state agree when both use identical input streams.
    prev_oracle=np.concatenate(([3],oracle[:-1]))
    with torch.no_grad():
        full_logits=temporal(torch.tensor(demo).unsqueeze(0),torch.tensor(prev_oracle).unsqueeze(0))
        st_cache=None;st_logits=[]
        for i in range(len(demo)):
            l,st_cache=temporal.step(torch.tensor([[int(demo[i])]]),torch.tensor([[int(prev_oracle[i])]]),st_cache)
            st_logits.append(l)
        incremental=torch.cat(st_logits,dim=1)
        err=(full_logits-incremental).abs().max().item()
    print('Max logits difference: batched causal vs incremental KV-cache:',f'{err:.8f}')

    with (root/'per_frame_log.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f)
        writer.writerow(['frame','time_s','user_active','model_action','depth_codes_if_speak','kv_len_each_layer'])
        writer.writerows(step_log)
    with (root/'train_history.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['step','loss','teacher_forced_acc']);writer.writerows(history)
    tracks=synthesize_demo(demo,pred,root/'full_duplex_stereo.wav')
    plot_results(demo,pred,oracle,tracks,root/'full_duplex_timeline.png')
    torch.save({'temporal':temporal.state_dict(),'depth':depth.state_dict()},root/'toy_weights.pt')
    for i,row in enumerate(step_log):
        if i%5==0 or i in (17,18,19,20,29,30,31,34,35,36,37):
            print(row)
    print('Files written to:',root.resolve())


if __name__=='__main__': main()
