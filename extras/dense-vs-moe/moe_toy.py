"""Demonstrate dense FFN vs Top-2 sparse MoE and gradient activation.

This is an educational PyTorch MoE router, not a distributed production MoE.
One expert intentionally never selected and receives no gradient.
"""
import torch
from torch import nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, dim=4, hidden=12):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.SiLU(),nn.Linear(hidden,dim))
    def forward(self,x):return self.net(x)

class MoE(nn.Module):
    def __init__(self,dim=4,num_experts=6,top_k=2):
        super().__init__()
        self.router=nn.Linear(dim,num_experts,bias=False)
        self.experts=nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.top_k=top_k

    def forward(self,x):
        logits=self.router(x)           # [tokens,experts]
        top_scores,top_ids=torch.topk(logits,self.top_k,dim=-1)
        gates=F.softmax(top_scores,dim=-1)  # normalized among selected experts
        result=torch.zeros_like(x)
        for expert_id,expert in enumerate(self.experts):
            where=(top_ids==expert_id).nonzero(as_tuple=False) # rows: [token_idx, topK_slot]
            if where.numel()==0:continue
            token_idx,slot_idx=where[:,0],where[:,1]
            prediction=expert(x[token_idx])
            contribution=gates[token_idx,slot_idx,None]*prediction
            result.index_add_(0,token_idx,contribution)
        return result,top_ids,gates

def main():
    torch.manual_seed(0)
    dense=Expert()
    moe=MoE()
    # The test cases are orthogonal token embeddings; deterministic routing matrix
    # ensures all of experts 0..4 are selected at least once, and expert 5 never is.
    weights=torch.tensor([[5,0,0,1],[4,1,0,0],[0,5,1,0],[0,4,0,0],[0,0,5,2],[-8,-8,-8,-8.]],dtype=torch.float32)
    with torch.no_grad():moe.router.weight.copy_(weights)
    x=torch.eye(4)
    y,ids,gates=moe(x)
    print('input: 4 distinct token embeddings; router output:')
    for token in range(4):
        print(' token',token,'-> experts',ids[token].tolist(),'weights',[round(v,3) for v in gates[token].tolist()])
    print('Dense FFN params:',sum(p.numel() for p in dense.parameters()))
    print('Sparse MoE params:',sum(p.numel() for p in moe.parameters()))
    # Force a gradient path through selected experts
    loss=(y-torch.tensor([[-.3,.2,.4,.1],[.1,-.5,.4,.2],[.3,.2,-.1,.1],[.2,.2,.3,-.2]])).square().mean()
    loss.backward()
    for idx,expert in enumerate(moe.experts):
        magnitude=sum(float(p.grad.abs().sum()) for p in expert.parameters() if p.grad is not None)
        print(f' Expert {idx} accumulated abs gradient: {magnitude:.6f} {"ACTIVE" if magnitude>0 else "INACTIVE"}')
    print('Reminder: inactive expert still occupies model weight memory unless offloaded.')

if __name__=='__main__': main()
