import sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tacos_core import _compute_base_score,scores_multiscale_cover_sum_1d,apply_compensation,apply_drop_feature
from tacos_neuron import PaperLIFNode
from types import SimpleNamespace
from spikingjelly.activation_based.neuron import LIFNode
torch.set_num_threads(2);torch.manual_seed(13)
checks=[]
x=torch.tensor([[[1.,0.],[1.,1.],[0.,0.]]]);u=torch.tensor([[[2.,4.],[2.,2.],[0.,0.]]])
s=_compute_base_score(x,v_seq_btn=u)
assert torch.allclose(s,torch.tensor([[1.6,.6+.7*2/3,0.]]));checks.append('MEAC equations and first consistency=1')
v=torch.randn(2,7);windows=[2,4,8]
out=torch.zeros_like(v)
for w in windows:
 left=(w-1)//2
 e=torch.stack([sum(v[:,max(0,min(6,t+j-left))] for j in range(w)) for t in range(7)],1)
 out+=torch.stack([sum(e[:,max(0,min(6,t+j-left))] for j in range(w)) for t in range(7)],1)/(w*w)
assert torch.allclose(out,scores_multiscale_cover_sum_1d(v,windows),atol=1e-6);checks.append('Multiscale independent nested-loop reference including even-window boundaries')
for T,K,w in [(16,8,4),(16,2,8),(9,1,8),(8,8,4)]:
 z=torch.randint(0,2,(2,T,5)).float().requires_grad_();idx=torch.rand(2,T).argsort(1)[:,:K].sort(1).values
 lengths=torch.tensor([T,T-2]);kept=z.gather(1,idx[:,:,None].expand(-1,-1,5));reference=kept.detach().clone()
 for b in range(2):
  for j,k in enumerate(idx[b].tolist()):
   reference[b,j]=0 if k>=lengths[b] else kept[b,j].detach()+sum((z[b,t].detach() for t in range(max(0,k-w,int(idx[b,j-1]) if j else 0),k) if t not in idx[b].tolist() and t<lengths[b]),torch.zeros(5))
 a=apply_compensation(z,idx,kept,lengths,window=w)
 assert torch.equal(a,reference);a.sum().backward();assert torch.isfinite(z.grad).all()
 z2=z.detach().clone();z2[:,int(idx[:,-1].max())+1:]+=100
 assert torch.equal(a.detach(),apply_compensation(z2,idx,kept.detach(),lengths,window=w))
checks.append('CQA independent set-sum, no cap, bounded past, no future, gradients and invalid padding')
z=torch.ones(1,10,1);idx=torch.tensor([[9]]);assert apply_compensation(z,idx,z[:,:1],torch.tensor([10]),window=8).item()==9
cfg=SimpleNamespace(drop_compensation=True,drop_sequence=True,drop_ratio=.5,tacos_profile='paper',drop_windows=[2,4],drop_comp_window=4)
z=torch.randint(0,2,(8,2,5)).float();u=torch.randn_like(z);length=torch.tensor([8,3]);a,b,l,idx=apply_drop_feature(cfg,z,length,u,True)
assert a.shape==(4,2,5) and l.tolist()==[4,3] and idx[1,:3].tolist()==[0,1,2]
assert torch.equal(a[3,1],torch.zeros(5)) and torch.equal(b[3,1],torch.zeros(5));checks.append('Top-k mask, temporal order, original batch-longest convention')
for decay in [True,False]:
 for reset in [None,0.,.5]:
  a=PaperLIFNode(tau=2.,v_reset=reset,decay_input=decay,step_mode='m',backend='torch',store_v_seq=True)
  b=LIFNode(tau=2.,v_reset=reset,decay_input=decay,step_mode='m',backend='torch',store_v_seq=True)
  inp=torch.randn(6,2,3)*2;output=a(inp);assert torch.equal(output,b(inp))
  prev=torch.full_like(inp[0],reset or 0.);us=[]
  for t in range(6):
   h=prev+(inp[t]-(prev-(reset or 0.)))/2 if decay else prev-(prev-(reset or 0.))/2+inp[t]
   us.append(h);prev=h-output[t] if reset is None else h*(1-output[t])+reset*output[t]
  assert torch.allclose(a.paper_u_seq,torch.stack(us),atol=1e-6)
checks.append('Prespike U reference and exact original LIF spike outputs for 6 neuron settings')
print(json.dumps({'passed':checks},indent=2))
