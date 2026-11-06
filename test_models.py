import argparse,sys,json,torch,importlib,contextlib,io
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('domain');p.add_argument('model');a=p.parse_args()
sys.path.insert(0,str(R/('code_DVS' if a.domain=='dvs' else 'code_speech')))
from config_loader import Config
from spikingjelly.activation_based import functional
torch.set_num_threads(2);torch.manual_seed(42)
if a.domain=='dvs':
 filename='cifar10dvs_'+a.model+'.yaml';path=R/'code_DVS/configs/cifar10dvs/16T'/filename
else:path=R/'code_speech/configs/T=100/SHD'/('shd_'+a.model+'.yaml')
c=Config(str(path),args=argparse.Namespace(tacos_mode='tacos',backend='torch'))
modname={'sdt':'spike_driven_tranformer','spikformer':'spikformer_TIM'}.get(a.model,a.model) if a.domain=='dvs' else ('SDT' if a.model=='sdt' else a.model)
module=importlib.import_module('models.'+modname)
if a.domain=='dvs':
 cls=getattr(module,{'qkformer':'QKFormer','sdt':'SpikeDrivenTransformer','spikformer':'SpikFormer','spikingformer':'SpikingFormer'}[a.model])
 kw=dict(args=c,num_classes=c.n_outputs,num_heads=c.num_heads,mlp_ratios=c.mlp_ratios,embed_dims=256)
 if a.model=='qkformer':kw.update(in_channels=2,patch_size=16)
 m=cls(**kw);x=torch.rand(2,4,2,32,32);mask=None
else:
 m=module.SpikeDrivenTransformer(c);x=torch.rand(2,8,c.n_inputs);mask=torch.arange(8)[None,:]<torch.tensor([8,6])[:,None]
for n in m.modules():
 if hasattr(n,'backend'):n.backend='torch'
seen=[];original=module.apply_drop_feature;expected_q=[];actual_q=[]
for name,n in m.named_modules():
 if name.endswith('q_conv'):
  qhandle=n.register_forward_hook(lambda mod,ins,out:actual_q.append(ins[0].detach().reshape(-1).clone()))
  break
def record(cfg,z,length,v_seq=None,return_indices=False,spike_seq=None):
 assert spike_seq is not None and bool(((spike_seq==0)|(spike_seq==1)).all())
 out=original(cfg,z,length,v_seq,return_indices,spike_seq)
 qexpected=out[1].permute(1,2,0) if a.domain=='speech' and a.model=='spikcommander' else out[1]
 expected_q.append(qexpected.detach().reshape(-1).clone())
 seen.append({'shape':list(z.shape),'aggregation_binary':bool(((z==0)|(z==1)).all()),'scoring_binary':True,'lengths':length.tolist(),'out_length':out[0].shape[0]})
 return out
module.apply_drop_feature=record
m.eval()
with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):out=m(x) if a.domain=='dvs' else m(x,mask)
assert torch.isfinite(out).all() and len(seen)==1
assert seen[0]['out_length']==x.shape[1]//2
assert torch.equal(actual_q[-1],expected_q[-1])
compression=seen[0]
functional.reset_net(m)
c.drop_sequence=False;seen.clear()
with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):full=m(x) if a.domain=='dvs' else m(x,mask)
assert torch.isfinite(full).all() and not seen
functional.reset_net(m);c.drop_sequence=True;m.train()
# Verify that training still receives gradients through gathering and CQA.
x.requires_grad_()
with contextlib.redirect_stdout(io.StringIO()):train_out=m(x) if a.domain=='dvs' else m(x,mask)
train_out.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
assert torch.equal(actual_q[-1],expected_q[-1]);qhandle.remove()
print(json.dumps({'domain':a.domain,'model':a.model,'tacos_shape':list(out.shape),'full_shape':list(full.shape),'params':sum(p.numel() for p in m.parameters()),'compression':compression,'backward_finite':True,'actual_query_equals_CQA':True,'passed':True}))
