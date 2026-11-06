"""MEAC/CQA equations in tacosmain.tex; time-major model interface."""
import torch
import torch.nn.functional as F

def _to_btn(x,x_len):
    if x.ndim!=3 or x.shape[1]!=len(x_len):raise ValueError('Expected T,B,N')
    return x.transpose(0,1).contiguous(),True

def _parse_windows(ws):
    return [int(w) for w in (ws.replace(',',' ').split() if isinstance(ws,str) else ws)]

def _dice_sim_with_prev(active, eps=1e-8):
    c=torch.ones(active.shape[:2],device=active.device,dtype=torch.float32)
    a,b=active[:,1:],active[:,:-1]
    c[:,1:]=2*(a & b).sum(-1).float()/(a.sum(-1).float()+b.sum(-1).float()+eps)
    return c

def _compute_base_score(x_btn, score_name='v_x_spike', thr=0., v_seq_btn=None):
    if score_name!='v_x_spike' or v_seq_btn is None:
        raise ValueError('Paper MEAC requires v_x_spike and prespike membrane sequence')
    return .3*v_seq_btn.float().abs().mean(-1)+.7*_dice_sim_with_prev(x_btn>0)

def scores_multiscale_cover_sum_1d(sig_bt,ws=(2,4,8),pad_mode='replicate',normalize='w2'):
    if pad_mode!='replicate' or normalize!='w2':raise ValueError('Paper uses replicate padding and w^2 normalization')
    out=torch.zeros_like(sig_bt,dtype=torch.float32);x=sig_bt.float().unsqueeze(1)
    for w in ws:
        w=int(w)
        if w<1:raise ValueError('Window must be positive')
        pad=((w-1)//2,w-1-(w-1)//2);kernel=x.new_ones(1,1,w)
        e=F.conv1d(F.pad(x,pad,mode=pad_mode),kernel)
        out+=F.conv1d(F.pad(e,pad,mode=pad_mode),kernel).squeeze(1)/(w*w)
    return out

def apply_compensation(x_orig,kept_idx,x_drop,x_len,cap=None,window=8):
    if cap is not None:raise ValueError('Paper CQA has a bounded window, not an amplitude cap')
    B,T,N=x_orig.shape;K=kept_idx.shape[1]
    valid=torch.arange(T,device=x_orig.device)[None,:]<x_len[:,None]
    kept_mask=torch.zeros(B,T,device=x_orig.device,dtype=torch.bool).scatter_(1,kept_idx,True)
    dropped=x_orig.masked_fill((~valid|kept_mask).unsqueeze(-1),0)
    prefix=F.pad(dropped.float().cumsum(1),(0,0,1,0))
    left=(kept_idx-int(window)).clamp(min=0)
    if K>1:left[:,1:]=torch.maximum(left[:,1:],kept_idx[:,:-1])
    delta=prefix.gather(1,kept_idx[:,:,None].expand(-1,-1,N))-prefix.gather(1,left[:,:,None].expand(-1,-1,N))
    good=kept_idx<x_len[:,None]
    return (x_drop+delta.to(x_drop.dtype)).masked_fill(~good[:,:,None],0)

def apply_drop_feature(config,x,x_len,v_seq=None,return_indices=False,spike_seq=None):
    # All in-tree model callers explicitly pass T,B,N. This resolves B==T.
    use_comp=bool(config.drop_compensation)
    def ret(a,b,l,idx):
        ans=(a,b,l) if use_comp else (a,l)
        return ans+(idx,) if return_indices else ans
    if not config.drop_sequence or config.drop_ratio<=0:return ret(x,x,x_len,None)
    if x.ndim!=3 or x.shape[1]!=len(x_len):raise ValueError('Expected T,B,N')
    if v_seq is None or v_seq.shape!=x.shape:raise ValueError('Membrane/spike shapes must match')
    z=x.transpose(0,1);u=v_seq.transpose(0,1);B,T,N=z.shape
    if not (0<=config.drop_ratio<1):raise ValueError('Invalid pruning ratio')
    if bool(((x_len<1)|(x_len>T)).any()):raise ValueError('Invalid input lengths')
    # Preserve batch-longest input convention. Invalid locations cannot win top-k.
    K=max(1,round(T*(1-config.drop_ratio)))
    # Scoring spikes and aggregated residual features need not be the same tensor.
    # In residual backbones H0 is real-valued; S must come from the paired LIF.
    scoring=x if spike_seq is None else spike_seq
    if scoring.shape!=x.shape:raise ValueError('Scoring spike shape mismatch')
    if not bool(((scoring==0)|(scoring==1)).all()):raise ValueError('MEAC consistency requires binary spikes')
    raw=_compute_base_score(scoring.transpose(0,1),v_seq_btn=u)
    # Refine only valid prefixes, so batch padding cannot affect valid scores.
    scores=raw.new_full((B,T),-float('inf'))
    for b in range(B):
        n=int(x_len[b]);scores[b,:n]=scores_multiscale_cover_sum_1d(raw[b:b+1,:n],config.drop_windows)[0]
    idx=scores.topk(K,dim=1,largest=True).indices.sort(dim=1).values
    valid=idx<x_len[:,None];kept=z.gather(1,idx[:,:,None].expand(-1,-1,N)).masked_fill(~valid[:,:,None],0)
    lengths=valid.sum(1).clamp(min=1)
    comp=apply_compensation(z,idx,kept,x_len,window=config.drop_comp_window) if use_comp else kept
    return ret(kept.transpose(0,1).contiguous(),comp.transpose(0,1).contiguous(),lengths,idx)
