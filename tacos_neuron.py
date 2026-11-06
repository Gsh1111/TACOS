"""Expose pre-reset membrane without changing the baseline LIF output."""
import torch
from spikingjelly.activation_based.neuron import LIFNode

class PaperLIFNode(LIFNode):
    def forward(self,x):
        previous=self.v.detach().clone() if isinstance(self.v,torch.Tensor) else self.v
        out=super().forward(x)
        if self.step_mode=='m' and self.store_v_seq:
            with torch.no_grad():
                self.paper_s_seq=out.detach()
                first=previous if isinstance(previous,torch.Tensor) else torch.full_like(x[0],float(previous))
                prev=torch.cat([first.unsqueeze(0),self.v_seq[:-1]],dim=0)
                reset=0. if self.v_reset is None else self.v_reset
                self.paper_u_seq=prev+(x-(prev-reset))/self.tau if self.decay_input else prev-(prev-reset)/self.tau+x
        return out
