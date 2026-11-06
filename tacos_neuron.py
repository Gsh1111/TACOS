import torch
from spikingjelly.activation_based.neuron import LIFNode


class PaperLIFNode(LIFNode):
    def forward(self, x):
        initial_voltage = self.v
        if isinstance(initial_voltage, torch.Tensor):
            initial_voltage = initial_voltage.detach().clone()

        spikes = super().forward(x)
        if self.step_mode == "m" and self.store_v_seq:
            with torch.no_grad():
                self.paper_s_seq = spikes.detach()
                if not isinstance(initial_voltage, torch.Tensor):
                    initial_voltage = torch.full_like(x[0], float(initial_voltage))
                previous = torch.cat([initial_voltage.unsqueeze(0), self.v_seq[:-1]], dim=0)
                if self.v_reset is None:
                    reset_voltage = 0.0
                else:
                    reset_voltage = self.v_reset
                voltage_above_reset = previous - reset_voltage
                if self.decay_input:
                    self.paper_u_seq = previous + (x - voltage_above_reset) / self.tau
                else:
                    self.paper_u_seq = previous - voltage_above_reset / self.tau + x
        return spikes
