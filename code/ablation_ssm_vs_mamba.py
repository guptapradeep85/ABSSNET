"""ablation_ssm_vs_mamba.py  --  Reviewer 1 (Comment 2)

Drop-in replacement of ABSS-Net's gated depth-wise-conv "state-space" block with
a GENUINE selective-scan Mamba block (input-dependent A, B, C computed by a
hardware-aware parallel scan). This quantifies what a true SSM adds over the
gated convolutional approximation used in the paper.

Setup : pip install mamba-ssm causal-conv1d
Use   : set self.ssm1/self.ssm2 to MambaBlock in ABSSNet.__init__, then re-run
        run_cross_validation() and compare Acc / QWK / ECE to the gated block.
"""
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_OK = True
except Exception:
    MAMBA_OK = False


class MambaBlock(nn.Module):
    """True selective-scan SSM block, API-compatible with StateSpaceBlock."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        assert MAMBA_OK, "pip install mamba-ssm causal-conv1d"
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):        # x: [B, N, D]
        return self.drop(self.mamba(self.norm(x))) + x


# In ABSSNet.__init__, replace:
#     self.ssm1 = StateSpaceBlock(256); self.ssm2 = StateSpaceBlock(256)
# with (guarded by a USE_MAMBA flag):
#     self.ssm1 = MambaBlock(256);      self.ssm2 = MambaBlock(256)
# Keep every other hyper-parameter identical for a controlled comparison.
