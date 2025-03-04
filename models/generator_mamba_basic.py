import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
from .mamba_block_basic import MambaBasic

class BasicMambaGenerator(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.
    
    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.
        
        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(BasicMambaGenerator, self).__init__()
        self.cfg = cfg
        self.num_basic_blocks = cfg['model_cfg']['num_basic_mamba_f'] if cfg['model_cfg']['num_basic_mamba_f'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.Mamba_basic_blocks = nn.ModuleList([MambaBasic(cfg) for _ in range(self.num_basic_blocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)

    def forward(self, noisy_mag):
        """
        Forward pass for the SEMamba model.
        
        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].
        
        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Encode input
        x = self.dense_encoder(noisy_mag)
        # Apply Mamba blocks

        for block in self.Mamba_basic_blocks:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        return denoised_mag #, denoised_pha, denoised_com
