import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMambaGenerator(nn.Module):
    def __init__(self, mamba_model,cfg):# input_channels=1, feature_dim=128, freq_bins=257):
        """
        Mamba-based speech enhancement model with time-domain feature extraction.
        
        Args:
            mamba_model (nn.Module): The main Mamba model for magnitude enhancement.
            input_channels (int): Number of spectrogram input channels (default: 1).
            feature_dim (int): Dimensionality of time-domain feature embeddings.
            freq_bins (int): Number of frequency bins in STFT magnitude.
        """
        super().__init__()
        self.mamba_generator = mamba_model  # Your existing Mamba model
        self.feature_dim = cfg["model_cfg"]["hid_feature"]
        self.freq_bins = cfg["stft_cfg"]["n_fft"]//2 +1
        self.win_size = cfg["stft_cfg"]["win_size"]
        self.hop_length = cfg["stft_cfg"]["hop_size"]
        self.max_pool_kernel = cfg["model_cfg"]["max_pool_kernel"]
        self.conv_kernel_size = cfg["model_cfg"]["conv_kernel"]
        self.fusion_type = cfg["model_cfg"]["fusion_type"]
        
        # CNN to process time-domain features (STE, ZCR, Envelope)
        self.time_feature_extractor = nn.Sequential(
            nn.Conv1d(3, self.freq_bins//2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.freq_bins // 2),
            nn.ReLU(),
            nn.Conv1d(self.freq_bins//2, self.freq_bins, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.freq_bins),
            nn.ReLU(),
        )
        
        if self.fusion_type == "linear":
            self.fusion_layer = nn.Linear(2*self.freq_bins, self.freq_bins)
        elif self.fusion_type == "conv":
            self.fusion_layer = nn.Conv1d(
                in_channels=2 * self.freq_bins,  
                out_channels=self.freq_bins,  
                kernel_size=3,  
                padding=1, 
            )

    def resize_feature(self,feature, target_T):
        """Resize feature (B, T') to match target_T using interpolation."""
        B, T_orig = feature.shape
        feature = feature.unsqueeze(1)  # (B, 1, T_orig) for interpolation
        feature_resized = F.interpolate(feature, size=target_T, mode="linear", align_corners=False)
        return feature_resized.squeeze(1)  # (B, target_T)

    def compute_ste(self, signal, target_T):
        """Compute STE and resize it dynamically to match T."""
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)  # Ensure batch dimension

        unfolded = signal.unfold(-1, self.win_size, self.hop_length)
        if unfolded.shape[-1] == 0:
            return torch.zeros(signal.shape[0], target_T, device=signal.device)

        energy = unfolded.pow(2).sum(-1)  # (B, T')
        return self.resize_feature(energy, target_T)  # (B, target_T)

    def compute_zcr(self, signal, target_T):
        """Compute ZCR and resize it dynamically to match T."""
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        sign_changes = torch.diff(torch.sign(signal), dim=-1).abs()
        sign_changes = F.pad(sign_changes, (1, 0))

        zcr = sign_changes.unfold(-1, self.win_size, self.hop_length).sum(-1) / self.win_size
        if zcr.shape[-1] == 0:
            return torch.zeros(signal.shape[0], target_T, device=signal.device)

        return self.resize_feature(zcr, target_T)  # (B, target_T)

    def compute_envelope(self, signal,target_T):
        signal_abs = torch.abs(signal)
        envelope = F.max_pool1d(signal_abs.unsqueeze(1), self.max_pool_kernel, stride=1, padding=self.max_pool_kernel // 2).squeeze(1)
        return self.resize_feature(envelope,target_T)   # (B, target_T)

    def forward(self, waveform, magnitude_spectrogram):
        B, F, T = magnitude_spectrogram.shape
        #compute features in time domain
        ste = self.compute_ste(waveform, T)
        zcr = self.compute_zcr(waveform, T)
        envelope = self.compute_envelope(waveform, T)
        time_features = torch.stack([ste, zcr, envelope], dim=1)
        time_features = self.time_feature_extractor(time_features)

        #generate denoised magnitude
        mag_features = self.mamba_generator(magnitude_spectrogram)

        # Concatenate along the frequency dimension (dim=1)
        combined_features = torch.cat([mag_features, time_features], dim=1)
        # Apply fusion layer to combine magnitude and time features to get (B, F, T) enhanced mag
        if self.fusion_type == "linear":
            enhanced_mag = self.fusion_layer(combined_features.transpose(2, 1)).transpose(2,1) 
        elif self.fusion_type == "conv":
            enhanced_mag = self.fusion_layer(combined_features)

        return enhanced_mag

    
