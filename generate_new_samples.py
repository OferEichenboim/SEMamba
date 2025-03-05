
#generate_new_sample
#save_to_directory
#run
import torch
import torchaudio
import librosa
import random
from models.generator_mamba_basic import BasicMambaGenerator
from models.generator_mamba_fusion import FusionMambaGenerator
from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

from dataloaders.dataloader_vctk import VCTKDemandDataset
from torch.utils.data import DistributedSampler, DataLoader
from models.stfts import mag_phase_stft, mag_phase_istft


rank=0
cfg = load_config("./recipes/SEMamba_fusion/SEMamba_fusion.yaml")
n_fft=cfg['stft_cfg']['n_fft']
hop_size=cfg['stft_cfg']['hop_size']
win_size=cfg['stft_cfg']['win_size']
compress_factor=cfg['model_cfg']['compress_factor']
#initialize_process_group(cfg, rank)

def get_noisy_test_sample(noisy_path,cfg):
    n_fft=cfg['stft_cfg']['n_fft']
    hop_size=cfg['stft_cfg']['hop_size']
    win_size=cfg['stft_cfg']['win_size']
    compress_factor=cfg['model_cfg']['compress_factor']
    
    noisy_audio, _ = librosa.load( noisy_path, sr=16000)
    segment_size = noisy_audio.shape[0]
    
    noisy_audio = torch.FloatTensor(noisy_audio)
    norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
    noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

    if noisy_audio.size(1) >= segment_size:
        max_audio_start = noisy_audio.size(1) - segment_size
        audio_start = random.randint(0, max_audio_start)
        noisy_audio = noisy_audio[:, audio_start:audio_start + segment_size]
    else:
        noisy_audio = torch.nn.functional.pad(noisy_audio, (0, segment_size - noisy_audio.size(1)), 'constant')

    noisy_mag, noisy_pha, _ = mag_phase_stft(noisy_audio, n_fft, hop_size, win_size, compress_factor)

    noisy_audio = noisy_audio.squeeze()
    return (noisy_audio, noisy_mag, noisy_pha)

def get_clean_phase(clean_path,cfg):
    _,_,clean_phase = get_noisy_test_sample(clean_path,cfg)
    return clean_phase

def compute_rms(signal):
    return torch.sqrt(torch.mean(signal ** 2) + 10**(-8))  # Add small value to avoid NaN

def normalize_energy(enhanced):
    rms_enhanced = compute_rms(enhanced)
    
    # Avoid division by zero
    scale_factor = 1 / (rms_enhanced +  10**(-8))  
    return enhanced * scale_factor

#inputs
output_dir = "./EnhancedSamples/SEMamba_fusion/"
model_path = "./exp/./SEMamba_fusion//g_00020000.pth"
clean_path_dir = "./EnhancedSamples/ref_clean/"
noisy_dir = "./EnhancedSamples/ref_noisy/"

noisy_files = ["p232_005.wav","p232_013.wav","p232_095.wav","p232_121.wav","p232_151.wav","p232_162.wav"]

#set the model
device = torch.device('cuda:{:d}'.format(rank))
generator_basic = BasicMambaGenerator(cfg).to(device)
generator = FusionMambaGenerator(generator_basic,cfg).to(device)
checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
if "generator" in checkpoint:
    generator.load_state_dict(checkpoint["generator"])
else:
    generator.load_state_dict(checkpoint)

generator.eval()


for noisy_file in noisy_files:
    noisy_file_clean_pha = noisy_file.split(".")[0]+"_clean_pha.wav"
    clean_path = clean_path_dir + noisy_file

    noisy_full_path = noisy_dir + noisy_file

    noisy_audio, noisy_mag, noisy_pha  = get_noisy_test_sample(noisy_full_path,cfg)
    noisy_mag, noisy_pha,noisy_audio = noisy_mag.to(device), noisy_pha.to(device),noisy_audio.to(device)

    # Forward pass
    with torch.no_grad():  # Disable gradient tracking for inference
        mag_g = generator(noisy_audio.unsqueeze(0),noisy_mag)


    #test with clean phase:
    audio_g = mag_phase_istft(mag_g, noisy_pha, n_fft, hop_size, win_size, compress_factor)
    audio_g_norm = normalize_energy(audio_g)

    # Example: PyTorch tensor containing audio data
    sample_rate = 16000  # Adjust to your sample rate

    # Save as WAV file
    output_path = output_dir + noisy_file
    
    audio_g_norm = audio_g_norm.cpu()  # Copy tensor to CPU
    torchaudio.save(output_path, audio_g_norm, sample_rate)

    print(f"Generated file was saved to {output_path}")

    #test now with clean phase:

    clean_pha = get_clean_phase(clean_path,cfg)
    clean_pha = clean_pha.to(device)
    audio_g_clean_pha = mag_phase_istft(mag_g, clean_pha, n_fft, hop_size, win_size, compress_factor)
    audio_g_clean_pha_norm = normalize_energy(audio_g)

    # Example: PyTorch tensor containing audio data
    sample_rate = 16000  # Adjust to your sample rate

    # Save as WAV file
    output_path_clean_pha = output_dir + noisy_file_clean_pha
    audio_g_clean_pha_norm = audio_g_clean_pha_norm.cpu()  # Copy tensor to CPU
    torchaudio.save(output_path_clean_pha, audio_g_clean_pha_norm, sample_rate)

    print(f"Generated file was saved to {output_path_clean_pha}")