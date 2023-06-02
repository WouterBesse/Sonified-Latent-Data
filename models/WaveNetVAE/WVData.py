import torch
from torch.utils.data import Dataset
import numpy as np
import os
import librosa
import torchaudio
from torchaudio.functional import compute_deltas
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


class WVDataset(Dataset):
    """
    Custom dataset for the WaveVAE model.
    """
    def __init__(self, audio_path, length, skip_size, sample_rate, max_files=0, is_generating = False, train = False, device = 'cuda'):
        super(WVDataset, self).__init__()

        self.length = length
        self.skip_size = skip_size
        self.mulaw = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
        self.device = device
        self.is_generating = is_generating
        self.sr = sample_rate
        self.mfcc = MFCCwDeriv(device=device)
        self.resample = torchaudio.transforms.Resample(22050, self.sr).to(device) # Resample to 22050Hz, yet to implement dynamic resampling

        # Values for MFCC normalisation
        self.mfcc_max = 0
        self.mfcc_min = 99999999

        self.files = []
        filecount = 0
        fileamount = get_path_length(audio_path)
        for full_path in tqdm(walkdir(audio_path), total=fileamount, desc='Loading and preprocessing files to dataset.', colour="blue"):
            """
            Load audio files and add them to the dataset
            The first 32 files are used for testing, the rest for training
            """
            if is_audio_file(full_path) and filecount < max_files:
              if not train and filecount < 32:                
                mulaw_audio, norm_audio = self.process_audio(full_path)
                self.add_snippets(mulaw_audio, norm_audio)
              
              elif train and filecount > 32:
                mulaw_audio, norm_audio = self.process_audio(full_path)
                self.add_snippets(mulaw_audio, norm_audio)

              filecount += 1
              
    def process_audio(self, full_path):
        """Load, flatten, normalise, mulaw encode audio
        Args:
            full_path (String): path of audio file
        Returns:
            mulawq (Tensor): Mulaw encoded audio, shape (timesteps)
            norm_audio (Tensor): Normalised audio, shape (timesteps)
        """
        audio, _ = torchaudio.load(full_path)
        audio = self.resample(audio.to(self.device))

        if audio.size()[0] == 2:  # Make mono if stereo
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        
        audio = audio.squeeze()
        norm_audio = audio / torch.max(torch.abs(audio))  # Normalise to be between -1 and 1
        norm_audio = torch.clamp(norm_audio, -1.0, 1.0)
        
        mulawq = self.mulaw(norm_audio.squeeze())
        # audio = F.one_hot(mulawq, 256)

        return mulawq, norm_audio
    
    def add_snippets(self, mulaw_audio, norm_audio):
        """Cut wave file in self.length sized pieces, skipping every sel.skip_size samples
            Extracts the MFCC+derivatives for each snippet and adds them to the dataset
        Args:
            mulawq (Tensor): Mulaw encoded audio, shape (timesteps)
            norm_audio (Tensor): Normalised audio, shape (timesteps)
        """
        sampletotal = 32000 if self.is_generating else mulaw_audio.size()[-1] - self.length

        for i in trange(0, sampletotal, self.skip_size, leave=False):
            input_audio = mulaw_audio[i:i + self.length + 1].cpu()
            mfcc = self.mfcc(norm_audio[i:i + self.length + 1]).cpu()

            # Update MFCC normalisation values
            if mfcc.max() > self.mfcc_max:
                self.mfcc_max = mfcc.max()
            if mfcc.min() < self.mfcc_min:
                self.mfcc_min = mfcc.min()

            self.files.append([input_audio, mfcc])
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get item from dataset
        Args:
            idx (Scalar): which index of the dataset to get
        Returns:
            snippet (Tensor): Mulaw encoded audio, shape (timesteps)
            mfcc (Tensor): MFCC+derivatives, shape (39 x 112)
        """
        snippet, mfcc = self.files[idx]
        mfcc = (mfcc - self.mfcc_min) / (self.mfcc_max - self.mfcc_min) # Normalise MFCCs
        return snippet.type(torch.FloatTensor), mfcc.type(torch.FloatTensor)

    
"""
Utility Functions/Classes
"""
class MFCCwDeriv(object):
    """
    Class to get MFCC+derivatives from audio snippet
    Inspired by the ProcessWav class at: https://github.com/hrbigelow/ae-wavenet
    """   
    def __init__(self, sample_rate=16000, win_sz=400, hop_sz=175, n_mels=80,
            n_mfcc=13, device='cuda'):
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                         n_mfcc=n_mfcc,
                                                         melkwargs={"n_fft": win_sz, "hop_length": hop_sz, "n_mels": n_mels, }).to(device)

    def __call__(self, wav):
        mfcc = self.mfcc_transform(wav)        
        mfcc_delta = compute_deltas(mfcc)
        mfcc_delta2 = compute_deltas(mfcc_delta)
        mfcc_and_derivatives = torch.concatenate((mfcc, mfcc_delta, mfcc_delta2), dim=0)

        return mfcc_and_derivatives

def is_audio_file(
        filename):  # is_audio_file from https://github.com/swasun/VQ-VAE-Speech/blob/master/src/dataset/vctk.py
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def walkdir(folder):
    """Walk through every files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))
            
def get_path_length(path):
    length = 0
    for _ in walkdir(path):
        length += 1
    return length