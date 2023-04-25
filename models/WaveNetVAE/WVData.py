import torch
from torch.utils.data import Dataset
import numpy as np
import os
import librosa
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]

class WVDataset(Dataset):

    def __init__(self, audio_path, length, sample_rate):
        super(WVDataset, self).__init__()

        self.length = length
        self.skip_size = length // 2
        self.mulaw = torchaudio.transforms.MuLawEncoding(quantization_channels = 256)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate = sample_rate,
            n_mfcc = 40,
            melkwargs = {"hop_length": 160}
        )

        path_list = os.listdir(audio_path)

        self.files = []

        for path in tqdm(path_list, desc='Loading and preprocessing files to dataset.'):
            full_path = os.path.join(audio_path, path)
            if is_audio_file(full_path):
                waveform = load_wav(full_path, sample_rate)

                onehot_wave, norm_audio = self.process_audio(waveform)

                i = 0

                while i < norm_audio.size()[-1] - self.skip_size:
                    input_audio = onehot_wave[i:i + length]
                    target_sample = norm_audio[i + length:i + length + 1] # might be squeezed!!!
                    mfcc = self.process_mfcc(norm_audio[i:i + length])
                    self.files.append((input_audio, mfcc, target_sample))

                    i += self.skip_size


    def process_audio(self, audio):
        """
        Process, normalise, mulaw encode and one hot encode audio.
        """
        audio = torch.from_numpy(audio)

        if audio.size()[0] == 2: # Make mono if stereo
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        norm_audio = audio / torch.max(torch.abs(audio)) # Normalise to be between -1 and 1

        audio = self.mulaw(norm_audio)
        audio = F.one_hot(audio, 256)

        return audio, norm_audio

    def process_mfcc(self, audio):
        mfcc = self.mfcc(audio)
        mfcc /= torch.max(mfcc)

        return mfcc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        onehot, mfcc, target = self.files[idx]

        # , self.pure_noise[idx]#, waveform_noisy_unquantized
        return onehot, mfcc, target[-1]



"""
Utility Functions
"""
def load_wav(filename, sampling_rate, res_type = 'kaiser_fast', top_db = 20, trimming_duration=None):
    raw, _ = librosa.load(filename, sampling_rate, res_type=res_type)
    if trimming_duration is None:
        trimmed_audio, trimming_indices = librosa.effects.trim(raw, top_db=top_db)
        trimming_time = trimming_indices[0] / sampling_rate
    else:
        trimmed_audio = raw[int(trimming_duration * sampling_rate):]
        trimming_time = trimming_duration
    trimmed_audio /= np.abs(trimmed_audio).max()
    trimmed_audio = trimmed_audio.astype(np.float32)

    return trimmed_audio

def is_audio_file(filename): # is_audio_file and load_wav from https://github.com/swasun/VQ-VAE-Speech/blob/master/src/dataset/vctk.py
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

