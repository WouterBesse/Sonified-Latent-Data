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

    def __init__(self, audio_path, length, sample_rate, hop_length = 160):
        super(WVDataset, self).__init__()


        # Operations to keep sample length a multiple of the hop length
        # length += hop_length * 6 # Seems to make sure that the mfcc length is an even number :)
        self.length = length
        # self.length -= length % hop_length
        # half_mfcc_len = self.length // hop_length // 2 + 1
        # print(half_mfcc_len)
        # self.length -= (self.length % half_mfcc_len)
        # self.length += half_mfcc_len
        print(self.length)
        # self.length = int(self.length)

        self.skip_size = self.length // 2
        self.mulaw = torchaudio.transforms.MuLawEncoding(quantization_channels = 256)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate = sample_rate,
            n_mfcc = 40,
            melkwargs = {"hop_length": hop_length}
        )

        path_list = os.listdir(audio_path)

        self.files = []

        for path in tqdm(path_list, desc='Loading and preprocessing files to dataset.'):
            full_path = os.path.join(audio_path, path)
            if is_audio_file(full_path):
                waveform = load_wav(full_path, sample_rate)

                onehot_wave, norm_audio, mulaw_audio = self.process_audio(waveform)

                i = 0

                while i < norm_audio.size()[-1] - self.length:
                    input_audio = onehot_wave[i:i + self.length]
                    target_sample = mulaw_audio[i + self.length:i + self.length + 1]
                    mfcc = self.process_mfcc(norm_audio[i:i + self.length])
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

        mulawq = self.mulaw(norm_audio)
        audio = F.one_hot(mulawq, 256)

        return audio, norm_audio, mulawq

    def process_mfcc(self, audio):
        mfcc = self.mfcc(audio)
        mfcc /= torch.max(mfcc)

        return mfcc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        onehot, mfcc, target = self.files[idx]

        # , self.pure_noise[idx]#, waveform_noisy_unquantized
        return torch.transpose(onehot, 0, 1).type(torch.FloatTensor), mfcc.type(torch.FloatTensor), target[-1].type(torch.LongTensor)



"""
Utility Functions
"""
def load_wav(filename, sampling_rate, res_type = 'kaiser_fast', top_db = 20, trimming_duration=None):
    raw, _ = librosa.load(filename, sr = sampling_rate, res_type=res_type)
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

