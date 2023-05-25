import torch
from torch.utils.data import Dataset
import numpy as np
import os
import librosa
import torchaudio
import torch.nn.functional as F
from tqdm.auto import tqdm
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


def load_data(dat_file):
    try:
        with open(dat_file, 'rb') as dat_fh:
            dat = pickle.load(dat_fh)
    except IOError:
        print(f'Could not open preprocessed data file {dat_file}.', file=stderr)
        stderr.flush()
    return dat

class WVDataset(Dataset):

    def __init__(self, audio_path, length, skip_size, sample_rate, max_files=0, win_length=400, is_generating = False):
        super(WVDataset, self).__init__()

        self.length = length
        self.skip_size = skip_size
        self.mulaw = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
        self.is_generating = is_generating
        self.sr = sample_rate
        # self.mfcc = torchaudio.transforms.MFCC(
        #     sample_rate=sample_rate,
        #     n_mfcc=13,
        #     melkwargs={"win_length": win_length, "hop_length": 100, "n_mels": 40}
        # )
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=30,
            melkwargs={"hop_length": 128, "n_mels": 64}
        )
        
        self.deltagen = torchaudio.transforms.ComputeDeltas(win_length= 9)

        path_list = os.listdir(audio_path)
        self.mfcc_max = 0
        self.mfcc_min = 9999999999999

        if max_files != 0:
            path_list = path_list[0:max_files]

        self.files = []

        for path in tqdm(path_list, desc='Loading and preprocessing files to dataset.', colour="blue"):
            full_path = os.path.join(audio_path, path)
            if is_audio_file(full_path):
                waveform = load_wav(full_path, sample_rate)

                onehot_wave, norm_audio, mulaw_audio = self.process_audio(waveform)
                # mfcc_long = self.process_mfcc(norm_audio)
                # print('MFCC size: ', mfcc_long.size(), 'mulaw_audio size: ', mulaw_audio.size())
                

                i = 0
                k = 0
                m = 0

                if is_generating:
                    with tqdm(total=4096*2, leave=False) as pbar:
                        for i in range(4096*2):
                            self.get_snippets(mulaw_audio, onehot_wave, norm_audio, i)
                            i += self.skip_size
                            k += 1
                            if k % 4096 == 4095:
                                m += 33
                            pbar.update(1)
                else:
                    with tqdm(total=norm_audio.size()[-1] // skip_size - 5, leave=False) as pbar:
                        while i < norm_audio.size()[-1] - self.length * 3:
                            self.get_snippets(mulaw_audio, onehot_wave, norm_audio, i)
                            i += self.skip_size
                            k += 1
                            if k % 2 == 1:
                                m += 33
                            pbar.update(1)
                
                # print('k = ', k)

    def process_audio(self, audio):
        """
        Process, normalise, mulaw encode and one hot encode audio.
        """
        audio = torch.from_numpy(audio)

        if audio.size()[0] == 2:  # Make mono if stereo
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        norm_audio = audio / torch.max(torch.abs(audio))  # Normalise to be between -1 and 1
        norm_audio = torch.clamp(norm_audio, -1.0, 1.0)
        
        mulawq = self.mulaw(norm_audio)
        audio = F.one_hot(mulawq, 256)

        return audio, norm_audio, mulawq

    def process_mfcc(self, y):
        mfcc = self.mfcc(y)
        # mfcc = torch.from_numpy(librosa.feature.mfcc(y=y.numpy(), sr=self.sr, hop_length= 128))
        mfcc_delta = self.deltagen(mfcc)
        # mfcc_delta2 = self.deltagen(mfcc_delta)
        mfcc = torch.cat((mfcc, mfcc_delta), dim=0)
        # mfcc = torch.cat((mfcc, mfcc_delta2), dim=0)
        
        
        mfcc /= torch.max(mfcc)

        return mfcc
    
    def get_snippets(self, mulaw_audio, onehot_audio, norm_audio, i):
        input_audio = mulaw_audio[i:i + self.length]
        target_sample = mulaw_audio[i:i + self.length + 1]
        mfcc = self.process_mfcc(norm_audio[i:i + self.length])
        # mfcc = mfcc_long[:, m:m+33]
        if self.is_generating:
            onehot_target = onehot_audio[i:i + self.length + 1]
            self.files.append((input_audio, mfcc, target_sample, onehot_target))
        else:
            self.files.append((input_audio, mfcc, target_sample))

        if torch.max(mfcc) > self.mfcc_max:
            self.mfcc_max = torch.max(mfcc)
        if torch.min(mfcc) < self.mfcc_min:
            self.mfcc_min = torch.min(mfcc)
        return

        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        onehot= 0
        mfcc = 0
        target = 0
        oht = 0
        if self.is_generating:
            onehot, mfcc, target, oht = self.files[idx]
            mfcc = (mfcc - self.mfcc_min) / (self.mfcc_max - self.mfcc_min)
            return onehot.type(torch.FloatTensor).unsqueeze(0), mfcc.type(torch.FloatTensor), target.type(torch.LongTensor), oht.type(torch.FloatTensor)
        else:
            onehot, mfcc, target = self.files[idx]
            mfcc = (mfcc - self.mfcc_min) / (self.mfcc_max - self.mfcc_min)
            return onehot.type(torch.FloatTensor).unsqueeze(0), mfcc.type(torch.FloatTensor), target.type(torch.LongTensor), oht
        # mfcc = (mfcc - self.mfcc_min) / (self.mfcc_max - self.mfcc_min)
        # print('WVDATA target size:', target.type(torch.LongTensor).size())
        
        # return torch.unsqueeze(onehot, 0).type(torch.FloatTensor), mfcc.type(torch.FloatTensor), target.type(
            # torch.FloatTensor)


"""
Utility Functions
"""


def load_wav(filename, sampling_rate, res_type='kaiser_fast', top_db=20, trimming_duration=None):
    raw, _ = librosa.load(filename, sr=sampling_rate, res_type=res_type)
    if trimming_duration is None:
        trimmed_audio, trimming_indices = librosa.effects.trim(raw, top_db=top_db)
        trimming_time = trimming_indices[0] / sampling_rate
    else:
        trimmed_audio = raw[int(trimming_duration * sampling_rate):]
        trimming_time = trimming_duration
    trimmed_audio /= np.abs(trimmed_audio).max()
    trimmed_audio = trimmed_audio.astype(np.float32)

    return trimmed_audio


def is_audio_file(
        filename):  # is_audio_file and load_wav from https://github.com/swasun/VQ-VAE-Speech/blob/master/src/dataset/vctk.py
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

class SliceDataset(Dataset):
    """
    Return slices of wav files of fixed size
    """
    def __init__(self, slice_size, n_win_batch, dat_file):
        self.slice_size = slice_size
        self.n_win_batch = n_win_batch 
        self.in_start = []
        
        dat = load_data(dat_file)
        self.samples = dat['samples']
        self.snd_data = dat['snd_data'].astype(dat['snd_dtype'])

        w = self.n_win_batch
        # for sam in self.samples:
        sam = self.samples[0]
        # for b in range(sam.wav_b, sam.wav_e - self.slice_size, w):
        for b in range(sam.wav_b, 4096 * 2, w):
            self.in_start.append((b, sam.voice_index))

    def num_speakers(self):
        ns = max(s.voice_index for s in self.samples) + 1
        return ns

    def __len__(self):
        return len(self.in_start)

    def __getitem__(self, item):
        s, voice_ind = self.in_start[item]
        return self.snd_data[s:s + self.slice_size], voice_ind