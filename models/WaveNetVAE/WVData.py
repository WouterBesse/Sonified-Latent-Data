import torch
from torch.utils.data import Dataset
import numpy as np
import os
import librosa
import torchaudio
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


class WVDataset(Dataset):

    def __init__(self, audio_path, length, skip_size, sample_rate, max_files=0, is_generating = False):
        super(WVDataset, self).__init__()

        self.length = length
        self.skip_size = skip_size
        self.mulaw = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
        self.is_generating = is_generating
        self.sr = sample_rate

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

                mulaw_audio = self.process_audio(waveform)
                
                self.add_snippets(mulaw_audio)
                                            

    def process_audio(self, audio):
        """
        Process, normalise, mulaw encode audio
        """
        audio = torch.from_numpy(audio)

        if audio.size()[0] == 2:  # Make mono if stereo
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        norm_audio = audio / torch.max(torch.abs(audio))  # Normalise to be between -1 and 1
        norm_audio = torch.clamp(norm_audio, -1.0, 1.0)
        
        mulawq = self.mulaw(norm_audio)
        # audio = F.one_hot(mulawq, 256)

        return mulawq
    
    def add_snippets(self, mulaw_audio):
        """
        Cut wave file in self.length sized pieces, skipping every
        """   
        sampletotal = 4096*2 if self.is_generating else mulaw_audio.size()[-1] - self.length

        for i in trange(0, sampletotal, self.skip_size, leave=False):
            input_audio = mulaw_audio[i:i + self.length + 1]
            self.files.append(input_audio)
        return


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        snippet = self.files[idx]
        return snippet.type(torch.FloatTensor)


class ProcessWav(object):
    """
    Class to get MFCC+derivatives from audio snippet, used in Collate()
    Sourced from: https://github.com/hrbigelow/ae-wavenet
    """   
    def __init__(self, sample_rate=16000, win_sz=400, hop_sz=160, n_mels=80,
            n_mfcc=13, name=None):
        self.sample_rate = sample_rate
        self.window_sz = win_sz
        self.hop_sz = hop_sz
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_out = n_mfcc * 3

    def __call__(self, wav):
        import librosa
        wav = wav.numpy()
        # See padding_notes.txt 
        # NOTE: This function can't be executed on GPU due to the use of
        # librosa.feature.mfcc
        # C, T: n_mels, n_timesteps
        # Output: C, T
        # This assert doesn't seem to work when we just want to process an entire wav file
        adj = 1 if self.window_sz % 2 == 0 else 0
        adj_l_wing_sz = (self.window_sz - 1)//2 + adj 

        left_pad = adj_l_wing_sz % self.hop_sz
        trim_left = adj_l_wing_sz // self.hop_sz
        trim_right = (self.window_sz - 1 - ((self.window_sz - 1)//2)) // self.hop_sz

        # wav = wav.numpy()
        wav_pad = np.concatenate((np.zeros(left_pad), wav), axis=0) 
        mfcc = librosa.feature.mfcc(y=wav_pad, sr=self.sample_rate,
                n_fft=self.window_sz, hop_length=self.hop_sz,
                n_mels=self.n_mels, n_mfcc=self.n_mfcc)

        def mfcc_pred_output_size(in_sz, window_sz, hop_sz):
            '''Reverse-engineered output size calculation derived by observing the
            behavior of librosa.feature.mfcc'''
            n_extra = 1 if window_sz % 2 == 0 else 0
            n_pos = in_sz + n_extra
            return n_pos // hop_sz + (1 if n_pos % hop_sz > 0 else 0)

        assert mfcc.shape[1] == mfcc_pred_output_size(wav_pad.shape[0],
            self.window_sz, self.hop_sz)

        mfcc_trim = mfcc[:,trim_left:-trim_right or None]

        mfcc_delta = librosa.feature.delta(mfcc_trim)
        mfcc_delta2 = librosa.feature.delta(mfcc_trim, order=2)
        mfcc_and_derivatives = np.concatenate((mfcc_trim, mfcc_delta, mfcc_delta2), axis=0)

        return mfcc_and_derivatives

class Collate():
    """
    Collate class to return the right data
    Sourced from: https://github.com/hrbigelow/ae-wavenet
    """   
    def __init__(self, mfcc, train_mode = True):
        self.train_mode = train_mode
        self.mfcc = mfcc

    def __call__(self, batch):
        # print(len(batch))

        wav = torch.stack([d for d in batch]).float()
        mel = torch.stack([torch.from_numpy(self.mfcc(d)) for d in batch]).float()

        if self.train_mode:
            return wav, mel
        else:
            paths = [b[0][2] for b in batch]
            return wav, mel
            
"""
Utility Functions
"""


def load_wav(filename, sampling_rate, res_type='kaiser_fast', top_db=20, trimming_duration=None):
    raw, _ = librosa.load(filename, sr=sampling_rate, res_type=res_type)
    # if trimming_duration is None:
    #     trimmed_audio, trimming_indices = librosa.effects.trim(raw, top_db=top_db)
    #     trimming_time = trimming_indices[0] / sampling_rate
    # else:
    #     trimmed_audio = raw[int(trimming_duration * sampling_rate):]
    #     trimming_time = trimming_duration
    # trimmed_audio /= np.abs(raw).max()
    trimmed_audio = raw.astype(np.float32)

    return trimmed_audio


def is_audio_file(
        filename):  # is_audio_file and load_wav from https://github.com/swasun/VQ-VAE-Speech/blob/master/src/dataset/vctk.py
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)
