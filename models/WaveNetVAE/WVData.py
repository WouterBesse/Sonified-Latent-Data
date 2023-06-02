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
    Custom dataset that slices audio up and mulaw quantises it
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
        # path_list = os.listdir(audio_path)
        # root, subdirs, path_list = os.walk(audio_path)
        # _, osr = torchaudio.load(os.path.join(audio_path, path_list[2]))
        self.resample = torchaudio.transforms.Resample(22050, self.sr).to(device)
        self.mfcc_max = 0
        self.mfcc_min = 99999999
        

        # if max_files != 0:
        #     path_list = path_list[0:max_files]

        self.files = []
        filecount = 0
        fileamount = get_path_length(audio_path)
        for full_path in tqdm(walkdir(audio_path), total=fileamount, desc='Loading and preprocessing files to dataset.', colour="blue"):
            # full_path = os.path.join(audio_path, path)
            if is_audio_file(full_path) and filecount < max_files:
              if train and filecount < 32:   
                waveform, _ = torchaudio.load(full_path)
                waveform = self.resample(waveform.to(device))

                mulaw_audio, norm_audio = self.process_audio(waveform)
                
                self.add_snippets(mulaw_audio, norm_audio)
              
              elif not train and filecount > 32:
                waveform, _ = torchaudio.load(full_path)
                waveform = self.resample(waveform.to(device))

                mulaw_audio, norm_audio = self.process_audio(waveform)
                
                self.add_snippets(mulaw_audio, norm_audio)

              filecount += 1
              
                                            

    def process_audio(self, audio):
        """
        Process, normalise, mulaw encode audio
        """

        if audio.size()[0] == 2:  # Make mono if stereo
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        
        audio = audio.squeeze()
        norm_audio = audio / torch.max(torch.abs(audio))  # Normalise to be between -1 and 1
        norm_audio = torch.clamp(norm_audio, -1.0, 1.0)
        
        mulawq = self.mulaw(norm_audio.squeeze())
        # audio = F.one_hot(mulawq, 256)

        return mulawq, norm_audio
    
    def add_snippets(self, mulaw_audio, norm_audio):
        """
        Cut wave file in self.length sized pieces, skipping every
        """   
        sampletotal = 32000 if self.is_generating else mulaw_audio.size()[-1] - self.length

        for i in trange(0, sampletotal, self.skip_size, leave=False):
            input_audio = mulaw_audio[i:i + self.length + 1].cpu()
            mfcc = self.mfcc(norm_audio[i:i + self.length + 1]).cpu()
            if mfcc.max() > self.mfcc_max:
                self.mfcc_max = mfcc.max()
            if mfcc.min() < self.mfcc_min:
                self.mfcc_min = mfcc.min()
            self.files.append([input_audio, mfcc])
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        snippet, mfcc = self.files[idx]
        mfcc = (mfcc - self.mfcc_min) / (self.mfcc_max - self.mfcc_min)
        return snippet.type(torch.FloatTensor), mfcc.type(torch.FloatTensor)


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
    
class ProcessWav(object):
    """
    Class to get MFCC+derivatives from audio snippet
    Sourced from: https://github.com/hrbigelow/ae-wavenet
    """   
    def __init__(self, sample_rate=16000, win_sz=400, hop_sz=160, n_mels=80,
            n_mfcc=13, name=None):
        self.sample_rate = sample_rate
        self.window_sz = win_sz
        self.hop_sz = hop_sz
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate,
                                                         n_mfcc=n_mfcc,
                                                         melkwargs={"n_fft": win_sz, "hop_length": hop_sz, "n_mels": n_mels, }).cuda()

    def __call__(self, wav):
        # See padding_notes.txt 
        # NOTE: This function can't be executed on GPU due to the use of
        # librosa.feature.mfcc
        # C, T: n_mels, n_timesteps
        # Output: C, T
        # This assert doesn't seem to work when we just want to process an entire wav file
        import librosa
        adj = 1 if self.window_sz % 2 == 0 else 0
        adj_l_wing_sz = (self.window_sz - 1)//2 + adj 

        left_pad = adj_l_wing_sz % self.hop_sz
        trim_left = adj_l_wing_sz // self.hop_sz
        trim_right = (self.window_sz - 1 - ((self.window_sz - 1)//2)) // self.hop_sz

        wav_pad = torch.cat((torch.zeros(left_pad).cuda(), wav), dim=0) 
        mfcc = self.mfcc_transform(wav_pad)

        def mfcc_pred_output_size(in_sz, window_sz, hop_sz):
            '''Reverse-engineered output size calculation derived by observing the
            behavior of librosa.feature.mfcc'''
            n_extra = 1 if window_sz % 2 == 0 else 0
            n_pos = in_sz + n_extra
            return n_pos // hop_sz + (1 if n_pos % hop_sz > 0 else 0)

        assert mfcc.size()[1] == mfcc_pred_output_size(wav_pad.size()[0],
            self.window_sz, self.hop_sz)

        mfcc_trim = mfcc[:,trim_left:-trim_right or None]
        
        mfcc_delta = compute_deltas(mfcc_trim)
        mfcc_delta2 = librosa.feature.delta(mfcc_trim.cpu().numpy(), order=2)
        mfcc_and_derivatives = torch.concatenate((mfcc_trim, mfcc_delta, torch.from_numpy(mfcc_delta2).cuda()), dim=0)

        return mfcc_and_derivatives
            
"""
Utility Functions
"""


# def load_wav(filename, sampling_rate, res_type='kaiser_fast', top_db=20, trimming_duration=None):
#     raw, _ = librosa.load(filename, sr=sampling_rate, res_type=res_type)
#     # raw, osr = torchaudio.load(filename)
#     # if trimming_duration is None:
#     #     trimmed_audio, trimming_indices = librosa.effects.trim(raw, top_db=top_db)
#     #     trimming_time = trimming_indices[0] / sampling_rate
#     # else:
#     #     trimmed_audio = raw[int(trimming_duration * sampling_rate):]
#     #     trimming_time = trimming_duration
#     # trimmed_audio /= np.abs(raw).max()
#     trimmed_audio = raw.astype(np.float32)

#     return torch.from_numpy(trimmed_audio)


def is_audio_file(
        filename):  # is_audio_file and load_wav from https://github.com/swasun/VQ-VAE-Speech/blob/master/src/dataset/vctk.py
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