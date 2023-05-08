import torch
from torch.utils.data import Dataset
import numpy as np
import os
import librosa
import torchaudio
import torch.nn.functional as F
from tqdm.auto import tqdm

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


class WVDataset(Dataset):

    def __init__(self, audio_path, length, skip_size, sample_rate, max_files=0, hop_length=160, is_generating = False):
        super(WVDataset, self).__init__()

        self.length = length
        self.skip_size = skip_size
        self.mulaw = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={"hop_length": hop_length}
        )

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

                i = 0

                if is_generating:
                    with tqdm(total=4096, leave=False) as pbar:
                        for i in range(4096):
                            get_snippets()
                            i += self.skip_size
                            pbar.update(1)
                else:
                    with tqdm(total=norm_audio.size()[-1] // skip_size - 2, leave=False) as pbar:
                        while i < norm_audio.size()[-1] - self.length:
                            get_snippets()
                            i += self.skip_size
                            pbar.update(1)
                            

    def process_audio(self, audio):
        """
        Process, normalise, mulaw encode and one hot encode audio.
        """
        audio = torch.from_numpy(audio)

        if audio.size()[0] == 2:  # Make mono if stereo
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        norm_audio = audio / torch.max(torch.abs(audio))  # Normalise to be between -1 and 1

        mulawq = self.mulaw(norm_audio)
        audio = F.one_hot(mulawq, 256)

        return audio, norm_audio, mulawq

    def process_mfcc(self, audio):
        mfcc = self.mfcc(audio)
        mfcc /= torch.max(mfcc)

        return mfcc
    
    def get_snippets(self, mulaw_audio, onehot_audio, norm_audio, i):
        input_audio = mulaw_audio[i:i + self.length]
        target_sample = mulaw_audio[i:i + self.length + 1]
        # onehot_target = onehot_wave[i:i + self.length + 1]
        mfcc = self.process_mfcc(norm_audio[i:i + self.length])
        self.files.append((input_audio, mfcc, target_sample))

        if torch.max(mfcc) > self.mfcc_max:
            self.mfcc_max = torch.max(mfcc)
        if torch.min(mfcc) < self.mfcc_min:
            self.mfcc_min = torch.min(mfcc)
        return

        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        onehot, mfcc, target = self.files[idx]
        # mfcc = (mfcc - self.mfcc_min) / (self.mfcc_max - self.mfcc_min)
        # print('WVDATA target size:', target.type(torch.LongTensor).size())
        return onehot.type(torch.LongTensor), mfcc.type(torch.FloatTensor), target.type(torch.LongTensor)
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
