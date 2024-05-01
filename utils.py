import torch
import torchaudio

from numpy.random import RandomState
import numpy as np

import os
import random
import pandas as pd

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def _wav2fbank(self, filename, target_length=1024):

        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

        n_frames = fbank.shape[0]

        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

class UrbanSound(torch.utils.data.Dataset):
    def __init__(self, folds, targetLength=1024):
        super(UrbanSound, self).__init__()
        self.DATA_PATH = 'data/UrbanSound8K/'
        self.df = pd.read_csv(self.DATA_PATH + 'UrbanSound8K.csv')
        self.targetLength = targetLength
        self.folds = folds
        self.files = None
        self.labels = None
        self._read_files()

    def _read_files(self):
        self.df['fullPath'] = self.DATA_PATH + 'fold' + self.df['fold'].astype(str) + '/' + self.df['slice_file_name'].astype(str)
        self.df = self.df[self.df['fold'].isin(self.folds)]
        self.files = self.df['fullPath'].values
        self.labels = self.df['classID'].values
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.numclasses = len(self.df['class'].unique())

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        feature = _wav2fbank(filename, self.targetLength)
        label = self.labels[idx]
        return feature, label

class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, targetLength=1024):
        super(ESC50Dataset, self).__init__()
        self.data_dir = 'data/ESC-50-master/'
        self.targetLength = targetLength    
        self.metadata = pd.read_csv(self.data_dir + 'meta/' + 'esc50.csv')
        self.files = self.metadata['filename'].values
        self.labels = self.metadata['category'].values
        self.labels = pd.Categorical(self.labels).codes
        self.numclasses = len(self.metadata['category'].unique())
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        feature = _wav2fbank(self.data_dir + 'audio/' + filename, self.targetLength)
        label = self.labels[idx]
        return feature, label

