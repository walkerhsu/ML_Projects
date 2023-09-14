import random
import json

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import os
from pathlib import Path

from torchaudio.transforms import Resample

import torchaudio

SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200


class InstrumentDataset(Dataset):
    def __init__(self, data_path, split, batch_size, pre_load=True, **kwargs):
        super(InstrumentDataset, self).__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        self.split = split
        self.batch_size = batch_size
        self.instruments = self.data["instruments"]
        self.idx2instruments = {value: key for key, value in self.instruments.items()}
        self.instruments_num = len(self.instruments)
        self.meta_data = self.data["meta_data"]
        _, origin_sr = torchaudio.load(self.meta_data[0]["path"])
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path)
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.instruments[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return EXAMPLE_DATASET_SIZE

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels

    def getInstrument(self, idx):
        return INSTRUMENTS[idx]

    def getInstrumentIdx(self, instrument):
        return self.ins2idx[instrument]
