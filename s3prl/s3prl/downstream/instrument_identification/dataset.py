import random
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import os
from pathlib import Path

import torchaudio

SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200

INSTRUMENTS = [
    "cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"
]


class InstrumentDataset(Dataset):
    def __init__(self, df, split, batch_size, IRMAS_root, **kwargs):
        super(InstrumentDataset, self).__init__()
        self.df = df
        self.split = split
        self.batch_size = batch_size
        self.IRMAS_root = IRMAS_root
        self.class_num = 48
        self.trainDataset = self.getTrainingAudioInstrument()

        self.ins2idx = {instrument: idx for idx, instrument in enumerate(INSTRUMENTS)}        

    def __getitem__(self, idx):
        samples = random.randint(EXAMPLE_WAV_MIN_SEC * SAMPLE_RATE, EXAMPLE_WAV_MAX_SEC * SAMPLE_RATE)
        wav_path = self.df.loc[idx]['id']
        wav, sr = torchaudio.load(wav_path)
        label = 1
        return wav, label

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
    
    def getTrainingAudioInstrument(self):
        trainingAudioInstrument = {}
        for dir in os.listdir(self.IRMAS_root):
            dir = os.path.join(self.IRMAS_root, dir)
            if os.path.isdir(dir):
                for musicFile in Path(dir).glob('*.wav'):
                    audioIns = musicFile.split('/')[-1].split('__')[-2]
                    audioIns[-1] = audioIns[-1][:-4]
                    for ins in audioIns:
                        if ins not in INSTRUMENTS:
                            audioIns.remove(ins)
                    trainingAudioInstrument[musicFile] = audioIns
        
        df = pd.DataFrame.from_dict(trainingAudioInstrument, orient='index')
        return trainingAudioInstrument

