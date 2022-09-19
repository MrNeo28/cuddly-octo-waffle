import torch
import torch.utils.data as torchdata

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])


class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 mode='train'):
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]

        wav_path = sample["file_path"]
        labels = sample["primary_label"]

        image = np.load(wav_path)
        if mode == "train":
            image = augment(image, sample_rate=config.sample_rate)

        targets = np.zeros(len(config.target_columns), dtype=float)
        for ebird_code in labels.split():
            targets[config.target_columns.index(ebird_code)] = 1.0

        return {
            "image": image,
            "primary_targets": targets,
        }



class BirdDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, waveform_transforms = None, validation = False):
        self.df = df
        self.waveform_transforms = waveform_transforms
        self.validation = validation
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int):
        
        label, secondary_labels, filename = TRAIN_DF[['primary_label', 'secondary_labels', 'filename']].iloc[index]
        
        targets = np.zeros(config.num_classes, dtype=float)
        targets[BIRDS_CODE[label]] = 1.
        
        for birds in secondary_labels:
            if birds != "" and birds in BIRDS_CODE.keys():
                targets[BIRDS_CODE[birds]] = 1.
        
        
        y, sr = audiofile.read(Path(config.root) / 'train_audio' / filename)
        y = librosa.to_mono(y)
        
        len_y = len(y)
        
        effective_length = int(np.ceil(sr * config.period))
        if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
        elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
        else:
                y = y.astype(np.float32)

        if self.waveform_transforms:
            y = self.waveform_transforms(y, sample_rate=config.sample_rate)

        y = np.nan_to_num(y)


        return torch.tensor(y), targets
                           