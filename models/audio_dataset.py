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
