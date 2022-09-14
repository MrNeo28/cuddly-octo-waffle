from ctypes import Array
import librosa

import numpy as np
import soundfile as sf

def to_mono(x):
    if x.ndim > 1:
            x = np.mean(x, axis=tuple(range(x.ndim - 1)))
    
    return x

def read_audio(y, mono: bool= True):
    x = sf.read(y)
    if mono:
        x = to_mono(x)
    return x.astype(np.float32)

def audio_to_melspectogram(audio, config):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=config.sampling_rate,
                                                 hop_length = config.hop_length,
                                                 n_mels = config.n_mels,
                                                 n_fft=config.n_fft,
                                                 fmin=config.fmin,
                                                 fmax=config.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(mels, config, title='Log-frequency power spectrogram'):
    import matplotlib.pyplot as plt
    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=config.sampling_rate, hop_length=config.hop_length,
                             fmin=config.fmin, fmax=config.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalise(arrs, min_val=None, max_val=None):
    arr2 = []
    for arr in arrs:
        if min_val is None:
            min_val = arr.min()
        if max_val is None:
            max_val = arr.max()
        arr = ((arr - min_val) * (1/(max_val - min_val) * 255))
        arr2.append(arr)
    return np.stack(arr2,axis=2).astype('uint8')