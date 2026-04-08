import soundfile as sf
import torch


def load_audio_file(path: str) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    return waveform, sample_rate
