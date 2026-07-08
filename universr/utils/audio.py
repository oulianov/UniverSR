import soundfile as sf
import torch


def load_audio_file(path: str, start: int = 0, frames: int | None = None) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
        start=start,
        frames=-1 if frames is None else frames,
    )
    waveform = torch.from_numpy(audio.T.copy())
    return waveform, sample_rate
