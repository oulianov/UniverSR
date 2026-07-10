import os
import random
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader

from universr.utils.audio import load_audio_file
from universr.utils.utils import _worker_init_fn, read_file_list


def make_dataset(config, mode:str):
    return Dataset(
        **config.dataset.common,
        **config.dataset[mode],
        mode=mode,
    )

def prepare_dataloader(config):
    train_dataset = make_dataset(config, 'train')
    val_dataset = make_dataset(config, 'val')

    dl_args = dict(config.dataloader)
    val_batch_size = int(dl_args.pop("val_batch_size", 1))

    collator = WaveformCollator(
        target_sr=config.dataset.common.sr,
        sampling_rates_probs=config.collator.sampling_rates_probs,
        return_hr_only=config.collator.get("lowpass_on_device", False),
    )

    dl_args['worker_init_fn'] = _worker_init_fn
    dl_args['collate_fn'] = collator
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_args)
    
    collator = WaveformCollator(
        target_sr=config.dataset.common.sr,
        sampling_rates_probs=config.collator.validation_probs,
        return_hr_only=config.collator.get("lowpass_on_device", False),
    )
    val_loader_args = dict(dl_args)
    val_loader_args['worker_init_fn'] = _worker_init_fn
    val_loader_args['collate_fn'] = collator
    val_loader_args['batch_size'] = val_batch_size
    val_loader = DataLoader(val_dataset, shuffle=False, **val_loader_args)

    return train_loader, val_loader

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 file_list: str,
                 num_samples=24000,
                 sr=24000,
                 channel_mode="mono",
                 segment_read=False,
                 val_seconds=5,
                 val_segment_position="start",
                 mode="train"):
        if channel_mode not in {"mono", "mid-side"}:
            raise ValueError("channel_mode must be 'mono' or 'mid-side'.")
        if val_segment_position not in {"start", "middle"}:
            raise ValueError("val_segment_position must be 'start' or 'middle'.")
        self.num_samples = num_samples
        self.sr = sr
        self.channel_mode = channel_mode
        self.segment_read = segment_read
        self.val_seconds = val_seconds
        self.val_segment_position = val_segment_position
        self.mode = mode
        
        self.wb_paths = read_file_list(file_list)
        if self.channel_mode == "mid-side":
            self.wb_items = [(path, "mid") for path in self.wb_paths] + [(path, "side") for path in self.wb_paths]
        else:
            self.wb_items = [(path, None) for path in self.wb_paths]
        print(len(self.wb_items), 'samples loaded!')
        # derive common root directory to preserve input folder structure in outputs
        dirnames = [os.path.dirname(p) for p in self.wb_paths]
        self.common_root = os.path.commonpath(dirnames) if len(dirnames) > 0 else ""

    def __len__(self):
        return len(self.wb_items)

    def _pad(self, wav, N=80):
        pad = (N - wav.shape[-1] % N) % N
        return torch.nn.functional.pad(wav, (0,pad))

    def _ensure(self, wav, L, repeat=True):
        # if short: repeat, else: crop
        if wav.shape[-1] < L and repeat: 
            wav = torch.nn.functional.pad(wav, (0, 4000)) # offset
            reps = (L + wav.shape[-1] - 1) // wav.shape[-1]          # ceil(L / wav.shape[-1])
            wav = wav.repeat(1, reps)[..., :L]   # repeat
        elif wav.shape[-1] < L and not repeat:
            pad = L - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))        
        elif wav.shape[-1] > L:        
            wav = wav[..., :L]
        return wav

    def _target_len(self):
        if self.mode == "train":
            return self.num_samples
        if self.mode == "val":
            return self.sr * self.val_seconds
        raise ValueError(f"Unsupported mode '{self.mode}'. Expected 'train' or 'val'.")

    def _read_audio(self, wb_path):
        if not self.segment_read:
            return load_audio_file(wb_path)

        info = sf.info(wb_path)
        if info.frames <= 0 or info.samplerate <= 0:
            return load_audio_file(wb_path)

        target_input_frames = max(1, math.ceil(self._target_len() * info.samplerate / self.sr))
        if self.mode == "train" and info.frames > target_input_frames:
            max_start = max(0, info.frames - target_input_frames)
            start = int(np.random.randint(0, max_start + 1))
        elif self.mode == "val" and self.val_segment_position == "middle" and info.frames > target_input_frames:
            start = max(0, (info.frames - target_input_frames) // 2)
        else:
            start = 0
        frames = min(info.frames - start, target_input_frames)
        try:
            return load_audio_file(wb_path, start=start, frames=frames)
        except RuntimeError:
            return load_audio_file(wb_path)

    def __getitem__(self, idx):
        wb_path, channel_label = self.wb_items[idx]
        y, sr = self._read_audio(wb_path)

        # Apply one attenuation to the stereo segment before channel conversion.
        # This preserves the natural mid/side balance and never amplifies quiet
        # passages or low-energy side-channel noise.
        gain_db = float(np.random.uniform(-6.0, -1.0)) if self.mode == "train" else -3.0
        y = y * (10.0 ** (gain_db / 20.0))

        if self.channel_mode == "mid-side":
            if y.size(0) == 1:
                left = y
                right = y
            else:
                left = y[0:1]
                right = y[1:2]
            y = (left + right) * 0.5 if channel_label == "mid" else (left - right) * 0.5
        elif y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
        
        # resample
        if sr != self.sr:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sr)
        
        if self.mode=="train":
            target_signal_len = self._target_len()
            current_signal_len = y.shape[-1]
            if current_signal_len <= target_signal_len:
                y = self._ensure(y, target_signal_len)
            else:
                s = np.random.randint(0, current_signal_len - target_signal_len)
                y = y[..., s:s+target_signal_len]
        elif self.mode in ['val']:
            target_signal_len = self._target_len()
            current_signal_len = y.shape[-1]
            if current_signal_len < target_signal_len:
                y = self._ensure(y, target_signal_len, repeat=False)
            elif current_signal_len > target_signal_len and self.val_segment_position == "middle":
                s = max(0, (current_signal_len - target_signal_len) // 2)
                y = y[..., s:s + target_signal_len]
            else:
                y = y[..., :target_signal_len]
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'. Expected 'train' or 'val'.")
         
        outdict = {
            'hr': y,
            'filename': f"{Path(wb_path).stem}_{channel_label}" if channel_label else Path(wb_path).stem,
            'path': wb_path,
            'relpath': (
                os.path.join(channel_label, os.path.relpath(wb_path, self.common_root))
                if channel_label and self.common_root else
                os.path.relpath(wb_path, self.common_root) if self.common_root else Path(wb_path).name
            ),
        }

        return outdict

class WaveformCollator:
    def __init__(self, 
                 target_sr=48000, 
                 sampling_rates_probs={8: 0.7, 12: 0.1, 16: 0.1, 24: 0.1},
                 return_hr_only=False):
        """
        Initializes the collator.
        Args:
            target_sr (int): The high-resolution sample rate (e.g., 48000).
            sampling_rates_probs (dict): A dictionary mapping sample rates (in kHz) to their sampling probabilities.
                                         Example: {8: 0.7, 12: 0.1, 16: 0.1, 24: 0.1}
        """
        self.target_sr = target_sr
        self.sampling_rates = list(sampling_rates_probs.keys())  # [8, 12, 16, 24]
        self.probs = list(sampling_rates_probs.values()) # [0.7, 0.1, 0.1, 0.1]
        self.return_hr_only = return_hr_only

    def _apply_lpf(self, hr_wave, low_sr_khz):
        """
        Applies a low-pass filter by downsampling and then upsampling the waveform.
        This correctly simulates the anti-aliasing filter effect.
        """
        original_len = hr_wave.shape[-1]
        target_sr_hz = low_sr_khz * 1000
        
        # Downsample to the target low sample rate
        lr_wave_resampled = torchaudio.functional.resample(
            hr_wave, orig_freq=self.target_sr, new_freq=target_sr_hz
        )
        # Upsample back to the original high sample rate to match lengths
        lr_wave_upsampled = torchaudio.functional.resample(
            lr_wave_resampled, orig_freq=target_sr_hz, new_freq=self.target_sr
        )
        
        lr_wave_upsampled = lr_wave_upsampled[..., :original_len]
        return lr_wave_upsampled

    def __call__(self, batch):
        """
        Processes a batch of data items from the Dataset.
        """
        # 1. Choose one low_sr for the entire batch based on given probabilities
        low_sr_khz = random.choices(self.sampling_rates, self.probs, k=1)[0]
        
        # 2. Stack HR waveforms from the batch items
        # Assuming dataset returns {'hr': tensor_shape_[1, T], ...}
        hr_waves = torch.stack([item['hr'].squeeze(0) for item in batch])

        # 3. Create LR versions by applying the LPF
        outdict = {
            'hr': hr_waves.unsqueeze(1),       # Shape: [B, 1, T]
            'low_sr': [low_sr_khz] * len(batch), # [B]
            'filename': [item['filename'] for item in batch], # [B]
            'relpath': [item.get('relpath', item['filename']) for item in batch], # [B]
            'path': [item.get('path', item['filename']) for item in batch], # [B]
        }

        if not self.return_hr_only:
            lr_waves = self._apply_lpf(hr_waves, low_sr_khz)
            outdict['lr_wave'] = lr_waves.unsqueeze(1) # Shape: [B, 1, T]

        # 4. Return a dictionary that matches the trainer's expectation
        return outdict
