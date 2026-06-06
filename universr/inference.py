"""
UniverSR: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching
Inference wrapper module.
"""

import logging
import os
import time
from typing import Literal, Optional, Union

import numpy as np
import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download

from universr.models.unet import ConvNeXtUNetCond
from universr.flow.path import OriginalCFMPath
from universr.flow.solver import CFGVectorFieldODE, VectorFieldODE, TorchDiffeqSolver
from universr.utils.audio import load_audio_file
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT


# Supported input sample rates (kHz) and their corresponding LR frequency bins
SUPPORTED_INPUT_SR = {8000, 12000, 16000, 24000}
TARGET_SR = 48000
RECONSTRUCTION_METHODS = {"original", "original_signal"}
ReconstructionMethod = Literal["original", "original_signal"]
logger = logging.getLogger(__name__)


class UniverSR(torch.nn.Module):
    """
    UniverSR inference wrapper.

    Performs audio super-resolution from low sample rates (8/12/16/24 kHz)
    to 48 kHz using vocoder-free flow matching in the complex STFT domain.

    Example:
        >>> model = UniverSR.from_pretrained("woongzip1/universr-speech")
        >>> output = model.enhance("input.wav", input_sr=16000)
        >>> torchaudio.save("output.wav", output.cpu(), 48000)
    """

    def __init__(
        self,
        model: ConvNeXtUNetCond,
        transform: AmplitudeCompressedComplexSTFT,
        path: OriginalCFMPath,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.transform = transform
        self.path = path
        self._device = device
        self.debug_logs = False

    def _debug_logs_enabled(self) -> bool:
        env_value = os.environ.get("UNIVERSR_DEBUG_LOGS", "").strip().lower()
        return bool(self.debug_logs) or env_value in {"1", "true", "yes", "on"}

    def _sync_timing_device(self) -> None:
        if (
            isinstance(self._device, str)
            and self._device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()

    def _start_timing(self) -> float:
        self._sync_timing_device()
        return time.perf_counter()

    def _finish_timing(self, started_at: float) -> float:
        self._sync_timing_device()
        return time.perf_counter() - started_at

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        device: str = "cuda",
        revision: Optional[str] = None,
        checkpoint_filename: str = "pytorch_model.bin",
    ) -> "UniverSR":
        """
        Load a pretrained UniverSR model.

        Args:
            repo_id_or_path: HuggingFace repo ID (e.g. "woongzip1/universr-speech")
                             or local directory path containing config.yaml and checkpoint file.
            device: Device to load the model on.
            revision: Optional HuggingFace revision (branch, tag, or commit hash).
            checkpoint_filename: Checkpoint filename to load from the repo or local directory.

        Returns:
            UniverSR instance ready for inference.
        """
        if os.path.isdir(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.yaml")
            model_path = os.path.join(repo_id_or_path, checkpoint_filename)
        else:
            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.yaml", revision=revision
            )
            model_path = hf_hub_download(
                repo_id=repo_id_or_path,
                filename=checkpoint_filename,
                revision=revision,
            )

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Build model
        model = ConvNeXtUNetCond(**config["model"])
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        # Build transform
        transform = AmplitudeCompressedComplexSTFT(**config["transform"])
        transform.to(device)

        # Build probability path
        path_args = config.get("path", {}).get("init_args", {"sigma_min": 1e-4})
        path = OriginalCFMPath(**path_args)

        return cls(model=model, transform=transform, path=path, device=device)

    @classmethod
    def from_local(
        cls,
        ckpt_path: str,
        config_path: str,
        device: str = "cuda",
    ) -> "UniverSR":
        """
        Load UniverSR from a local checkpoint (e.g. training checkpoint with optimizer state).

        This handles the standard training checkpoint format where weights are stored
        under the 'model_state_dict' key. from_pretrained() can also load local
        checkpoints, but this constructor is still useful when the config path
        needs to be provided explicitly.

        Args:
            ckpt_path: Path to checkpoint file (.pth).
            config_path: Path to YAML config file.
            device: Device to load the model on.

        Returns:
            UniverSR instance ready for inference.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model = ConvNeXtUNetCond(**config["model"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle both formats: raw state_dict or training checkpoint
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.to(device).eval()

        transform = AmplitudeCompressedComplexSTFT(**config["transform"])
        transform.to(device)

        path_args = config.get("path", {}).get("init_args", {"sigma_min": 1e-4})
        path = OriginalCFMPath(**path_args)

        return cls(model=model, transform=transform, path=path, device=device)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def enhance(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        input_sr: Optional[int] = None,
        target_sr: int = TARGET_SR,
        ode_method: str = "midpoint",
        ode_steps: int = 4,
        guidance_scale: Optional[float] = 1.5,
        reconstruction_method: ReconstructionMethod = "original",
    ) -> torch.Tensor:
        """
        Enhance a low-resolution audio signal to high-resolution.

        Args:
            audio: Input audio. Can be:
                   - str: path to a .wav file
                   - torch.Tensor: waveform tensor of shape (T,), (1, T), or (1, 1, T)
                   - np.ndarray: waveform array
            input_sr: Effective bandwidth of the input in Hz (e.g. 8000, 16000).
                      For file input: auto-detected from the file's native sample rate
                      if it matches a supported rate (8/12/16/24 kHz). Required if the
                      file is already at 48 kHz but has limited bandwidth.
                      For tensor/array input: always required.
            target_sr: Target sample rate in Hz. Default: 48000.
            ode_method: ODE solver method. One of 'euler', 'midpoint', 'rk4'.
            ode_steps: Number of ODE integration steps.
            guidance_scale: Classifier-free guidance scale. None or 0 disables CFG.
            reconstruction_method: Spectrum assembly method. "original" keeps the
                   legacy behavior and uses the bandwidth-limited model input for
                   low-frequency reconstruction. "original_signal" keeps the
                   bandwidth-limited signal as the model condition but uses the
                   unfiltered 48 kHz input for the final low-frequency bins.

        Returns:
            Enhanced waveform tensor of shape (1,T) at target_sr.
        """
        # Load audio
        if reconstruction_method not in RECONSTRUCTION_METHODS:
            raise ValueError(
                f"Unsupported reconstruction_method={reconstruction_method!r}. "
                f"Supported methods: {sorted(RECONSTRUCTION_METHODS)}"
            )

        wav, file_sr = self._load_audio(audio, input_sr=input_sr)
        wav = wav.to(self._device)

        # Determine the effective bandwidth SR
        effective_sr = input_sr if input_sr is not None else file_sr

        if effective_sr not in SUPPORTED_INPUT_SR:
            if effective_sr == target_sr and input_sr is None:
                raise ValueError(
                    f"Input audio is already at {target_sr} Hz. "
                    f"Please specify input_sr to indicate the effective bandwidth "
                    f"(e.g., input_sr=16000). Supported: {sorted(SUPPORTED_INPUT_SR)}"
                )
            raise ValueError(
                f"Effective input sample rate {effective_sr} Hz is not supported. "
                f"Supported rates: {sorted(SUPPORTED_INPUT_SR)}"
            )

        original_wav_48k = wav

        # Prepare the 48 kHz LR input for the model
        if file_sr == target_sr:
            # Simulate the training degradation: downsample → upsample to match
            wav = self._apply_bandwidth_limit(wav, effective_sr, target_sr)
        elif file_sr != target_sr:
            # File is truly low-resolution; resample up to 48 kHz
            original_wav_48k = torchaudio.functional.resample(
                wav, orig_freq=file_sr, new_freq=target_sr
            )
            wav = original_wav_48k

        # Minimum length guard
        MIN_SAMPLES = 32_768
        original_len = wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, max(0, MIN_SAMPLES - wav.shape[-1])))
        if reconstruction_method == "original_signal":
            if original_wav_48k.shape[-1] > wav.shape[-1]:
                original_wav_48k = original_wav_48k[..., : wav.shape[-1]]
            elif original_wav_48k.shape[-1] < wav.shape[-1]:
                original_wav_48k = torch.nn.functional.pad(
                    original_wav_48k,
                    (0, wav.shape[-1] - original_wav_48k.shape[-1]),
                )

        # Ensure shape is [B, C, T] = [1, 1, T]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)
        if reconstruction_method == "original_signal":
            if original_wav_48k.dim() == 1:
                original_wav_48k = original_wav_48k.unsqueeze(0).unsqueeze(0)
            elif original_wav_48k.dim() == 2:
                original_wav_48k = original_wav_48k.unsqueeze(0)

        sr_khz = effective_sr // 1000

        # Run flow matching SR
        output = self._inference(
            wav,
            sr_khz,
            ode_method,
            ode_steps,
            guidance_scale,
            reconstruction_method=reconstruction_method,
            reconstruction_audio=(
                original_wav_48k if reconstruction_method == "original_signal" else None
            ),
        )

        # (1,T)
        return output[..., :original_len]

    # ------------------------------------------------------------------ #
    #  Internal methods                                                   #
    # ------------------------------------------------------------------ #

    def _load_audio(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        input_sr: Optional[int] = None,
    ) -> tuple:
        """
        Load and validate audio input.

        Returns:
            (waveform, file_sr): The waveform tensor and its *actual* sample rate.
        """
        if isinstance(audio, str):
            wav, file_sr = load_audio_file(audio)
            # Mix to mono if stereo
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav, file_sr

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if isinstance(audio, torch.Tensor):
            if input_sr is None:
                raise ValueError("input_sr is required when passing a tensor or array.")
            return audio.float(), input_sr

        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _apply_bandwidth_limit(
        self,
        wav: torch.Tensor,
        effective_sr: int,
        target_sr: int,
    ) -> torch.Tensor:
        """
        Simulate low-resolution input from a high-sample-rate waveform.

        Applies the same downsample-then-upsample pipeline used during training
        (see WaveformCollator._apply_lpf) so that the spectral cutoff pattern
        matches what the model expects.

        Args:
            wav: Waveform at target_sr. Shape: (1, T) or (T,).
            effective_sr: The effective bandwidth in Hz (e.g. 8000).
            target_sr: The native sample rate of wav (e.g. 48000).

        Returns:
            Bandwidth-limited waveform at target_sr, same length as input.
        """
        original_len = wav.shape[-1]
        lr = torchaudio.functional.resample(
            wav, orig_freq=target_sr, new_freq=effective_sr
        )
        lr = torchaudio.functional.resample(
            lr, orig_freq=effective_sr, new_freq=target_sr
        )
        return lr[..., :original_len]

    def _preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to amplitude-compressed complex STFT representation.
        [B, C, T] -> [B, 2, F-1, T_frames]  (real/imag channels, drop Nyquist bin)
        """
        spec = self.transform(waveform)  # [B, C, F, T_frames] complex
        real = torch.view_as_real(spec.squeeze(1))  # [B, F, T_frames, 2]
        real = real.permute(0, 3, 1, 2)  # [B, 2, F, T_frames]
        return real[:, :, :-1, :]  # drop Nyquist bin

    def _postprocess(self, spec: torch.Tensor, orig_length: int) -> torch.Tensor:
        """
        Convert STFT representation back to waveform.
        [B, 2, F-1, T_frames] -> [B, T]
        """
        spec = torch.nn.functional.pad(spec, [0, 0, 0, 1], value=0)  # restore Nyquist
        spec = spec.permute(0, 2, 3, 1).contiguous()  # [B, F, T, 2]
        spec = torch.view_as_complex(spec)  # [B, F, T] complex
        waveform = self.transform.invert(spec, orig_length=orig_length)  # [B, T]
        return waveform

    def _inference(
        self,
        lr_audio: torch.Tensor,
        sr_khz: int,
        ode_method: str,
        ode_steps: int,
        guidance_scale: Optional[float],
        reconstruction_method: ReconstructionMethod = "original",
        reconstruction_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Core inference pipeline:
        1. STFT the (resampled) LR audio
        2. Extract LR condition bins
        3. Sample noise for HF region
        4. Solve ODE (flow matching)
        5. Concatenate LR + generated HF
        6. iSTFT to waveform
        """
        debug_logs_enabled = self._debug_logs_enabled()
        stage_timings: dict[str, float] = {}
        if debug_logs_enabled:
            logger.info(
                "[UniverSR] Inference start: input_shape=%s sr_khz=%s ode_method=%s "
                "ode_steps=%s guidance_scale=%s device=%s",
                tuple(lr_audio.shape),
                sr_khz,
                ode_method,
                ode_steps,
                guidance_scale,
                self._device,
            )
            self.model.reset_forward_profile()
            total_start = self._start_timing()

        # Frequency bin bookkeeping
        lr_bin_count = self.model.sr_to_lr_bins[sr_khz]
        hf_start_bin = self.model.total_freq_bins - self.model.hr_freq_bins
        orig_length = lr_audio.shape[-1]

        # STFT
        if debug_logs_enabled:
            preprocess_start = self._start_timing()
        Y = self._preprocess(lr_audio)  # [B, 2, F-1, T]
        if debug_logs_enabled:
            stage_timings["preprocess"] = self._finish_timing(preprocess_start)
            logger.info(
                "[UniverSR] Preprocess complete: spec_shape=%s lr_bin_count=%s "
                "hf_start_bin=%s reconstruction_method=%s",
                tuple(Y.shape),
                lr_bin_count,
                hf_start_bin,
                reconstruction_method,
            )
        Y_lr = Y[:, :, :lr_bin_count, :]  # LR condition
        Y_hr = Y[:, :, hf_start_bin:, :]  # HR target region (for shape reference)

        # Initial noise
        if debug_logs_enabled:
            sample_source_start = self._start_timing()
        x0 = self.path.sample_source(Y_hr).to(self._device)
        if debug_logs_enabled:
            stage_timings["sample_source"] = self._finish_timing(sample_source_start)
            logger.info(
                "[UniverSR] Source sampling complete: y_lr_shape=%s y_hr_shape=%s x0_shape=%s",
                tuple(Y_lr.shape),
                tuple(Y_hr.shape),
                tuple(x0.shape),
            )

        # Build ODE solver
        if guidance_scale is not None and guidance_scale > 0 and guidance_scale != 1.0:
            ode = CFGVectorFieldODE(net=self.model, guidance_scale=guidance_scale)
        else:
            ode = VectorFieldODE(net=self.model)
        solver = TorchDiffeqSolver(ode, method=ode_method)

        # Time discretization
        ts = torch.linspace(0, 1, ode_steps + 1, device=self._device)

        # Solve ODE
        if debug_logs_enabled:
            ode_solve_start = self._start_timing()
        x1_spec = solver.simulate(
            x0, ts=ts, y=Y_lr, sr_values=torch.tensor([sr_khz], device=self._device)
        )
        if debug_logs_enabled:
            stage_timings["ode_solve"] = self._finish_timing(ode_solve_start)
            logger.info(
                "[UniverSR] ODE solve complete: ts_steps=%s x1_spec_shape=%s",
                len(ts) - 1,
                tuple(x1_spec.shape),
            )

        # Concatenate LR bins + generated HF bins (handle overlapping region)
        if debug_logs_enabled:
            assemble_start = self._start_timing()
        slice_start = max(0, lr_bin_count - hf_start_bin)
        x1_spec = x1_spec[:, :, slice_start:, :]
        if reconstruction_method == "original":
            full_spec = torch.cat([Y_lr, x1_spec], dim=2)
        elif reconstruction_method == "original_signal":
            if reconstruction_audio is None:
                raise ValueError(
                    "reconstruction_audio is required when "
                    "reconstruction_method='original_signal'."
                )
            Y_reconstruction = self._preprocess(reconstruction_audio.to(self._device))
            full_spec = torch.cat(
                [Y_reconstruction[:, :, :lr_bin_count, :], x1_spec],
                dim=2,
            )
        else:
            raise ValueError(
                f"Unsupported reconstruction_method={reconstruction_method!r}. "
                f"Supported methods: {sorted(RECONSTRUCTION_METHODS)}"
            )
        if debug_logs_enabled:
            stage_timings["assemble"] = self._finish_timing(assemble_start)
            logger.info(
                "[UniverSR] Spectrum assembly complete: slice_start=%s full_spec_shape=%s",
                slice_start,
                tuple(full_spec.shape),
            )

        # iSTFT
        if debug_logs_enabled:
            postprocess_start = self._start_timing()
        output = self._postprocess(full_spec, orig_length=orig_length)
        if debug_logs_enabled:
            stage_timings["postprocess"] = self._finish_timing(postprocess_start)
            stage_timings["total"] = self._finish_timing(total_start)
            logger.info(
                "[UniverSR] Postprocess complete: output_shape=%s orig_length=%s",
                tuple(output.shape),
                orig_length,
            )
            logger.info(
                "[UniverSR] Stage timings: preprocess=%.3fs sample_source=%.3fs "
                "ode_solve=%.3fs assemble=%.3fs postprocess=%.3fs total=%.3fs",
                stage_timings.get("preprocess", 0.0),
                stage_timings.get("sample_source", 0.0),
                stage_timings.get("ode_solve", 0.0),
                stage_timings.get("assemble", 0.0),
                stage_timings.get("postprocess", 0.0),
                stage_timings.get("total", 0.0),
            )

            forward_profile = self.model.consume_forward_profile()
            if forward_profile["calls"] > 0:
                forward_total_s = float(sum(forward_profile["totals"].values()))
                logger.info(
                    "[UniverSR] Forward summary: calls=%s total=%.3fs avg=%.3fs",
                    forward_profile["calls"],
                    forward_total_s,
                    forward_total_s / max(1, forward_profile["calls"]),
                )
                top_stages = sorted(
                    forward_profile["totals"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:8]
                if top_stages:
                    stage_parts = ", ".join(
                        (
                            f"{name}={elapsed_s:.3f}s "
                            f"({(elapsed_s / max(forward_total_s, 1e-9)) * 100.0:.1f}%)"
                        )
                        for name, elapsed_s in top_stages
                    )
                    logger.info("[UniverSR] Top forward stages: %s", stage_parts)
                    logger.info(
                        "[UniverSR] Bottleneck stage: %s at %.3fs",
                        top_stages[0][0],
                        top_stages[0][1],
                    )
        return output
