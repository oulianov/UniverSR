import csv
import os
import resource
import shutil
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import wandb
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from universr.flow.loss import flow_matching_loss
from universr.flow.solver import CFGVectorFieldODE, TorchDiffeqSolver, VectorFieldODE
from universr.utils.utils import draw_spec, t2n

try:
    from torch_peaq import PEAQBasic
    PEAQ_AVAILABLE = True
except ImportError:
    print("Warning: torch_peaq not available. 2f-model metrics will be disabled.")
    PEAQ_AVAILABLE = False


MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    """Returns model size in bytes."""
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def memory_row(device: torch.device) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "mps_allocated_mb": "",
        "mps_driver_mb": "",
        "mps_recommended_max_mb": "",
        "cuda_allocated_mb": "",
        "cuda_reserved_mb": "",
    }
    if device.type == "mps" and torch.backends.mps.is_available():
        row["mps_allocated_mb"] = torch.mps.current_allocated_memory() / MiB
        row["mps_driver_mb"] = torch.mps.driver_allocated_memory() / MiB
        if hasattr(torch.mps, "recommended_max_memory"):
            row["mps_recommended_max_mb"] = torch.mps.recommended_max_memory() / MiB
    elif device.type == "cuda" and torch.cuda.is_available():
        row["cuda_allocated_mb"] = torch.cuda.memory_allocated(device) / MiB
        row["cuda_reserved_mb"] = torch.cuda.memory_reserved(device) / MiB
    return row


def max_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / MiB if os.uname().sysname == "Darwin" else max_rss / 1024


def append_csv_row(path: str | os.PathLike, row: dict[str, int | float | str | bool]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists() and output_path.stat().st_size > 0
    fieldnames = list(row.keys())
    if exists:
        with output_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            if existing_fieldnames and existing_fieldnames != fieldnames:
                existing_rows = list(reader)
                fieldnames = existing_fieldnames + [field for field in fieldnames if field not in existing_fieldnames]
                tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
                with tmp_path.open("w", newline="") as tmp_handle:
                    writer = csv.DictWriter(tmp_handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                    writer.writerow(row)
                    tmp_handle.flush()
                    os.fsync(tmp_handle.fileno())
                tmp_path.replace(output_path)
                return

    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


class CSVTrainingMonitor:
    fieldnames = [
        "time",
        "phase",
        "epoch",
        "step",
        "epoch_step",
        "loss",
        "cfm_loss",
        "lsd_total",
        "lsd_high",
        "lsd_low",
        "data_seconds",
        "compute_seconds",
        "samples_per_second",
        "lr",
        "mps_allocated_mb",
        "mps_driver_mb",
        "mps_recommended_max_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "max_rss_mb",
    ]

    def __init__(self, path: str | os.PathLike, device: torch.device) -> None:
        self.path = Path(path)
        self.device = device
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self.handle = self.path.open("a", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if not exists:
            self.writer.writeheader()

    def close(self) -> None:
        self.handle.close()

    def log(
        self,
        phase: str,
        epoch: int,
        step: int,
        epoch_step: int,
        loss: float,
        data_seconds: float,
        compute_seconds: float,
        batch_size: int,
        lr: float,
        extra: dict[str, float | str] | None = None,
    ) -> None:
        row = {field: "" for field in self.fieldnames}
        row.update(
            {
                "time": time.time(),
                "phase": phase,
                "epoch": epoch,
                "step": step,
                "epoch_step": epoch_step,
                "loss": loss,
                "data_seconds": data_seconds,
                "compute_seconds": compute_seconds,
                "samples_per_second": batch_size / compute_seconds if compute_seconds > 0 else 0,
                "lr": lr,
                "max_rss_mb": max_rss_mb(),
            }
        )
        row.update(memory_row(self.device))
        if extra:
            row.update(extra)
        self.writer.writerow(row)
        self.handle.flush()


class CSVValidationTrackMonitor:
    fieldnames = [
        "time",
        "epoch",
        "step",
        "validation_index",
        "sample_index",
        "folder",
        "channel",
        "filename",
        "relpath",
        "source_relpath",
        "path",
        "original_audio_path",
        "degraded_audio_path",
        "restored_audio_path",
        "stereo_original_audio_path",
        "stereo_degraded_audio_path",
        "stereo_restored_audio_path",
        "low_sr",
        "loss",
        "lsd_total",
        "lsd_high",
        "lsd_low",
        "data_seconds",
        "compute_seconds",
        "val_max_sec",
        "ode_steps",
        "lr",
        "mps_allocated_mb",
        "mps_driver_mb",
        "mps_recommended_max_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "max_rss_mb",
    ]

    def __init__(self, path: str | os.PathLike, device: torch.device) -> None:
        self.path = Path(path)
        self.device = device
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self.handle = self.path.open("a", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        if not exists:
            self.writer.writeheader()

    def close(self) -> None:
        self.handle.close()

    @staticmethod
    def _metadata_from_relpath(relpath: str) -> tuple[str, str, str]:
        parts = Path(relpath).parts
        channel = ""
        if parts and parts[0] in {"mid", "side"}:
            channel = parts[0]
            parts = parts[1:]
        folder = parts[0] if len(parts) > 1 else "."
        source_relpath = str(Path(*parts)) if parts else relpath
        return folder, channel, source_relpath

    @staticmethod
    def _as_list(value, batch_size: int) -> list:
        if isinstance(value, list):
            return value
        return [value] * batch_size

    @staticmethod
    def _value_at(value, sample_index: int):
        if isinstance(value, list):
            return value[sample_index]
        return value

    def log(
        self,
        epoch: int,
        step: int,
        validation_index: int,
        batch_data: dict,
        loss: float,
        lsd_total: float | str,
        lsd_high: float | str,
        lsd_low: float | str,
        data_seconds: float,
        compute_seconds: float,
        lr: float,
        val_max_sec: int,
        ode_steps: int,
        audio_paths: list[dict[str, str]] | None = None,
    ) -> None:
        batch_size = int(batch_data["hr"].shape[0]) if "hr" in batch_data else 1
        filenames = self._as_list(batch_data.get("filename", ""), batch_size)
        relpaths = self._as_list(batch_data.get("relpath", ""), batch_size)
        paths = self._as_list(batch_data.get("path", ""), batch_size)
        low_srs = self._as_list(batch_data.get("low_sr", ""), batch_size)
        audio_paths = audio_paths or [{} for _ in range(batch_size)]
        memory = memory_row(self.device)
        rss_mb = max_rss_mb()

        for sample_index in range(batch_size):
            sample_audio_paths = audio_paths[sample_index] if sample_index < len(audio_paths) else {}
            relpath = str(relpaths[sample_index])
            folder, channel, source_relpath = self._metadata_from_relpath(relpath)
            row = {
                "time": time.time(),
                "epoch": epoch,
                "step": step,
                "validation_index": validation_index,
                "sample_index": sample_index,
                "folder": folder,
                "channel": channel,
                "filename": str(filenames[sample_index]),
                "relpath": relpath,
                "source_relpath": source_relpath,
                "path": str(paths[sample_index]),
                "original_audio_path": sample_audio_paths.get("original", ""),
                "degraded_audio_path": sample_audio_paths.get("degraded", ""),
                "restored_audio_path": sample_audio_paths.get("restored", ""),
                "stereo_original_audio_path": sample_audio_paths.get("stereo_original", ""),
                "stereo_degraded_audio_path": sample_audio_paths.get("stereo_degraded", ""),
                "stereo_restored_audio_path": sample_audio_paths.get("stereo_restored", ""),
                "low_sr": low_srs[sample_index],
                "loss": self._value_at(loss, sample_index),
                "lsd_total": self._value_at(lsd_total, sample_index),
                "lsd_high": self._value_at(lsd_high, sample_index),
                "lsd_low": self._value_at(lsd_low, sample_index),
                "data_seconds": data_seconds,
                "compute_seconds": compute_seconds,
                "val_max_sec": val_max_sec,
                "ode_steps": ode_steps,
                "lr": lr,
                **memory,
                "max_rss_mb": rss_mb,
            }
            self.writer.writerow(row)
        self.handle.flush()


class Trainer(ABC):
    """Abstract base class for training."""

    def __init__(self, model, train_loader, val_loader, device, logger):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.optimizer = None
        self.scheduler = None
        self.logger = logger
        self.checkpoint_global_step = None
        self.checkpoint_epoch_step = None

    @abstractmethod
    def _train_step(self, **kwargs) -> torch.Tensor:
        pass

    def _val_step(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), **config)

    def get_scheduler(self, optimizer, config):
        scheduler_type = config.get('type', 'CosineLR')
        scheduler_args = config.get('init_args', {})
        if scheduler_type == 'ExponentialLR':
            return lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_type == 'CosineLR':
            return get_cosine_schedule_with_warmup(optimizer=optimizer, **scheduler_args)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch, is_best, save_dir, filename=None, global_step=None, epoch_step=None):
        os.makedirs(save_dir, exist_ok=True)
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        if global_step is not None:
            state['global_step'] = int(global_step)
        if epoch_step is not None:
            state['epoch_step'] = int(epoch_step)
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        ckpt_path = os.path.join(save_dir, filename or 'recent.pth')
        torch.save(state, ckpt_path)
        print(f"Checkpoint saved at epoch {epoch}: {ckpt_path}")

        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            shutil.copyfile(ckpt_path, best_path)
            print(f"Best model updated at epoch {epoch} (loss={self.best_loss:.6f})")

    def load_checkpoint(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
            return
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be initialized before loading a checkpoint.")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
            self.checkpoint_global_step = checkpoint.get('global_step')
            self.checkpoint_epoch_step = checkpoint.get('epoch_step')
            self.start_epoch = checkpoint['epoch'] if self.checkpoint_epoch_step is not None else checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            resume_msg = f"Training checkpoint loaded from {ckpt_path}. Resuming from epoch {self.start_epoch}"
            if self.checkpoint_global_step is not None:
                resume_msg += f", global step {self.checkpoint_global_step}"
            if self.checkpoint_epoch_step is not None:
                resume_msg += f", epoch step {self.checkpoint_epoch_step}"
            print(resume_msg + ".")
        else:
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            print(f"Model weights loaded from {ckpt_path}. Starting training from epoch 1.")

    @staticmethod
    def load_model_for_inference(model, ckpt_path, device='cuda'):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"Model loaded from {ckpt_path}")
        return model

    # ------------------------------------------------------------------ #
    #  Validation / Train loops
    # ------------------------------------------------------------------ #
    def validate(self, global_step, val_idx, ode_steps=4, val_max_sec=5, val_max_batches=None,
                 epoch_idx=0, monitor=None, track_monitor=None, lr=0.0,
                 upscale_validation=False, validation_loss_metric="loss",
                 upscale_chunk_sec=None, val_audio_output_dir=None,
                 val_audio_reference_once=False, val_guidance_scale=None,
                 val_reconstruction_method="original"):
        self.model.eval()
        total_val_loss = 0.0
        total_lsd_total = 0.0
        total_lsd_high = 0.0
        total_lsd_low = 0.0
        total_samples = 0
        total_data_seconds = 0.0
        total_compute_seconds = 0.0
        cnt = 0
        num_batches = len(self.val_loader)
        if val_max_batches is not None:
            num_batches = min(num_batches, int(val_max_batches))
        val_pbar = tqdm(range(num_batches), desc='Validating...', dynamic_ncols=True)
        val_iter = iter(self.val_loader)
        data_start = time.perf_counter()
        wall_start = time.perf_counter()

        with torch.no_grad():
            for idx in val_pbar:
                batch = next(val_iter)
                data_seconds = time.perf_counter() - data_start
                compute_start = time.perf_counter()
                outdict = self._val_step(batch, idx, val_idx,
                                         ode_steps=ode_steps,
                                         val_max_sec=val_max_sec,
                                         upscale_validation=upscale_validation,
                                         validation_loss_metric=validation_loss_metric,
                                         upscale_chunk_sec=upscale_chunk_sec,
                                         global_step=global_step,
                                         val_audio_output_dir=val_audio_output_dir,
                                         val_audio_reference_once=val_audio_reference_once,
                                         guidance_scale=val_guidance_scale,
                                         reconstruction_method=val_reconstruction_method)
                loss = outdict['loss']
                loss_value = float(loss.detach().cpu())
                batch_samples = int(batch['hr'].shape[0]) if isinstance(batch, dict) and 'hr' in batch else 1
                loss_values = outdict.get('loss_per_sample')
                total_val_loss += sum(loss_values) if loss_values else loss_value * batch_samples

                if outdict['lsd_high'] is not None:
                    lsd_total_values = outdict.get('lsd_total_per_sample')
                    lsd_high_values = outdict.get('lsd_high_per_sample')
                    lsd_low_values = outdict.get('lsd_low_per_sample')
                    if lsd_high_values:
                        total_lsd_high += sum(lsd_high_values)
                        total_lsd_total += sum(lsd_total_values)
                        total_lsd_low += sum(lsd_low_values)
                        cnt += len(lsd_high_values)
                    else:
                        total_lsd_high += outdict['lsd_high'] * batch_samples
                        total_lsd_total += (outdict.get('lsd_total') or 0.0) * batch_samples
                        total_lsd_low += (outdict.get('lsd_low') or 0.0) * batch_samples
                        cnt += batch_samples

                synchronize_if_needed(self.device)
                compute_seconds = time.perf_counter() - compute_start
                total_samples += batch_samples
                total_data_seconds += data_seconds
                total_compute_seconds += compute_seconds

                if monitor is not None:
                    monitor.log(
                        "valid",
                        epoch_idx,
                        global_step,
                        idx + 1,
                        loss_value,
                        data_seconds,
                        compute_seconds,
                        batch_samples,
                        lr,
                        {
                            "lsd_total": outdict.get('lsd_total') if outdict.get('lsd_total') is not None else "",
                            "lsd_high": outdict['lsd_high'] if outdict['lsd_high'] is not None else "",
                            "lsd_low": outdict.get('lsd_low') if outdict.get('lsd_low') is not None else "",
                        },
                    )
                if track_monitor is not None:
                    track_monitor.log(
                        epoch_idx,
                        global_step,
                        idx + 1,
                        batch,
                        outdict.get('loss_per_sample') or loss_value,
                        outdict.get('lsd_total_per_sample') or (
                            outdict.get('lsd_total') if outdict.get('lsd_total') is not None else ""
                        ),
                        outdict.get('lsd_high_per_sample') or (
                            outdict['lsd_high'] if outdict['lsd_high'] is not None else ""
                        ),
                        outdict.get('lsd_low_per_sample') or (
                            outdict.get('lsd_low') if outdict.get('lsd_low') is not None else ""
                        ),
                        data_seconds,
                        compute_seconds,
                        lr,
                        val_max_sec,
                        ode_steps,
                        audio_paths=outdict.get('audio_paths'),
                    )

                val_pbar.set_postfix({'val_loss': f'{loss_value:.6f}'})

                if outdict['log_payload']:
                    self.logger.log(outdict['log_payload'], step=global_step)

                data_start = time.perf_counter()

        avg_val_loss = total_val_loss / max(1, total_samples)
        avg_lsd_total = total_lsd_total / cnt if cnt > 0 else 0
        avg_lsd_high = total_lsd_high / cnt if cnt > 0 else 0
        avg_lsd_low = total_lsd_low / cnt if cnt > 0 else 0
        self.logger.log({
            "val/loss": avg_val_loss,
            "val/lsd_total": avg_lsd_total,
            "val/lsd_high": avg_lsd_high,
            "val/lsd_low": avg_lsd_low,
        }, step=global_step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()
        wall_seconds = time.perf_counter() - wall_start
        return {
            "loss": avg_val_loss,
            "lsd_total": avg_lsd_total,
            "lsd_high": avg_lsd_high,
            "lsd_low": avg_lsd_low,
            "wall_seconds": wall_seconds,
            "samples": total_samples,
            "samples_per_second": total_samples / wall_seconds if wall_seconds > 0 else 0.0,
            "compute_samples_per_second": total_samples / total_compute_seconds if total_compute_seconds > 0 else 0.0,
            "mean_data_seconds": total_data_seconds / max(1, num_batches),
            "mean_compute_seconds": total_compute_seconds / max(1, num_batches),
        }

    def train(self,
              num_epochs,
              max_steps=500000,
              optimizer_config=None,
              scheduler_config=None,
              ckpt_save_dir='ckpts',
              ckpt_load_path=None,
              log_step_interval=100,
              val_step_interval=5000,
              num_val_log_samples=10,
              val_ode_steps=4,
              val_max_sec=5,
              val_max_batches=None,
              train_metrics_interval=1,
              zero_grad_set_to_none=False,
              channels_last=False,
              metrics_csv_path=None,
              epoch_summary_csv_path=None,
              val_track_metrics_csv_path=None,
              upscale_validation=False,
              validation_loss_metric="loss",
              upscale_chunk_sec=None,
              val_audio_output_dir=None,
              val_audio_reference_once=False,
              val_guidance_scale=None,
              val_reconstruction_method="original",
              resume_global_step=None,
              resume_epoch=None,
              resume_epoch_step=None,
              **kwargs):
        self.log_step_interval = log_step_interval
        total_val_batches = len(self.val_loader)
        val_idx = set(torch.linspace(0, total_val_batches - 1, num_val_log_samples).long().tolist())
        monitor = CSVTrainingMonitor(metrics_csv_path, self.device) if metrics_csv_path else None
        track_monitor = CSVValidationTrackMonitor(val_track_metrics_csv_path, self.device) if val_track_metrics_csv_path else None
        if monitor is not None:
            print(f"CSV step metrics: {Path(metrics_csv_path).resolve()}")
        if track_monitor is not None:
            print(f"CSV validation track metrics: {Path(val_track_metrics_csv_path).resolve()}")
        if epoch_summary_csv_path:
            print(f"CSV summaries: {Path(epoch_summary_csv_path).resolve()}")
        if val_audio_output_dir:
            print(f"Validation audio: {Path(val_audio_output_dir).resolve()}")

        try:
            print(f'Training model with size: {model_size_b(self.model) / MiB:.3f} MiB')
            if channels_last and self.device.type in ("cuda", "mps"):
                self.model.to(self.device, memory_format=torch.channels_last)
            else:
                self.model.to(self.device)
            self.optimizer = self.get_optimizer(optimizer_config)
            if scheduler_config:
                self.scheduler = self.get_scheduler(self.optimizer, scheduler_config)
            if ckpt_load_path:
                self.load_checkpoint(ckpt_load_path)

            if resume_epoch is not None:
                self.start_epoch = int(resume_epoch)
            if resume_global_step is not None:
                global_step = int(resume_global_step)
            elif self.checkpoint_global_step is not None:
                global_step = int(self.checkpoint_global_step)
            else:
                global_step = (self.start_epoch - 1) * len(self.train_loader)
            if resume_epoch_step is not None:
                start_epoch_step = int(resume_epoch_step)
            elif self.checkpoint_epoch_step is not None:
                start_epoch_step = int(self.checkpoint_epoch_step)
            else:
                start_epoch_step = 0
            self.model.train()
            print(f"--- Starting training from epoch {self.start_epoch}, global step {global_step} ---")

            for epoch_idx in range(self.start_epoch, num_epochs + 1):
                resume_batches = start_epoch_step if epoch_idx == self.start_epoch else 0
                epoch_pbar = tqdm(range(resume_batches, len(self.train_loader)),
                                  total=len(self.train_loader),
                                  initial=resume_batches,
                                  desc=f'Epoch {epoch_idx}/{num_epochs}',
                                  dynamic_ncols=True, leave=True)
                train_iter = iter(self.train_loader)
                for _ in range(resume_batches):
                    next(train_iter)
                total_epoch_loss = 0.0
                processed_batches = 0
                total_samples = 0
                total_data_seconds = 0.0
                total_compute_seconds = 0.0
                epoch_start = time.perf_counter()
                data_start = time.perf_counter()

                for batch_idx in epoch_pbar:
                    batch = next(train_iter)
                    data_seconds = time.perf_counter() - data_start
                    compute_start = time.perf_counter()
                    global_step += 1

                    self.optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
                    loss, loss_dict = self._train_step(batch, global_step)
                    loss.backward()
                    self.optimizer.step()
                    if scheduler_config:
                        self.scheduler.step()

                    synchronize_if_needed(self.device)
                    compute_seconds = time.perf_counter() - compute_start
                    loss_value = float(loss.detach().cpu())
                    batch_samples = int(batch['hr'].shape[0]) if isinstance(batch, dict) and 'hr' in batch else 1
                    total_epoch_loss += loss_value
                    processed_batches += 1
                    total_samples += batch_samples
                    total_data_seconds += data_seconds
                    total_compute_seconds += compute_seconds
                    epoch_pbar.set_postfix({'loss': f'{loss_value:.6f}'})

                    lr = self.optimizer.param_groups[0]['lr']
                    if monitor is not None and (
                            train_metrics_interval <= 1
                            or global_step == 1
                            or global_step % train_metrics_interval == 0):
                        monitor.log(
                            "train",
                            epoch_idx,
                            global_step,
                            batch_idx + 1,
                            loss_value,
                            data_seconds,
                            compute_seconds,
                            batch_samples,
                            lr,
                            {key: float(value.detach().cpu()) for key, value in loss_dict.items()},
                        )

                    # Step logging
                    if global_step % log_step_interval == 0:
                        self.logger.log({"model/loss": loss_value}, step=global_step)
                        self.logger.log({"charts/lr-adam": lr}, step=global_step)
                        self.logger.log({f"model/{k}": v.item() for k, v in loss_dict.items()}, step=global_step)

                    # Validation
                    if global_step % val_step_interval == 0:
                        val_results = self.validate(
                            global_step,
                            val_idx,
                            ode_steps=val_ode_steps,
                            val_max_sec=val_max_sec,
                            val_max_batches=val_max_batches,
                            epoch_idx=epoch_idx,
                            monitor=monitor,
                            track_monitor=track_monitor,
                            lr=lr,
                            upscale_validation=upscale_validation,
                            validation_loss_metric=validation_loss_metric,
                            upscale_chunk_sec=upscale_chunk_sec,
                            val_audio_output_dir=val_audio_output_dir,
                            val_audio_reference_once=val_audio_reference_once,
                            val_guidance_scale=val_guidance_scale,
                            val_reconstruction_method=val_reconstruction_method,
                        )
                        avg_val_loss = val_results['loss']
                        print(f'\nStep {global_step} | Val Loss: {avg_val_loss:.6f}, '
                              f'Val LSD-high: {val_results["lsd_high"]:.4f}\n')

                        is_best = avg_val_loss < self.best_loss
                        if is_best:
                            self.best_loss = avg_val_loss
                        if epoch_summary_csv_path:
                            append_csv_row(
                                epoch_summary_csv_path,
                                {
                                    "time": time.time(),
                                    "phase": "validation",
                                    "epoch": epoch_idx,
                                    "step": global_step,
                                    "train_loss_running": total_epoch_loss / max(1, processed_batches),
                                    "valid_loss": avg_val_loss,
                                    "valid_lsd_total": val_results["lsd_total"],
                                    "valid_lsd_high": val_results["lsd_high"],
                                    "valid_lsd_low": val_results["lsd_low"],
                                    "valid_wall_seconds": val_results["wall_seconds"],
                                    "valid_samples": val_results["samples"],
                                    "valid_samples_per_second": val_results["samples_per_second"],
                                    "valid_compute_samples_per_second": val_results["compute_samples_per_second"],
                                    "valid_mean_data_seconds": val_results["mean_data_seconds"],
                                    "valid_mean_compute_seconds": val_results["mean_compute_seconds"],
                                    "is_best": is_best,
                                    "best_loss": self.best_loss,
                                    "lr": lr,
                                    **memory_row(self.device),
                                    "max_rss_mb": max_rss_mb(),
                                },
                            )
                        self.save_checkpoint(
                            epoch=epoch_idx,
                            is_best=is_best,
                            save_dir=ckpt_save_dir,
                            global_step=global_step,
                            epoch_step=batch_idx + 1,
                        )

                    if global_step >= max_steps:
                        print(f'\nReached max_steps ({max_steps}). Finishing training.')
                        self.save_checkpoint(epoch=epoch_idx, is_best=False, save_dir=ckpt_save_dir,
                                             filename=f'step_{global_step}.pth',
                                             global_step=global_step,
                                             epoch_step=batch_idx + 1)
                        return

                    data_start = time.perf_counter()

                avg_epoch_loss = total_epoch_loss / max(1, processed_batches)
                epoch_wall_seconds = time.perf_counter() - epoch_start
                print(f'Epoch {epoch_idx} completed. Average Loss: {avg_epoch_loss:.6f}')
                self.logger.log({"model/epoch_loss": avg_epoch_loss, "charts/epoch": epoch_idx}, step=global_step)
                if epoch_summary_csv_path:
                    append_csv_row(
                        epoch_summary_csv_path,
                        {
                            "time": time.time(),
                            "phase": "epoch",
                            "epoch": epoch_idx,
                            "step": global_step,
                            "train_loss": avg_epoch_loss,
                            "train_wall_seconds": epoch_wall_seconds,
                            "train_samples": total_samples,
                            "train_samples_per_second": total_samples / epoch_wall_seconds if epoch_wall_seconds > 0 else 0.0,
                            "train_compute_samples_per_second": total_samples / total_compute_seconds if total_compute_seconds > 0 else 0.0,
                            "train_mean_data_seconds": total_data_seconds / max(1, len(self.train_loader)),
                            "train_mean_compute_seconds": total_compute_seconds / max(1, len(self.train_loader)),
                            "lr": self.optimizer.param_groups[0]['lr'],
                            **memory_row(self.device),
                            "max_rss_mb": max_rss_mb(),
                        },
                    )

            self.model.eval()
            print('Training finished!')
        finally:
            if monitor is not None:
                monitor.close()
            if track_monitor is not None:
                track_monitor.close()


# ====================================================================== #
#  STFTTrainer
# ====================================================================== #

class STFTTrainer(Trainer):
    """Flow-matching trainer operating in the STFT domain."""

    def __init__(self, model, path, transform, lowpass_on_device=False, channels_last=False, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.transform = transform
        self.lowpass_on_device = lowpass_on_device
        self.channels_last = channels_last

    # ------------------------------------------------------------------ #
    #  Spectral pre/post-processing
    # ------------------------------------------------------------------ #
    def _preprocess(self, waveform):
        """waveform [B,C,T] -> real-valued STFT [B,2,F,T]"""
        spec = self.transform(waveform)
        real = torch.view_as_real(spec.squeeze(1))
        real = real.permute(0, 3, 1, 2)
        return real[:, :, :-1, :]

    def _make_lr_wave(self, hr_wave, sr_values):
        current_sr = sr_values[0].item() if hasattr(sr_values[0], 'item') else sr_values[0]
        target_sr = getattr(getattr(self.transform, "complex_stft", None), "sampling_rate", 48000)
        original_len = hr_wave.shape[-1]
        low_sr = int(current_sr) * 1000

        lr_wave = torchaudio.functional.resample(hr_wave, orig_freq=target_sr, new_freq=low_sr)
        lr_wave = torchaudio.functional.resample(lr_wave, orig_freq=low_sr, new_freq=target_sr)
        if lr_wave.shape[-1] < original_len:
            lr_wave = torch.nn.functional.pad(lr_wave, (0, original_len - lr_wave.shape[-1]))
        return lr_wave[..., :original_len]

    def _as_model_input(self, tensor):
        if self.channels_last and tensor.ndim == 4 and tensor.device.type in ("cuda", "mps"):
            return tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _postprocess(self, spec, orig_length):
        """real-valued STFT [B,2,F,T] -> waveform [B,T]"""
        spec = torch.nn.functional.pad(spec, pad=[0, 0, 0, 1], value=0)
        spec = spec.permute(0, 2, 3, 1).contiguous()
        spec = torch.view_as_complex(spec)
        return self.transform.invert(spec, orig_length=orig_length)

    def _get_freq_bins(self, sr_value):
        """Return (lr_bin_count, hf_start_bin) for a given sampling rate."""
        lr_bin_count = self.model.sr_to_lr_bins[sr_value]
        hf_start_bin = self.model.total_freq_bins - self.model.hr_freq_bins
        return lr_bin_count, hf_start_bin

    def _split_spectrum(self, Y, Z, lr_bin_count, hf_start_bin):
        """Split full spectra into LR condition and HR target regions."""
        Y_lr = Y[:, :, :lr_bin_count, :]
        Y_hr = Y[:, :, hf_start_bin:, :]
        Z_hr = Z[:, :, hf_start_bin:, :]
        return Y_lr, Y_hr, Z_hr

    def _assemble_fullband(self, Y_lr, x1_hr, lr_bin_count, hf_start_bin, reconstruction_lr=None):
        """Concatenate LR condition with generated HR to form fullband spectrum."""
        slice_start = max(0, lr_bin_count - hf_start_bin)
        low_band = reconstruction_lr if reconstruction_lr is not None else Y_lr
        return torch.cat([low_band, x1_hr[:, :, slice_start:, :]], dim=2)

    # ------------------------------------------------------------------ #
    #  ODE inference
    # ------------------------------------------------------------------ #
    def _run_ode(self, Y_lr, Y_hr, sr_values, ode_steps, guidance_scale=None):
        """Run ODE sampling in the HR spectral region."""
        if guidance_scale is not None and guidance_scale != 0 and guidance_scale != 1.0:
            ode = CFGVectorFieldODE(net=self.model, guidance_scale=float(guidance_scale))
        else:
            ode = VectorFieldODE(net=self.model)
        solver = TorchDiffeqSolver(ode, method='midpoint')
        ts = torch.linspace(0, 1, int(ode_steps) + 1, device=self.device)
        x0 = self.path.sample_source(Y_hr)
        return solver.simulate(x0, ts, y=Y_lr, sr_values=sr_values)

    def _synthesize_waveform(self, Y_lr, Y_hr, sr_values, lr_bin_count, hf_start_bin,
                             ode_steps, guidance_scale=None, orig_length=None,
                             reconstruction_lr=None):
        """Full pipeline: ODE sampling -> spectral assembly -> waveform."""
        x1_hr = self._run_ode(Y_lr, Y_hr, sr_values, ode_steps, guidance_scale)
        x1_full = self._assemble_fullband(Y_lr, x1_hr, lr_bin_count, hf_start_bin, reconstruction_lr)
        return self._postprocess(x1_full, orig_length=orig_length)

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #
    def _stft_magnitude(self, audio, n_fft=1024, hop_length=256):
        """Compute STFT magnitude for metric calculations."""
        window = torch.hann_window(n_fft).to(audio.device)
        return torch.abs(torch.stft(audio, n_fft, hop_length, window=window, return_complex=True))

    def _compute_lsd(self, pred, target, sr_khz, reduce=True):
        """
        Compute Log-Spectral Distance with frequency band separation.
        Cutoff bin is derived from model config (sr_to_lr_bins).
        Returns: (lsd_total, lsd_high, lsd_low)
        """
        bin_idx = self.model.sr_to_lr_bins[sr_khz]

        sp = torch.log10(self._stft_magnitude(pred.squeeze(1)).square().clamp(min=1e-6))
        st = torch.log10(self._stft_magnitude(target.squeeze(1)).square().clamp(min=1e-6))

        def _lsd(a, b):
            values = (a - b).square().mean(dim=1).sqrt().mean(dim=-1)
            return values.mean() if reduce else values

        lsd_total = _lsd(sp, st)
        lsd_low = _lsd(sp[..., :bin_idx, :], st[..., :bin_idx, :])
        lsd_high = _lsd(sp[..., bin_idx:, :], st[..., bin_idx:, :])
        return lsd_total, lsd_high, lsd_low

    def _synthesize_upscale_chunks(self, y, z, sr_values, current_sr, ode_steps,
                                   chunk_sec=None, guidance_scale=None,
                                   reconstruction_method="original"):
        if reconstruction_method not in {"original", "original_signal"}:
            raise ValueError("reconstruction_method must be 'original' or 'original_signal'.")
        target_sr = getattr(getattr(self.transform, "complex_stft", None), "sampling_rate", 48000)
        chunk_samples = int(float(chunk_sec) * target_sr) if chunk_sec else z.shape[-1]
        if chunk_samples <= 0 or chunk_samples >= z.shape[-1]:
            lr_bin_count, hf_start_bin = self._get_freq_bins(current_sr)
            Y = self._preprocess(y)
            Z = self._preprocess(z)
            Y_lr, Y_hr, _ = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)
            reconstruction_lr = None
            if reconstruction_method == "original_signal":
                reconstruction_lr = Z[:, :, :lr_bin_count, :]
            return self._synthesize_waveform(
                self._as_model_input(Y_lr),
                self._as_model_input(Y_hr),
                sr_values,
                lr_bin_count,
                hf_start_bin,
                ode_steps,
                guidance_scale=guidance_scale,
                orig_length=z.shape[-1],
                reconstruction_lr=self._as_model_input(reconstruction_lr) if reconstruction_lr is not None else None,
            )

        chunks = []
        for start in range(0, z.shape[-1], chunk_samples):
            z_chunk = z[..., start:start + chunk_samples]
            y_chunk = y[..., start:start + z_chunk.shape[-1]]
            lr_bin_count, hf_start_bin = self._get_freq_bins(current_sr)
            Y = self._preprocess(y_chunk)
            Z = self._preprocess(z_chunk)
            Y_lr, Y_hr, _ = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)
            reconstruction_lr = None
            if reconstruction_method == "original_signal":
                reconstruction_lr = Z[:, :, :lr_bin_count, :]
            x1_chunk = self._synthesize_waveform(
                self._as_model_input(Y_lr),
                self._as_model_input(Y_hr),
                sr_values,
                lr_bin_count,
                hf_start_bin,
                ode_steps,
                guidance_scale=guidance_scale,
                orig_length=z_chunk.shape[-1],
                reconstruction_lr=self._as_model_input(reconstruction_lr) if reconstruction_lr is not None else None,
            )
            chunks.append(x1_chunk[..., :z_chunk.shape[-1]])
        return torch.cat(chunks, dim=-1)

    def _save_validation_audio_triplets(self, z, y, x1, batch_data, idx, global_step, output_dir,
                                        reference_once=False):
        if not output_dir:
            return None

        target_sr = getattr(getattr(self.transform, "complex_stft", None), "sampling_rate", 48000)
        batch_size = z.shape[0]
        relpaths = batch_data.get('relpath', [f"batch{idx:03d}_sample{i:02d}.wav" for i in range(batch_size)])
        if not isinstance(relpaths, list):
            relpaths = [relpaths] * batch_size

        base_dir = Path(output_dir)
        step_dir = base_dir / f"step_{int(global_step):08d}"
        reference_dir = base_dir / "reference"
        audio_paths = []

        def _sample_to_numpy(tensor, sample_index):
            sample = tensor[sample_index]
            if sample.dim() == 2 and sample.shape[0] == 1:
                sample = sample.squeeze(0)
            return t2n(sample)

        def _read_mono(path):
            audio, _ = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio[:, 0]
            return audio.astype(np.float32, copy=False)

        def _write_stereo_from_mid_side(mid_path, side_path, output_path):
            if not mid_path.exists() or not side_path.exists():
                return False
            mid = _read_mono(mid_path)
            side = _read_mono(side_path)
            length = min(mid.shape[-1], side.shape[-1])
            if length <= 0:
                return False
            stereo = np.stack([mid[:length] + side[:length], mid[:length] - side[:length]], axis=-1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, np.clip(stereo, -1.0, 1.0), target_sr)
            return True

        def _maybe_write_stereo(relpath, original_path, degraded_path, restored_path):
            parts = relpath.parts
            if not parts or parts[0] not in {"mid", "side"}:
                return {}
            channel = parts[0]
            other_channel = "side" if channel == "mid" else "mid"
            source_relpath = Path(*parts[1:])

            def _swap_channel(path):
                path_parts = list(path.parts)
                for index, part in enumerate(path_parts):
                    if part == channel:
                        path_parts[index] = other_channel
                        break
                return Path(*path_parts)

            stereo_ref_dir = reference_dir / "stereo" / source_relpath.parent
            stereo_step_dir = step_dir / "stereo" / source_relpath.parent
            stem = source_relpath.stem
            stereo_original = stereo_ref_dir / f"{stem}_original.wav"
            stereo_degraded = stereo_ref_dir / f"{stem}_degraded.wav"
            stereo_restored = stereo_step_dir / f"{stem}_restored.wav"

            original_other = _swap_channel(original_path)
            degraded_other = _swap_channel(degraded_path)
            restored_other = _swap_channel(restored_path)
            if not stereo_original.exists():
                _write_stereo_from_mid_side(
                    original_path if channel == "mid" else original_other,
                    original_other if channel == "mid" else original_path,
                    stereo_original,
                )
            if not stereo_degraded.exists():
                _write_stereo_from_mid_side(
                    degraded_path if channel == "mid" else degraded_other,
                    degraded_other if channel == "mid" else degraded_path,
                    stereo_degraded,
                )
            _write_stereo_from_mid_side(
                restored_path if channel == "mid" else restored_other,
                restored_other if channel == "mid" else restored_path,
                stereo_restored,
            )
            return {
                "stereo_original": str(stereo_original) if stereo_original.exists() else "",
                "stereo_degraded": str(stereo_degraded) if stereo_degraded.exists() else "",
                "stereo_restored": str(stereo_restored) if stereo_restored.exists() else "",
            }

        for sample_index in range(batch_size):
            relpath = Path(str(relpaths[sample_index]))
            out_dir = step_dir / relpath.parent
            ref_dir = reference_dir / relpath.parent if reference_once else out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            ref_dir.mkdir(parents=True, exist_ok=True)
            stem = relpath.stem
            original_path = ref_dir / f"{stem}_original.wav"
            degraded_path = ref_dir / f"{stem}_degraded.wav"
            restored_path = out_dir / f"{stem}_restored.wav"

            if not reference_once or not original_path.exists():
                sf.write(original_path, _sample_to_numpy(z, sample_index), target_sr)
            if not reference_once or not degraded_path.exists():
                sf.write(degraded_path, _sample_to_numpy(y, sample_index), target_sr)
            sf.write(restored_path, _sample_to_numpy(x1, sample_index), target_sr)
            paths = {
                "original": str(original_path),
                "degraded": str(degraded_path),
                "restored": str(restored_path),
            }
            paths.update(_maybe_write_stereo(relpath, original_path, degraded_path, restored_path))
            audio_paths.append(paths)
        return audio_paths

    def _calculate_2f_model_metric(self, z_c, x1_c):
        """Calculate 2f-model (PEAQ) metric for a batch of audio samples."""
        if not PEAQ_AVAILABLE:
            return None
        try:
            def _to_2d(x):
                if x.dim() == 3:
                    x = x.squeeze(1) if x.shape[1] == 1 else torch.mean(x, dim=1)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return torch.clamp(x, -0.99, 0.99)

            z_peaq = _to_2d(z_c.clone())
            x1_peaq = _to_2d(x1_c.clone())

            peaq_model = PEAQBasic(sampling_rate=48000).to(self.device)
            with torch.no_grad():
                mms_2f = peaq_model.compute_mms_2f(z_peaq, x1_peaq, self.device)
            return float(mms_2f.mean()) if torch.is_tensor(mms_2f) else float(mms_2f)
        except Exception as e:
            print(f"Warning: 2f-model calculation failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Logging helpers
    # ------------------------------------------------------------------ #
    def _build_sample_log(self, z, y, x1, idx, num_steps, prefix="val_samples"):
        """Create wandb log payload for a single audio sample."""
        return {
            f"{prefix}/{idx}/{num_steps}/audio_ground_truth": wandb.Audio(t2n(z), sample_rate=48000),
            f"{prefix}/{idx}/{num_steps}/audio_conditional":  wandb.Audio(t2n(y), sample_rate=48000),
            f"{prefix}/{idx}/{num_steps}/audio_generated":    wandb.Audio(t2n(x1), sample_rate=48000),
            f"{prefix}/{idx}/{num_steps}/spec_ground_truth":  wandb.Image(draw_spec(t2n(z), sr=48000, return_fig=True)),
            f"{prefix}/{idx}/{num_steps}/spec_conditional":   wandb.Image(draw_spec(t2n(y), sr=48000, return_fig=True)),
            f"{prefix}/{idx}/{num_steps}/spec_generated":     wandb.Image(draw_spec(t2n(x1), sr=48000, return_fig=True)),
        }

    # ------------------------------------------------------------------ #
    #  Train step
    # ------------------------------------------------------------------ #
    def _train_step(self, batch_data, step, **kwargs):
        sr_values = batch_data['low_sr']
        lr_bin_count, hf_start_bin = self._get_freq_bins(sr_values[0])

        z = batch_data['hr'].to(self.device)
        if 'lr_wave' in batch_data:
            y = batch_data['lr_wave'].to(self.device)
        elif self.lowpass_on_device:
            y = self._make_lr_wave(z, sr_values)
        else:
            raise KeyError("batch_data must include 'lr_wave' unless lowpass_on_device is enabled.")
        batch_size = z.shape[0]

        Z = self._preprocess(z)
        Y = self._preprocess(y)
        Y_lr, Y_hr, Z_hr = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)
        Y_lr = self._as_model_input(Y_lr)
        Y_hr = self._as_model_input(Y_hr)
        Z_hr = self._as_model_input(Z_hr)

        t = torch.rand([batch_size, 1, 1, 1], device=self.device)
        x0 = self.path.sample_source(Y_hr)
        xt = self._as_model_input(self.path.sample_xt(x0, Z_hr, t))

        output = self.model(xt, t, Y_lr, sr_values)
        target = self.path.get_target_vector_field(xt, x0, Z_hr, t)
        loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        return loss, {"cfm_loss": loss}

    # ------------------------------------------------------------------ #
    #  Validation step
    # ------------------------------------------------------------------ #
    def _val_step(self, batch_data, idx, val_idx, **kwargs):
        ode_steps = kwargs.get('ode_steps', 4)
        val_max_sec = kwargs.get('val_max_sec', 5)
        upscale_validation = kwargs.get('upscale_validation', False)
        validation_loss_metric = kwargs.get('validation_loss_metric', 'loss')
        upscale_chunk_sec = kwargs.get('upscale_chunk_sec', None)
        global_step = kwargs.get('global_step', 0)
        val_audio_output_dir = kwargs.get('val_audio_output_dir', None)
        val_audio_reference_once = kwargs.get('val_audio_reference_once', False)
        guidance_scale = kwargs.get('guidance_scale', None)
        reconstruction_method = kwargs.get('reconstruction_method', 'original')

        sr_values = batch_data['low_sr']
        current_sr = sr_values[0]
        lr_bin_count, hf_start_bin = self._get_freq_bins(current_sr)
        target_sr = getattr(getattr(self.transform, "complex_stft", None), "sampling_rate", 48000)

        z = batch_data['hr'].to(self.device)[..., :target_sr * val_max_sec]
        if 'lr_wave' in batch_data:
            y = batch_data['lr_wave'].to(self.device)[..., :target_sr * val_max_sec]
        elif self.lowpass_on_device:
            y = self._make_lr_wave(z, sr_values)
        else:
            raise KeyError("batch_data must include 'lr_wave' unless lowpass_on_device is enabled.")
        batch_size = z.shape[0]

        loss = None
        if not upscale_validation:
            Z = self._preprocess(z)
            Y = self._preprocess(y)
            Y_lr, Y_hr, Z_hr = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)
            Y_lr = self._as_model_input(Y_lr)
            Y_hr = self._as_model_input(Y_hr)
            Z_hr = self._as_model_input(Z_hr)

            t = torch.rand([batch_size, 1, 1, 1], device=self.device)
            x0 = self.path.sample_source(Y_hr)
            xt = self._as_model_input(self.path.sample_xt(x0, Z_hr, t))
            output = self.model(xt, t, Y_lr, sr_values)
            target = self.path.get_target_vector_field(xt, x0, Z_hr, t)
            loss = flow_matching_loss(predicted_vf=output, target_vf=target)

        with torch.no_grad():
            x1_wave = self._synthesize_upscale_chunks(
                y,
                z,
                sr_values,
                current_sr,
                ode_steps,
                chunk_sec=upscale_chunk_sec,
                guidance_scale=guidance_scale,
                reconstruction_method=reconstruction_method,
            )
            z_metric = z[..., :x1_wave.shape[-1]]
            lsd_total_values, lsd_high_values, lsd_low_values = self._compute_lsd(
                x1_wave,
                z_metric,
                current_sr,
                reduce=False,
            )
            lsd_total_tensor = lsd_total_values.mean()
            lsd_high_tensor = lsd_high_values.mean()
            lsd_low_tensor = lsd_low_values.mean()

        if upscale_validation:
            metric_losses = {
                "lsd_total": lsd_total_tensor,
                "lsd_high": lsd_high_tensor,
                "lsd_low": lsd_low_tensor,
            }
            loss = metric_losses.get(validation_loss_metric, lsd_high_tensor)
            loss_per_sample = {
                "lsd_total": lsd_total_values,
                "lsd_high": lsd_high_values,
                "lsd_low": lsd_low_values,
            }.get(validation_loss_metric, lsd_high_values)
        else:
            loss_per_sample = None

        audio_paths = self._save_validation_audio_triplets(
            z_metric,
            y[..., :z_metric.shape[-1]],
            x1_wave[..., :z_metric.shape[-1]],
            batch_data,
            idx,
            global_step,
            val_audio_output_dir,
            reference_once=val_audio_reference_once,
        )

        # Sample logging for selected indices
        log_payload = {}
        if idx in val_idx:
            with torch.no_grad():
                x1_wave_cfg = self._synthesize_upscale_chunks(
                    y[0:1],
                    z[0:1],
                    sr_values,
                    current_sr,
                    ode_steps,
                    chunk_sec=upscale_chunk_sec,
                    guidance_scale=1.5,
                    reconstruction_method=reconstruction_method,
                )

                min_len = min(z.shape[-1], x1_wave_cfg.shape[-1])
                log_payload = self._build_sample_log(
                    z[0:1, ..., :min_len], y[0:1, ..., :min_len], x1_wave_cfg,
                    idx, ode_steps, prefix="val_samples")

        return {
            'loss': loss,
            'loss_per_sample': [float(value) for value in loss_per_sample] if loss_per_sample is not None else None,
            'lsd_total': float(lsd_total_tensor),
            'lsd_high': float(lsd_high_tensor),
            'lsd_low': float(lsd_low_tensor),
            'lsd_total_per_sample': [float(value) for value in lsd_total_values],
            'lsd_high_per_sample': [float(value) for value in lsd_high_values],
            'lsd_low_per_sample': [float(value) for value in lsd_low_values],
            'audio_paths': audio_paths,
            'log_payload': log_payload,
        }

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, val_idx, ode_steps=4, guidance_scale=None,
                 max_batches=None, global_step=0, **kwargs):

        if hasattr(self, '_eval_output_base'):
            del self._eval_output_base

        self.model.eval()
        total_metrics = {'lsd_total': 0.0, 'lsd_high': 0.0, 'lsd_low': 0.0}
        if PEAQ_AVAILABLE:
            total_metrics['2f_model'] = 0.0
        cnt = 0

        with torch.no_grad():
            eval_pbar = tqdm(self.val_loader, desc='Evaluating...', dynamic_ncols=True)
            for batch_idx, batch in enumerate(eval_pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                outdict = self._evaluate_step(batch, batch_idx, val_idx,
                                              ode_steps=ode_steps,
                                              guidance_scale=guidance_scale)
                lsd_metrics = outdict.get('lsd_metrics', {})
                peaq_metric = outdict.get('2f_model', None)

                if lsd_metrics:
                    for key in total_metrics:
                        if key in lsd_metrics:
                            total_metrics[key] += lsd_metrics[key]
                    cnt += 1

                    postfix = {k: f'{lsd_metrics.get(k, 0):.4f}' for k in ('lsd_total', 'lsd_high', 'lsd_low')}
                    if peaq_metric is not None:
                        postfix['2f_model'] = f'{peaq_metric:.4f}'
                        total_metrics['2f_model'] += peaq_metric
                    eval_pbar.set_postfix(postfix)

                if outdict.get('log_payload'):
                    self.logger.log(outdict['log_payload'], step=global_step)

        avg_metrics = {k: v / cnt if cnt > 0 else 0.0 for k, v in total_metrics.items()}
        self.logger.log({f'eval/{k}': v for k, v in avg_metrics.items()}, step=global_step)

        print(f"Evaluation finished over {cnt} batches.")
        for k, v in avg_metrics.items():
            print(f"   {k}: {v:.4f}")

    def _evaluate_step(self, batch_data, idx, val_idx, **kwargs):
        ode_steps = kwargs.get('ode_steps', 5)
        guidance_scale = kwargs.get('guidance_scale', None)
        val_max_sec = kwargs.get('val_max_sec', 5)

        sr_values = batch_data['low_sr']
        current_sr = sr_values[0]
        lr_bin_count, hf_start_bin = self._get_freq_bins(current_sr)

        z = batch_data['hr'].to(self.device)[..., :48000 * val_max_sec]
        if 'lr_wave' in batch_data:
            y = batch_data['lr_wave'].to(self.device)[..., :48000 * val_max_sec]
        elif self.lowpass_on_device:
            y = self._make_lr_wave(z, sr_values)
        else:
            raise KeyError("batch_data must include 'lr_wave' unless lowpass_on_device is enabled.")

        Z = self._preprocess(z)
        Y = self._preprocess(y)
        Y_lr, Y_hr, _ = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)
        Y_lr = self._as_model_input(Y_lr)
        Y_hr = self._as_model_input(Y_hr)

        # ODE synthesis
        x1_wave = self._synthesize_waveform(
            Y_lr, Y_hr, sr_values, lr_bin_count, hf_start_bin, ode_steps, guidance_scale,
            orig_length=z.shape[-1])

        min_len = min(z.shape[-1], x1_wave.shape[-1])
        z_c = z[..., :min_len]
        x1_c = x1_wave[..., :min_len]

        # LSD metrics (generated vs GT)
        lsd_total, lsd_high, lsd_low = self._compute_lsd(x1_c, z_c, current_sr)
        lsd_metrics = {
            'lsd_total': float(lsd_total),
            'lsd_high': float(lsd_high),
            'lsd_low': float(lsd_low),
        }

        peaq_metric = self._calculate_2f_model_metric(z_c, x1_c) if PEAQ_AVAILABLE else None

        # Save audio
        self._save_audio_samples(z_c, y[..., :min_len], x1_c, batch_data, idx, ode_steps, current_sr)

        # Sample logging
        log_payload = {}
        if idx in val_idx:
            log_payload = self._build_sample_log(
                z_c[0:1], y[..., :min_len][0:1], x1_c[0:1],
                idx, ode_steps, prefix="eval_samples")

        result = {'lsd_metrics': lsd_metrics, 'log_payload': log_payload}
        if peaq_metric is not None:
            result['2f_model'] = peaq_metric
        return result

    def _save_audio_samples(self, z_c, y_c, x1_c, batch_data, batch_idx, ode_steps, current_sr):
        """Save generated audio to disk for offline evaluation."""
        if not hasattr(self, '_eval_output_base'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = getattr(self.logger, 'run_name', 'eval')
            self._eval_output_base = f"samples/outputs/{exp_name}_{current_sr}k_ode{ode_steps}_{timestamp}"
            os.makedirs(self._eval_output_base, exist_ok=True)

        batch_size = z_c.shape[0]
        relpaths = batch_data.get('relpath', [f"batch{batch_idx:03d}_sample{i:02d}.wav" for i in range(batch_size)])
        if not isinstance(relpaths, list):
            relpaths = [relpaths] * batch_size

        for i in range(batch_size):
            out_dir = os.path.join(self._eval_output_base, os.path.dirname(relpaths[i]))
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(relpaths[i]))[0]
            sf.write(os.path.join(out_dir, f"{base_name}.wav"), t2n(x1_c[i:i+1]).squeeze(), 48000)

        if batch_idx < 3:
            print(f"Saved batch {batch_idx} samples under {self._eval_output_base}")
