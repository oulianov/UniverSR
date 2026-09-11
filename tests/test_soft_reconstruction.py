import unittest
from unittest.mock import Mock, patch

import torch

from universr import UniverSR
from universr.flow.path import OriginalCFMPath
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT


class SoftReconstructionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = UniverSR(
            model=Mock(
                sr_to_lr_bins={8: 80, 12: 128, 16: 170, 24: 256},
                total_freq_bins=512,
                hr_freq_bins=432,
            ),
            transform=AmplitudeCompressedComplexSTFT(
                window_fn="hann",
                n_fft=1024,
                sampling_rate=48000,
                alpha=0.2,
                beta=1,
                comp_eps=1e-4,
                hop_length=256,
            ),
            path=OriginalCFMPath(),
            device="cpu",
        )

    def test_caps_power_per_frame_and_batch_without_changing_phase_or_low_band(
        self,
    ) -> None:
        for sr_khz in (8, 12, 16, 24):
            with self.subTest(sr_khz=sr_khz):
                frequencies = torch.fft.rfftfreq(1024, 1 / 48000)[:-1]
                cutoff = sr_khz * 500
                band = (frequencies >= cutoff - 500) & (frequencies < cutoff)
                # Unequal frames and batch items, including a silent reference.
                peak_power = torch.tensor([[0.8, 0.02, 0.0], [0.1, 0.4, 0.6]])
                source = torch.full((2, 512, 3), 10 + 10j)
                source[:, band, :] = torch.complex(
                    peak_power.sqrt(), torch.zeros_like(peak_power)
                )[:, None, :]
                source[:, band.nonzero()[0].item(), :] *= 0.5
                generated = torch.full((2, 432, 3), -2 + 3j)
                generated[:, -2, :] = 0.001 - 0.002j
                generated[:, -1, :] = 0
                source_spec = self.model.transform.compress(source)
                source_spec = torch.view_as_real(source_spec).permute(0, 3, 1, 2)
                generated_spec = torch.view_as_real(
                    self.model.transform.compress(generated)
                ).permute(0, 3, 1, 2)
                condition = source_spec * 0.01
                with (
                    patch.object(
                        self.model, "_preprocess", side_effect=[condition, source_spec]
                    ),
                    patch(
                        "universr.inference.TorchDiffeqSolver.simulate",
                        return_value=generated_spec,
                    ),
                    patch.object(
                        self.model, "_postprocess", side_effect=lambda spec, **_: spec
                    ),
                ):
                    output = self.model._inference(
                        torch.zeros(2, 1, 2048),
                        sr_khz,
                        "midpoint",
                        1,
                        None,
                        reconstruction_method="seamless_reconstruction_soft",
                        reconstruction_audio=torch.zeros(2, 1, 2048),
                    )
                low_bins = self.model.model.sr_to_lr_bins[sr_khz]
                torch.testing.assert_close(
                    output[:, :, :low_bins],
                    source_spec[:, :, :low_bins],
                    rtol=0,
                    atol=0,
                )
                high_complex = torch.view_as_complex(
                    output[:, :, low_bins:].permute(0, 2, 3, 1).contiguous()
                )
                high = self.model.transform.compress.invert(high_complex)
                reference_power = (
                    self.model.transform.compress.invert(
                        self.model.transform.compress(source[:, band, :])
                    )
                    .abs()
                    .square()
                    .amax(dim=1, keepdim=True)
                )
                self.assertTrue(
                    torch.all(high.abs().square() <= reference_power + 1e-6)
                )
                expected = self.model.transform.compress.invert(
                    self.model.transform.compress(generated[:, low_bins - 80 :])
                )
                torch.testing.assert_close(
                    high.angle(), expected.angle(), atol=1e-5, rtol=1e-5
                )
                # Quiet generated bins stay bit-for-bit unchanged when under the cap.
                torch.testing.assert_close(
                    output[:, :, -2:, :2], generated_spec[:, :, -2:, :2], rtol=0, atol=0
                )
                self.assertTrue(torch.isfinite(output).all())

    def test_real_waveform_reconstruction_attenuates_excess_highs(self) -> None:
        time = torch.arange(48000) / 48000
        source = (0.1 * torch.sin(2 * torch.pi * 7750 * time)).reshape(1, 1, -1)
        generated_wave = (0.9 * torch.sin(2 * torch.pi * 11000 * time)).reshape(
            1, 1, -1
        )
        generated_spec = self.model._preprocess(generated_wave)[:, :, 80:]
        with patch(
            "universr.inference.TorchDiffeqSolver.simulate", return_value=generated_spec
        ):
            normal = self.model._inference(
                source * 0.1, 16, "midpoint", 1, None, "original_signal", source
            )
            soft = self.model._inference(
                source * 0.1,
                16,
                "midpoint",
                1,
                None,
                "seamless_reconstruction_soft",
                source,
            )
        normal_fft = torch.fft.rfft(normal)
        soft_fft = torch.fft.rfft(soft)
        self.assertEqual(soft.shape, (1, 48000))
        self.assertTrue(torch.isfinite(soft).all())
        self.assertLess(soft_fft[0, 11000].abs(), normal_fft[0, 11000].abs() * 0.2)
        torch.testing.assert_close(
            soft_fft[0, 7750], normal_fft[0, 7750], rtol=0.01, atol=0.01
        )

    def test_enhance_passes_padded_source_to_soft_reconstruction(self) -> None:
        source = torch.linspace(-0.2, 0.2, 1000).unsqueeze(0)
        with (
            patch.object(self.model, "_load_audio", return_value=(source, 48000)),
            patch.object(
                self.model,
                "_inference",
                side_effect=lambda audio, *_, **kwargs: kwargs[
                    "reconstruction_audio"
                ].squeeze(1),
            ) as inference,
        ):
            output = self.model.enhance(
                source,
                input_sr=16000,
                reconstruction_method="seamless_reconstruction_soft",
            )
        torch.testing.assert_close(output, source)
        self.assertEqual(
            inference.call_args.kwargs["reconstruction_audio"].shape, (1, 1, 32768)
        )


if __name__ == "__main__":
    unittest.main()
