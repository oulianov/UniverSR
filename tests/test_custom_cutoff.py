import unittest
from unittest.mock import Mock, patch

import torch

from universr import UniverSR
from universr.flow.path import OriginalCFMPath
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT

METHODS = ("original", "original_signal", "seamless_reconstruction_soft")


def tones(*components: tuple[float, float]) -> torch.Tensor:
    time = torch.arange(48000) / 48000
    wave = sum(
        amplitude * torch.sin(2 * torch.pi * frequency * time)
        for frequency, amplitude in components
    )
    return wave.reshape(1, 1, -1)


class CustomCutoffTest(unittest.TestCase):
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

    def run_inference(self, generated: torch.Tensor, source: torch.Tensor, **kwargs):
        generated_spec = self.model._preprocess(generated)[:, :, 80:]
        with patch(
            "universr.inference.TorchDiffeqSolver.simulate",
            return_value=generated_spec,
        ):
            output = self.model._inference(
                self.model._apply_bandwidth_limit(source, 24000, 48000),
                24,
                "midpoint",
                1,
                None,
                reconstruction_audio=source,
                **kwargs,
            )
        return torch.fft.rfft(output)[0].abs()

    def test_keeps_source_below_cutoff_and_generated_above(self) -> None:
        # The 12 kHz model generates from 12 kHz; a 16 kHz cutoff keeps the
        # source below 16 kHz and drops the generated 13 kHz tone.
        source = tones((6000, 0.1), (14000, 0.1), (15750, 0.1))
        generated = tones((13000, 0.3), (19000, 0.05))
        source_fft = torch.fft.rfft(source[0])[0].abs()
        generated_fft = torch.fft.rfft(generated[0])[0].abs()
        for method in METHODS:
            with self.subTest(method=method):
                output = self.run_inference(
                    generated, source, reconstruction_method=method, cutoff_hz=16000
                )
                self.assertLess(output[13000], generated_fft[13000] * 0.01)
                torch.testing.assert_close(
                    output[19000], generated_fft[19000], rtol=0.05, atol=0
                )
                torch.testing.assert_close(
                    output[6000], source_fft[6000], rtol=0.05, atol=0
                )
                if method == "original":
                    # Like the authors' assembly at a model cutoff, the source is
                    # band-limited at the cutoff, which rolls off below it.
                    self.assertGreater(output[14000], source_fft[14000] * 0.5)
                else:
                    torch.testing.assert_close(
                        output[14000], source_fft[14000], rtol=0.05, atol=0
                    )

    def test_soft_reconstruction_caps_at_source_below_custom_cutoff(self) -> None:
        # Only the 500 Hz below the custom cutoff sets the ceiling: the loud
        # source tone at 11.9 kHz sits below it and must not lift the cap.
        source = tones((11900, 0.8), (15750, 0.05))
        generated = tones((19000, 0.5))
        source_fft = torch.fft.rfft(source[0])[0].abs()
        soft = self.run_inference(
            generated,
            source,
            reconstruction_method="seamless_reconstruction_soft",
            cutoff_hz=16000,
        )
        # Uncapped, the generated tone would be 10x the reference tone.
        self.assertLess(soft[19000], source_fft[15750] * 2)
        self.assertGreater(soft[19000], source_fft[15750] * 0.3)

    def test_model_cutoff_matches_default_reconstruction(self) -> None:
        source = tones((5000, 0.2), (11000, 0.1))
        generated = tones((13000, 0.3))
        for method in METHODS:
            with self.subTest(method=method):
                torch.testing.assert_close(
                    self.run_inference(
                        generated, source, reconstruction_method=method, cutoff_hz=12000
                    ),
                    self.run_inference(generated, source, reconstruction_method=method),
                    rtol=0,
                    atol=0,
                )

    def test_rejects_cutoffs_the_model_cannot_cover(self) -> None:
        source = tones((5000, 0.2))
        for cutoff_hz in (10000, 24000):
            with self.subTest(cutoff_hz=cutoff_hz), self.assertRaises(ValueError):
                self.run_inference(
                    source,
                    source,
                    reconstruction_method="original_signal",
                    cutoff_hz=cutoff_hz,
                )
        with self.assertRaises(ValueError):
            self.model._inference(
                source, 24, "midpoint", 1, None, "original", None, cutoff_hz=16000
            )

    def test_enhance_runs_the_highest_model_rate_at_or_below_cutoff(self) -> None:
        source = torch.zeros(1, 48000)
        for cutoff_hz, sr_khz in ((4500, 8), (10000, 16), (12000, 24), (16000, 24)):
            with (
                self.subTest(cutoff_hz=cutoff_hz),
                patch.object(self.model, "_load_audio", return_value=(source, 48000)),
                patch.object(
                    self.model, "_inference", return_value=source
                ) as inference,
            ):
                self.model.enhance(
                    source, reconstruction_method="original", cutoff_hz=cutoff_hz
                )
                self.assertEqual(inference.call_args.args[1], sr_khz)
                self.assertEqual(inference.call_args.kwargs["cutoff_hz"], cutoff_hz)
                self.assertEqual(
                    inference.call_args.kwargs["reconstruction_audio"].shape,
                    (1, 1, 48000),
                )
        with (
            patch.object(self.model, "_load_audio", return_value=(source, 48000)),
            self.assertRaises(ValueError),
        ):
            self.model.enhance(source, cutoff_hz=3000)


if __name__ == "__main__":
    unittest.main()
