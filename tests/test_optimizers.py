"""Small CPU regression checks; no datasets, downloads, or full experiments.

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from AdamSHANG import (  # noqa: E402
    AdamSHANG,
    AdamSHANGProjected,
    AdamSHANGs,
    AdamSHANGsProjected,
)
from SHANG_PlusPlus import SHANG_PlusPlus  # noqa: E402


ADAPTIVE_OPTIMIZERS = (
    AdamSHANG,
    AdamSHANGs,
    AdamSHANGProjected,
    AdamSHANGsProjected,
)
PROJECTED_PAIRS = (
    (AdamSHANG, AdamSHANGProjected),
    (AdamSHANGs, AdamSHANGsProjected),
)


def parameters():
    """Several tensor shapes exercise parameter-group trace aggregation."""
    return [
        torch.nn.Parameter(
            torch.tensor([[0.7, -0.4], [0.2, 0.9]], dtype=torch.float64, device="cpu")
        ),
        torch.nn.Parameter(
            torch.tensor([0.3, -0.8, 0.1], dtype=torch.float64, device="cpu")
        ),
    ]


def assign_gradients(params, step):
    """A reproducible nonrandom stream, with occasional all-zero gradients."""
    for index, parameter in enumerate(params):
        if step % 7 == 0:
            parameter.grad = torch.zeros_like(parameter)
        else:
            coordinates = torch.arange(
                parameter.numel(), dtype=parameter.dtype, device="cpu"
            ).reshape_as(parameter)
            parameter.grad = torch.sin(coordinates + 0.25 * step + index)


class AdaptiveOptimizerTests(unittest.TestCase):
    def assert_valid_state(self, optimizer, params):
        for parameter in params:
            self.assertEqual(parameter.device.type, "cpu")
            self.assertTrue(bool(torch.isfinite(parameter).all()))
            state = optimizer.state[parameter]
            self.assertEqual(state["P"].shape, parameter.shape)
            self.assertEqual(state["y"].shape, parameter.shape)
            self.assertTrue(bool(torch.isfinite(state["P"]).all()))
            self.assertTrue(bool(torch.isfinite(state["y"]).all()))
            self.assertTrue(bool((state["P"] > 0).all()))
        for group in optimizer.param_groups:
            self.assertTrue(math.isfinite(group["alpha_prev"]))
            self.assertGreater(group["alpha_prev"], 0.0)
            self.assertLessEqual(group["alpha_prev"], group["alpha_clip"])

    def test_all_adaptive_variants_have_finite_positive_states(self):
        for optimizer_class in ADAPTIVE_OPTIMIZERS:
            with self.subTest(optimizer=optimizer_class.__name__):
                params = parameters()
                optimizer = optimizer_class(
                    params, lr=0.05, beta=0.05, gamma=0.01, weight_decay=0.01
                )
                for step in range(30):
                    assign_gradients(params, step)
                    optimizer.step()
                    self.assert_valid_state(optimizer, params)

    def test_projection_bounds_auxiliary_state_when_clipping_is_active(self):
        radius = 0.1
        for _, optimizer_class in PROJECTED_PAIRS:
            with self.subTest(optimizer=optimizer_class.__name__):
                parameter = torch.nn.Parameter(
                    torch.tensor([10.0, -10.0, 5.0], dtype=torch.float64, device="cpu")
                )
                optimizer = optimizer_class(
                    [parameter], lr=0.1, beta=0.05, gamma=0.01,
                    weight_decay=0.0, R=radius
                )
                hit_boundary = False
                for step in range(20):
                    parameter.grad = torch.tensor(
                        [1.0, -2.0, 3.0], dtype=torch.float64, device="cpu"
                    ) * (-1.0 if step % 2 else 1.0)
                    optimizer.step()
                    self.assert_valid_state(optimizer, [parameter])
                    auxiliary = optimizer.state[parameter]["y"]
                    self.assertLessEqual(float(auxiliary.abs().max()), radius / 2.0)
                    hit_boundary |= bool((auxiliary.abs() == radius / 2.0).any())
                self.assertTrue(hit_boundary, "The test must exercise actual clipping.")

    def test_inactive_projection_matches_unprojected_updates(self):
        for unprojected_class, projected_class in PROJECTED_PAIRS:
            for weight_decay in (0.0, 0.01):
                with self.subTest(
                    optimizer=unprojected_class.__name__, weight_decay=weight_decay
                ):
                    controls, projected = parameters(), parameters()
                    kwargs = dict(
                        lr=0.05, beta=0.05, gamma=0.01, weight_decay=weight_decay
                    )
                    control_optimizer = unprojected_class(controls, **kwargs)
                    projected_optimizer = projected_class(projected, R=1e6, **kwargs)
                    for step in range(30):
                        assign_gradients(controls, step)
                        assign_gradients(projected, step)
                        control_optimizer.step()
                        projected_optimizer.step()
                        for control, candidate in zip(controls, projected):
                            torch.testing.assert_close(candidate, control, rtol=0, atol=0)
                            for state_name in ("P", "y"):
                                torch.testing.assert_close(
                                    projected_optimizer.state[candidate][state_name],
                                    control_optimizer.state[control][state_name],
                                    rtol=0, atol=0,
                                )
                        self.assertEqual(
                            projected_optimizer.param_groups[0]["alpha_prev"],
                            control_optimizer.param_groups[0]["alpha_prev"],
                        )

    def test_parameters_without_gradients_are_unchanged(self):
        for optimizer_class in ADAPTIVE_OPTIMIZERS:
            with self.subTest(optimizer=optimizer_class.__name__):
                params = parameters()
                original = params[1].detach().clone()
                optimizer = optimizer_class(params, weight_decay=0.1)
                params[0].grad = torch.ones_like(params[0])
                optimizer.step()
                torch.testing.assert_close(params[1], original, rtol=0, atol=0)
                self.assertNotIn(params[1], optimizer.state)


class ShangPlusPlusTests(unittest.TestCase):
    def test_multiple_cpu_steps_are_finite_and_update_parameters(self):
        params = parameters()
        initial = [parameter.detach().clone() for parameter in params]
        optimizer = SHANG_PlusPlus(
            params, lr=0.1, time_scale=3.0, rho=1.0, weight_decay=0.01
        )
        for step in range(20):
            assign_gradients(params, step)
            optimizer.step()
            for parameter in params:
                self.assertTrue(bool(torch.isfinite(parameter).all()))
                self.assertTrue(
                    bool(torch.isfinite(optimizer.state[parameter]["vel_prev"]).all())
                )
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(initial, params))
        )


class SafeImportTests(unittest.TestCase):
    def test_root_modules_do_not_train_download_or_create_results_on_import(self):
        # The abandoned equivalent-convex experiments are not paper entry points.
        legacy_names = {
            "reddi_convex_deterministic",
            "reddi_convex_stochastic",
            "reddi_stochastic_convex",
        }
        modules = [
            path.stem
            for path in sorted(REPOSITORY_ROOT.glob("*.py"))
            if path.stem not in legacy_names and not path.name.startswith("__")
        ]
        self.assertTrue(modules)
        child_code = textwrap.dedent(
            """
            import importlib
            from pathlib import Path
            import socket
            import sys
            from unittest.mock import patch
            import torch
            from torch.utils.data import DataLoader

            sys.path.insert(0, sys.argv[1])
            before = set(Path.cwd().iterdir())
            def forbidden(*args, **kwargs):
                raise AssertionError("Import attempted training, data loading, or network access.")
            with patch.object(socket.socket, "connect", forbidden), \\
                 patch.object(torch.Tensor, "backward", forbidden), \\
                 patch.object(DataLoader, "__init__", forbidden):
                importlib.import_module(sys.argv[2])
            assert set(Path.cwd().iterdir()) == before, "Import created output files/directories."
            """
        )
        with tempfile.TemporaryDirectory(prefix="adamshang-import-tests-") as directory:
            temporary = Path(directory)
            workdir = temporary / "work"
            workdir.mkdir()
            environment = os.environ.copy()
            environment.update(
                PYTHONDONTWRITEBYTECODE="1",
                MPLCONFIGDIR=str(temporary / "matplotlib-cache"),
                CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="1",
            )
            for module in modules:
                with self.subTest(module=module):
                    completed = subprocess.run(
                        [sys.executable, "-c", child_code, str(REPOSITORY_ROOT), module],
                        cwd=workdir,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=45,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode, 0,
                        msg=f"{module}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                    )
                    self.assertEqual(completed.stdout, "", msg=module)


if __name__ == "__main__":
    unittest.main()
