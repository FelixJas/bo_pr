#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Purely continuous test problems with a known number of active dimensions.

These mirror the semantics of bounce's `EffectiveDimBoTorchBenchmark` /
`ShiftedAckley10` (bounce/bounce/benchmarks.py) so that the identical objective
can be run in both codebases.
"""
from typing import List, Optional

import torch
from botorch.test_functions.synthetic import Ackley
from torch import Tensor
from torch.nn import Module

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class ContinuousTestProblem(DiscreteTestProblem):
    r"""Base class for problems without discrete parameters.

    `DiscreteTestProblem._setup` requires at least one discrete feature. This
    overrides it for the all-continuous case while still exposing every
    attribute the BO loop reads (`cont_indices`, `integer_indices`,
    `categorical_indices`, `effective_dim`, `one_hot_bounds`, ...).
    """

    def _setup(
        self,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> None:
        if integer_indices or categorical_indices:
            raise ValueError(
                "ContinuousTestProblem does not accept discrete features."
            )
        dim = self.bounds.shape[-1]
        device = self.bounds.device
        empty = torch.tensor([], dtype=torch.long, device=device)
        identity = torch.arange(dim, dtype=torch.long, device=device)

        self.register_buffer("_orig_integer_indices", empty.clone())
        self.register_buffer("_orig_categorical_indices", empty.clone())
        self.register_buffer("_orig_cont_indices", identity.clone())
        self.register_buffer("_orig_bounds", self.bounds.clone())
        # No reordering is required, since every feature is continuous.
        self.register_buffer("_remapper", identity.clone())
        self.register_buffer("_reverse_mapper", identity.clone())
        self.register_buffer("cont_indices", identity.clone())
        self.register_buffer("integer_indices", empty.clone())
        self.register_buffer("categorical_indices", empty.clone())
        self.effective_dim = dim
        self.register_buffer("one_hot_bounds", self.bounds.clone())


class ShiftedAckleyActiveDim(ContinuousTestProblem):
    r"""Ackley on an ambient space where only the first `active_dim` coordinates
    affect the objective; the remaining `dim - active_dim` are inert.

    This is bounce's `EffectiveDimBoTorchBenchmark` construction: the objective is
    `Ackley(dim=active_dim)` applied to `X[..., :active_dim]`. The optimizer is
    never told which coordinates are active.

    The per-coordinate offsets shift the search box so that the optimizer does not
    sit at its center. bounce warns that a centered optimizer yields "overly
    optimistic results" for random-embedding methods, whose subspaces always
    contain the center (see the `AckleyEffectiveDim` docstring); `ShiftedAckley10`
    is their fix, and these are its offsets.

    NOTE: `active_dim` is bounce's notion of "effective dimensionality". It is
    deliberately not called `effective_dim`, which in this codebase means the
    dimension of the (one-hot) search space and must stay equal to `dim`.
    """

    # bounce/bounce/benchmarks.py:674 (ShiftedAckley10.offsets)
    _OFFSETS: List[float] = [
        -14.15468831,
        -17.35934204,
        4.93227439,
        30.68108305,
        -20.94097318,
        -9.68946759,
        11.23919487,
        4.93101114,
        2.87604112,
        -31.0805155,
    ]
    _LB: float = -32.768
    _UB: float = 32.768

    def __init__(
        self,
        dim: int,
        active_dim: int,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        if active_dim > dim:
            raise ValueError(f"active_dim ({active_dim}) must not exceed dim ({dim}).")
        if active_dim > len(self._OFFSETS):
            raise ValueError(
                f"Only {len(self._OFFSETS)} offsets are defined, so active_dim must "
                f"be <= {len(self._OFFSETS)}; got {active_dim}."
            )
        Module.__init__(self)
        self.problem = Ackley(dim=active_dim, noise_std=None)
        self.dim = dim
        self.active_dim = active_dim
        self.noise_std = noise_std
        self.negate = negate

        offsets = torch.tensor(self._OFFSETS[:active_dim], dtype=torch.double)
        bounds = torch.stack(
            (
                torch.full((dim,), self._LB, dtype=torch.double),
                torch.full((dim,), self._UB, dtype=torch.double),
            )
        )
        bounds[:, :active_dim] -= offsets
        self.register_buffer("bounds", bounds)
        self._setup()

    def evaluate_true(self, X: Tensor) -> Tensor:
        # Only the first `active_dim` coordinates reach the objective. Noise and
        # negation are applied by DiscreteTestProblem.forward.
        return self.problem.evaluate_true(X[..., : self.active_dim])
