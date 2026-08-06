#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The CoCaBO synthetic mixed categorical/continuous benchmarks `Func2C` and `Func3C`.

Both come from Ru et al., "Bayesian Optimisation over Multiple Continuous and
Categorical Inputs" (https://arxiv.org/abs/1906.08878); 
Each benchmark sums two-dimensional classics (Rosenbrock, six-hump camel, Beale)
selected by the categorical variables, so a category choice replaces the continuous landscape . 
Both are **minimized** (CoCaBO reports `max(-f)`).

The categoricals have *unequal* cardinalities (3/5 and 3/5/4), which no other
benchmark in the PR-vs-Bounce study has, and neither optimum sits at all-zeros.

A matching implementation lives in bounce (`bounce/bounce/benchmarks.py`,
classes `Func2C` / `Func3C`); `Thesis_Repo/analysis/check_parity.py` verifies
that the two agree.
"""
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


# The three two-dimensional objectives, with CoCaBO's scaling divisors. 

# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
def _rosenbrock(x1: Tensor, x2: Tensor) -> Tensor:
    return (100.0 * (x2 - x1**2) ** 2 + (x1 - 1.0) ** 2) / 300.0

#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html       
def _sixhumpcamel(x1: Tensor, x2: Tensor) -> Tensor:
    term1 = (4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2
    term2 = x1 * x2
    term3 = (-4.0 + 4.0 * x2**2) * x2**2
    return (term1 + term2 + term3) / 10.0

# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
def _beale(x1: Tensor, x2: Tensor) -> Tensor:
    # CoCaBO's `mybeale` halves X on entry and doubles it again when unpacking x1/x2, which
    # cancels exactly; this is that function with the no-op removed.
    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2**2) ** 2
        + (2.625 - x1 + x1 * x2**3) ** 2
    ) / 50.0


class CoCaBOSynthetic(DiscreteTestProblem):
    r"""Base class for CoCaBO's `Func2C` / `Func3C`.

    Declares `len(_cardinalities)` categorical dimensions followed by two
    continuous ones on `[-1, 1]`, in the order CoCaBO uses. 
    `_setup` permutes them internally to (continuous, categorical); 
    `forward` un-permutes before calling `evaluate_true`, so subclasses always see CoCaBO's order.

    Subclasses evaluate one point at a time (`_evaluate_single`) so that their code
    can keep the reference implementation's `if`/`elif` structure over the category
    values -- the same approach `PestControl` takes in this repo.
    """

    # Number of categories per categorical variable, in declaration order.
    _cardinalities: Tuple[int, ...] = ()
    _cont_bounds: Tuple[float, float] = (-1.0, 1.0)
    _n_continuous: int = 2

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = len(self._cardinalities) + self._n_continuous
        self._bounds = [(0.0, float(c - 1)) for c in self._cardinalities] + [
            self._cont_bounds
        ] * self._n_continuous
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            categorical_indices=list(range(len(self._cardinalities))),
        )

    def _evaluate_single(self, x: Tensor) -> Tensor:
        r"""Evaluate one `dim`-dim point: categories first, then `x1`, `x2`."""
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        flat = X.reshape(-1, self.dim)
        return torch.stack([self._evaluate_single(x) for x in flat]).reshape(
            X.shape[:-1]
        )


class Func2C(CoCaBOSynthetic):
    r"""CoCaBO's `Func2C`: 2 categorical (3 and 5 values) + 2 continuous dims.

    Note that `ht2` has 5 values but only 3 branches, so categories 2, 3 and 4
    all mean Beale.

    Minimum `f = -0.2063` at `h = (1, 1)`, `x = (-0.0449, 0.3563)`, i.e. both
    categoricals pick the six-hump camel.
    """

    _cardinalities: Tuple[int, ...] = (3, 5)

    def _evaluate_single(self, x: Tensor) -> Tensor:
        ht1 = int(x[0].round())
        ht2 = int(x[1].round())
        #TODO: Why the scaling?
        # CoCaBO scales the continuous inputs by 2 before evaluating, so the
        # effective box is [-2, 2]^2, and reuses that point for every term.
        x1 = 2.0 * x[2]
        x2 = 2.0 * x[3]

        if ht1 == 0:  # rosenbrock
            f = _rosenbrock(x1, x2)
        elif ht1 == 1:  # six hump
            f = _sixhumpcamel(x1, x2)
        else:  # beale 
            f = _beale(x1, x2)  

        if ht2 == 0:  # rosenbrock
            f = f + _rosenbrock(x1, x2)
        elif ht2 == 1:  # six hump
            f = f + _sixhumpcamel(x1, x2)
        else:  # beale
            f = f + _beale(x1, x2)

        return f


class Func3C(CoCaBOSynthetic):
    r"""CoCaBO's `Func3C`: 3 categorical (3, 5 and 4 categories) + 2 continuous dims.

    The first two terms are `Func2C`'s; `ht3` adds a third, weighted one. Kept as a
    full copy rather than a call into `Func2C`, mirroring CoCaBO's implementation.

    Global optimum at 7 * Camel, `f = -0.7221` at `h = (1, 1, 0)`, `x = (0.0449, -0.3563)`.
    """

    _cardinalities: Tuple[int, ...] = (3, 5, 4)

    def _evaluate_single(self, x: Tensor) -> Tensor:
        ht1 = int(x[0].round())
        ht2 = int(x[1].round())
        ht3 = int(x[2].round())
        x1 = 2.0 * x[3]
        x2 = 2.0 * x[4]

        if ht1 == 0:  # rosenbrock
            f = _rosenbrock(x1, x2)
        elif ht1 == 1:  # six hump
            f = _sixhumpcamel(x1, x2)
        else:  # beale -- `elif ht1 == 2` in the reference, but ht1 has only 3
            f = _beale(x1, x2)  # categories, so an else avoids leaving f unbound

        if ht2 == 0:  # rosenbrock
            f = f + _rosenbrock(x1, x2)
        elif ht2 == 1:  # six hump
            f = f + _sixhumpcamel(x1, x2)
        else:  # beale
            f = f + _beale(x1, x2)

        if ht3 == 0:  # six hump
            f = f + 5 * _sixhumpcamel(x1, x2)
        elif ht3 == 1:  # rosenbrock
            f = f + 2 * _rosenbrock(x1, x2)
        else:  # beale, weighted by the category index itself
            f = f + ht3 * _beale(x1, x2)

        return f
