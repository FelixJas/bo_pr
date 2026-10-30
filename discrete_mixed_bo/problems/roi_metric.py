#!/usr/bin/env python3
r"""
Region of Interest (ROI) matching: 4 integer-valued + 1 continuous + 2 binary parameters.

A region of interest `(x1, y1, W, H)` is scored against a target box whose
position and size are switched by two binary variables `c` and `d`, with a
continuous `b` arbitrating which of the two error kinds matters more.

Variable order (the benchmark's natural order, shared with bounce):

===  ========  =================  ==================================================
idx  name      domain             role
===  ========  =================  ==================================================
0    `x1`      int `[0, 1000]`    box corner, x
1    `y1`      int `[0, 1000]`    box corner, y
2    `W`       int `[0, 1000]`    box width
3    `H`       int `[0, 1000]`    box height
4    `b`       float `[0, 1]`     weights size accuracy against position accuracy
5    `c`       binary             position target: 0 -> (200, 700), 1 -> (700, 800)
6    `d`       binary             size target:     0 -> (100, 100), 1 -> (300, 300)
===  ========  =================  ==================================================

A matching implementation lives in bounce (`bounce/bounce/benchmarks.py`, class
`ROIMetric`); `Thesis_Repo/analysis/check_parity.py` verifies the two agree.
"""
from typing import List, Optional

import torch
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class ROIMetric(DiscreteTestProblem):
    r"""ROI matching: 4 integer + 1 continuous + 2 binary dimensions.

    The binaries are declared as integers with bounds `[0, 1]`, the same convention
    `ackley53` and `contamination` use in this repo.

    The score is a **maximization** objective already, so `evaluate_true` returns it
    as-is and `get_problem` builds this with `negate=False`. Every other problem in
    this repo passes `negate=True` because the classic synthetics (Ackley, Hartmann,
    ...) are stated as minimization problems and BoTorch's optimizers maximize.
    """

    # x1, y1, W, H  |  b  |  c, d
    _integer_indices: List[int] = [0, 1, 2, 3, 5, 6]

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = 7
        self._bounds = (
            [(0.0, 1000.0)] * 4  # x1, y1, W, H
            + [(0.0, 1.0)]  # b
            + [(0.0, 1.0)] * 2  # c, d
        )
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=list(self._integer_indices),
        )

    def _evaluate_single(self, x: Tensor) -> Tensor:
        r"""Evaluate one 7-dim point given in the benchmark's natural order.

        Here, `c` and `d` arrive as 0/1 rather than as "Low"/"High".
        """
        # Round the discrete dims so a continuous relaxation cannot score a box the
        # problem does not admit -- bounce's benchmarks do the same on entry.
        x1, y1, W, H = (float(torch.round(v)) for v in x[:4])
        b = float(x[4])
        c, d = (int(torch.round(v)) for v in x[5:7])

        # Switch targets based on C, D
        target_x, target_y = (200, 700) if c == 0 else (700, 800)
        target_w, target_h = (100, 100) if d == 0 else (300, 300)

        x2 = min(x1+W, 1000) # Crops box to possible range
        y2 = min(y1+H, 1000)

        w_err = abs(abs(x2 - x1) - target_w)  # width matching, how far is the box width from a defined target?
        h_err = abs(abs(y2 - y1) - target_h)  # height matching
        x_err = abs(min(x1, x2) - target_x)   # x matching, x1 is always smaller than x2 though.
        y_err = abs(min(y1, y2) - target_y)   # y matching

        # Metric 1
        # Choosing correct position and size for a given c/d setting.
        # Variable b "chooses" whether position or size is more important.
        matching_error = (0.2+0.8*b) * (w_err + h_err) + (0.2+0.8*(1-b)) * (x_err + y_err)

        # Metric 2
        # Box should stay within FOV. Going over is allowed but expensive.
        wh_penalty = 0.4*max(x1+W-1000,0) + 0.4*max(y1+H-1000,0)

        # Metric 3
        # Make "high/ high" the best option so it actually has to deal with optimizing b.
        single_optimum_penalty = 0
        if c == 0 or d == 0:
            single_optimum_penalty = 30

        # Preference for b=1 over b=0. The constant 10 puts the global optimum at exactly 0.
        score = 10 + 10*b - matching_error - wh_penalty - single_optimum_penalty

        return torch.as_tensor(score, dtype=x.dtype, device=x.device)

    def evaluate_true(self, X: Tensor) -> Tensor:
        flat = X.reshape(-1, self.dim)
        return torch.stack([self._evaluate_single(x) for x in flat]).reshape(
            X.shape[:-1]
        )
