# Created: 2026-03-28

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class OptimizationResult:
    exposure: float
    gain: int
    score: float
    image: np.ndarray


class NelderMeadOptimizer:
    def __init__(self, image_formation_model, metric, bounds, max_iter: int = 25, tol: float = 1e-5):
        self.model = image_formation_model
        self.metric = metric
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol

    def _evaluate(self, image: np.ndarray, exp: float, gain: int) -> Tuple[float, np.ndarray]:
        formed = self.model.subframes_fusion(image, exp_time=float(exp), analog_gain=int(gain))
        score = float(self.metric.evaluate(formed))
        return score, formed

    def optimize(self, image: np.ndarray) -> OptimizationResult:
        def clamp_params(x: np.ndarray) -> Tuple[float, int]:
            exp = float(np.clip(x[0], self.bounds[0][0], self.bounds[0][1]))
            gain = int(np.clip(round(x[1]), self.bounds[1][0], self.bounds[1][1]))
            return exp, gain

        def objective(x: np.ndarray) -> float:
            exp, gain = clamp_params(x)
            score, _ = self._evaluate(image, exp, gain)
            return -score

        low_exp, high_exp = self.bounds[0]
        low_gain, high_gain = self.bounds[1]
        mid_exp = 0.5 * (low_exp + high_exp)
        mid_gain = int(round(0.5 * (low_gain + high_gain)))

        simplex = np.array(
            [
                [float(low_exp), float(low_gain)],
                [float(high_exp), float(high_gain)],
                [float(mid_exp), float(mid_gain)],
            ],
            dtype=np.float64,
        )

        n = simplex.shape[1]
        fs = np.array([objective(simplex[i]) for i in range(n + 1)], dtype=np.float64)

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for _ in range(self.max_iter):
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]

            x_best = simplex[0]
            x_worst = simplex[-1]
            centroid = simplex[:-1].mean(axis=0)

            x_r = centroid + alpha * (centroid - x_worst)
            f_r = objective(x_r)

            if f_r < fs[0]:
                x_e = centroid + gamma * (x_r - centroid)
                f_e = objective(x_e)
                if f_e < f_r:
                    new_point, f_new = x_e, f_e
                else:
                    new_point, f_new = x_r, f_r
            elif f_r < fs[-2]:
                new_point, f_new = x_r, f_r
            else:
                if f_r < fs[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid + rho * (x_worst - centroid)
                f_c = objective(x_c)

                if f_c < fs[-1]:
                    new_point, f_new = x_c, f_c
                else:
                    new_simplex = np.zeros_like(simplex)
                    new_simplex[0] = x_best
                    for j in range(1, n + 1):
                        new_simplex[j] = x_best + sigma * (simplex[j] - x_best)
                    simplex = new_simplex
                    fs = np.array([objective(simplex[j]) for j in range(n + 1)], dtype=np.float64)
                    if float(np.std(fs)) < self.tol:
                        break
                    continue

            simplex[-1] = new_point
            fs[-1] = f_new

            if float(np.std(fs)) < self.tol:
                break

        best_idx = int(np.argmin(fs))
        best_exp, best_gain = clamp_params(simplex[best_idx])
        best_score, best_img = self._evaluate(image, best_exp, best_gain)
        return OptimizationResult(exposure=best_exp, gain=best_gain, score=best_score, image=best_img)


class GridSearchOptimizer:
    def __init__(
        self,
        image_formation_model,
        metric,
        bounds,
        exposure_steps: int = 16,
        gain_step: int = 1,
    ):
        self.model = image_formation_model
        self.metric = metric
        self.bounds = bounds
        self.exposure_steps = max(exposure_steps, 2)
        self.gain_step = max(int(gain_step), 1)

    def optimize(self, image: np.ndarray) -> OptimizationResult:
        exp_min, exp_max = self.bounds[0]
        gain_min, gain_max = self.bounds[1]

        exp_values = np.linspace(float(exp_min), float(exp_max), num=self.exposure_steps, dtype=np.float32)
        gain_values: List[int] = list(range(int(gain_min), int(gain_max) + 1, self.gain_step))

        best_score = -1e18
        best_exp = float(exp_values[0])
        best_gain = int(gain_values[0])
        best_img = None

        for exp in exp_values:
            for gain in gain_values:
                formed = self.model.subframes_fusion(image, exp_time=float(exp), analog_gain=int(gain))
                score = float(self.metric.evaluate(formed))
                if score > best_score:
                    best_score = score
                    best_exp = float(exp)
                    best_gain = int(gain)
                    best_img = formed

        if best_img is None:
            best_img = self.model.subframes_fusion(image, exp_time=best_exp, analog_gain=best_gain)
        return OptimizationResult(exposure=best_exp, gain=best_gain, score=best_score, image=best_img)
