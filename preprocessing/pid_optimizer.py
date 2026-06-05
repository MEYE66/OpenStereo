# Created: 2026-03-28

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PIDResult:
    exposure: float
    gain: int
    score: float
    image: np.ndarray


class PIDOptimizer:
    def __init__(
        self,
        image_formation_model,
        metric,
        bounds,
        setpoint: float,
        max_iter: int = 30,
        tol: float = 1e-4,
        kp_exp: float = 0.45,
        ki_exp: float = 0.08,
        kd_exp: float = 0.05,
        kp_gain: float = 0.25,
        ki_gain: float = 0.02,
        kd_gain: float = 0.01,
        exp_step: float = 2.0,
        gain_step: int = 1,
    ):
        self.model = image_formation_model
        self.metric = metric
        self.bounds = bounds
        self.setpoint = setpoint
        self.max_iter = max_iter
        self.tol = tol

        self.kp_exp = kp_exp
        self.ki_exp = ki_exp
        self.kd_exp = kd_exp
        self.kp_gain = kp_gain
        self.ki_gain = ki_gain
        self.kd_gain = kd_gain

        self.exp_step = exp_step
        self.gain_step = gain_step

        self.prev_error = 0.0
        self.integral = 0.0

    def _clip_params(self, exposure: float, gain: float) -> Tuple[float, int]:
        exposure = float(np.clip(exposure, self.bounds[0][0], self.bounds[0][1]))
        gain = int(np.clip(round(gain), self.bounds[1][0], self.bounds[1][1]))
        return exposure, gain

    def optimize(self, image: np.ndarray) -> PIDResult:
        current_exp = 0.5 * (self.bounds[0][0] + self.bounds[0][1])
        current_gain = 0.5 * (self.bounds[1][0] + self.bounds[1][1])

        best_exp, best_gain = self._clip_params(current_exp, current_gain)
        best_img = self.model.subframes_fusion(image, exp_time=best_exp, analog_gain=best_gain)
        best_score = float(self.metric.evaluate(best_img))

        self.prev_error = 0.0
        self.integral = 0.0

        for step_idx in range(self.max_iter):
            exp_i, gain_i = self._clip_params(current_exp, current_gain)
            formed_img = self.model.subframes_fusion(image, exp_time=exp_i, analog_gain=gain_i)
            score = float(self.metric.evaluate(formed_img))

            if score > best_score:
                best_score = score
                best_exp, best_gain = exp_i, gain_i
                best_img = formed_img

            error = self.setpoint - score
            if abs(error) <= self.tol:
                break

            self.integral += error
            derivative = 0.0 if step_idx == 0 else (error - self.prev_error)

            pid_exp = self.kp_exp * error + self.ki_exp * self.integral + self.kd_exp * derivative
            pid_gain = self.kp_gain * error + self.ki_gain * self.integral + self.kd_gain * derivative

            exp_adjust = float(np.clip(pid_exp, -self.exp_step, self.exp_step))
            gain_adjust = float(np.clip(pid_gain, -self.gain_step, self.gain_step))

            current_exp += exp_adjust
            current_gain += gain_adjust
            self.prev_error = error

        return PIDResult(exposure=best_exp, gain=best_gain, score=best_score, image=best_img)
