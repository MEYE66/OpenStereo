# Created: 2025-05-26  
# Author: Gongzhe Li
import os
import cv2
import numpy as np
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer.utils import ImageNoiseMetric, ImageGradientMetric, ImageEntropyMetric, MixedImageMetric
from optimizer.utils import ImageFormationModel



class PIDOptimizer:
    def __init__(self, image_formation_model, metric, bounds, setpoint=0.62,
                 max_iter=30, tol=1e-5, 
                 kp_exp=1.0, ki_exp=0.1, kd_exp=0.01,
                 kp_gain=1.0, ki_gain=0.1, kd_gain=0.01,
                 exp_step=0.5, gain_step=1):
        """
        Enhanced PID Optimizer with full PID control and anti-windup

        Args:
            image_formation_model: Model for image formation
            metric: Quality metric evaluator
            bounds: Bounds for [(exp_min, exp_max), (gain_min, gain_max)]
            setpoint: Target value for the quality metric (default: 0.5)
            max_iter: Maximum iterations (default: 10)
            tol: Error tolerance for convergence (default: 1e-5)
            kp: Proportional gain (default: 1.0)
            ki: Integral gain (default: 0.1)
            kd: Derivative gain (default: 0.01)
            exp_step: Max exposure adjustment per step (default: 1.0)
            gain_step: Max gain adjustment per step (default: 1)
        """
        self.model = image_formation_model
        self.metric = metric
        self.bounds = bounds
        self.setpoint = setpoint
        self.max_iter = max_iter
        self.tol = tol

        self.exp_step = exp_step
        self.kp_exp = kp_exp
        self.ki_exp = ki_exp
        self.kd_exp = kd_exp


        self.gain_step = gain_step
        self.kp_gain = kp_gain
        self.ki_gain = ki_gain
        self.kd_gain = kd_gain

        # PID state variables
        self.prev_error = 0
        self.integral = 0

    def optimize(self, image):
        # Initialize with midpoint values
        current_exp = (self.bounds[0][0] + self.bounds[0][1]) / 2.0
        current_gain = int(round((self.bounds[1][0] + self.bounds[1][1]) / 2.0))

        best_exp, best_gain = current_exp, current_gain
        best_score = 0.
        best_img = None

        # Reset PID state for each optimization run
        self.prev_error = 0
        self.integral = 0

        for i in range(self.max_iter):
            # Apply bounds
            current_exp = np.clip(current_exp, self.bounds[0][0], self.bounds[0][1])
            current_gain = int(np.clip(round(current_gain), self.bounds[1][0], self.bounds[1][1]))

            # Generate image and evaluate metric
            formed_img = self.model.simulate(
                image,
                exp_time=current_exp,
                analog_gain=current_gain
            )
            current_score = self.metric.evaluate(formed_img)

            # Update best solution if improved
            if current_score > best_score:
                best_exp, best_gain = current_exp, current_gain
                best_score = current_score
                # best_img = formed_img

            # Calculate error and check convergence
            error = self.setpoint - current_score
            if abs(error) < self.tol:
                break

            # PID calculations
            P_exp = self.kp_exp * error
            P_gain = self.kp_gain * error

            # Update integral term with anti-windup protection
            self.integral += error
            I_exp = self.ki_exp * self.integral
            I_gain = self.ki_gain * self.integral

            # Derivative term (avoid on first iteration)
            D_exp = 0
            D_gain = 0
            if i > 0:
                D_exp = self.kd_exp * (error - self.prev_error)
                D_gain = self.kd_gain * (error - self.prev_error)
            self.prev_error = error

            # Combine PID components
            pid_output_exp = P_exp + I_exp + D_exp  
            pid_output_gain = P_gain + I_gain + D_gain

            # Calculate adjustments with clamping
            exp_adjust = np.clip(pid_output_exp, -self.exp_step, self.exp_step)
            gain_adjust = np.clip(pid_output_gain, -self.gain_step, self.gain_step)

            # Apply adjustments
            current_exp += exp_adjust
            current_gain += int(round(gain_adjust))
        return best_exp, best_gain


