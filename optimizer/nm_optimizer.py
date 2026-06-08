# Created: 2025-05-26  
# Author: Gongzhe Li
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from optimizer.utils import ImageNoiseMetric, ImageGradientMetric, ImageEntropyMetric, MixedImageMetric
from optimizer.utils import ImageFormationModel




class NelderMeadOptimizer:
    def __init__(self, image_formation_model, metric, bounds, max_iter=25, tol=1e-5):
        self.model = image_formation_model
        self.metric = metric
        self.bounds = bounds  # [(exp_min, exp_max), (gain_min, gain_max)]
        self.max_iter = max_iter
        self.tol = tol



    def optimize(self, image):
        # Objective: negative of the quality metric (since we minimize)
        def obj(x):
            # Apply bounds to exposure
            exp = np.clip(x[0], self.bounds[0][0], self.bounds[0][1])
            # Allow gain to float but clamp then round for evaluation
            gain_cont = np.clip(x[1], self.bounds[1][0], self.bounds[1][1])
            gain = int(round(gain_cont))
            gain = int(min(max(gain, self.bounds[1][0]), self.bounds[1][1]))
            # Form image and compute metric
            formed = self.model.simulate(image, exp_time=exp, analog_gain=gain)
            score = self.metric.evaluate(formed)
            # Negate because we want to maximize the metric
            return -score

        # Initial simplex: use midpoint and perturbations
        x0 = np.array([
            (self.bounds[0][0] + self.bounds[0][1]) / 2.0,
            (self.bounds[1][0] + self.bounds[1][1]) // 2.0
        ])
        n = len(x0)
        # simplex = np.zeros((n+1, n))
        # simplex[0] = x0
        # for i in range(n):
        #     y = np.array(x0, copy=True)
        #     5% of range as step size
            # y[i] = x0[i] + 0.05 * (self.bounds[i][1] - self.bounds[i][0])
            # simplex[i+1] = y
        low_exp = self.bounds[0][0]+4  # e.g. 1
        high_exp = self.bounds[0][1]  # e.g. 20
        low_gain = 4  # e.g. 0
        high_gain = int(self.bounds[1][1])  # e.g. 14
        mid_exp = 0.5 * (low_exp + high_exp)
        mid_gain = int(round((low_gain + high_gain) // 2.0))
        simplex = np.array([
            [low_exp, low_gain],
            [high_exp, high_gain],
            [mid_exp, mid_gain]
        ], dtype=float)

        # Evaluate objective at the simplex vertices
        fs = np.array([obj(simplex[i]) for i in range(n+1)])
        best_idx = np.argmin(fs)
        best_x = simplex[best_idx]
        best_val = fs[best_idx]

        # Nelder-Mead coefficients (common defaults)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for iteration in range(self.max_iter):
            # Sort vertices by objective value (ascending since we minimize)
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            # Identify best, worst, and second-worst points
            x_best = simplex[0]
            x_worst = simplex[-1]
            x_second = simplex[-2]
            # Centroid of all but worst
            centroid = simplex[:-1].mean(axis=0)

            # 1) Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            f_r = obj(x_r)

            if f_r < fs[0]:
                # 2) Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = obj(x_e)
                if f_e < f_r:
                    new_point, f_new = x_e, f_e
                else:
                    new_point, f_new = x_r, f_r
            else:
                if f_r < fs[-2]:
                    # Accept reflection
                    new_point, f_new = x_r, f_r
                else:
                    # 3) Contraction
                    if f_r < fs[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid + rho * (x_worst - centroid)
                    f_c = obj(x_c)
                    if f_c < fs[-1]:
                        new_point, f_new = x_c, f_c
                    else:
                        # 4) Shrink simplex towards the best point
                        new_simplex = np.zeros_like(simplex)
                        new_simplex[0] = simplex[0]
                        for j in range(1, n+1):
                            new_simplex[j] = x_best + sigma * (simplex[j] - x_best)
                        simplex = new_simplex
                        fs = np.array([obj(simplex[j]) for j in range(n+1)])
                        continue

            # Replace worst point with the new one
            simplex[-1] = new_point
            fs[-1] = f_new

            # Update best found
            current_best = np.min(fs)
            if current_best < best_val:
                best_val = current_best
                best_x = simplex[np.argmin(fs)]

            # Check convergence (std of values small)
            if np.std(fs) < self.tol:
                break

        # Return the best exposure and gain (with gain as integer) and its score
        best_exp = float(np.clip(best_x[0], self.bounds[0][0], self.bounds[0][1]))
        # best_gain = int(round(best_x[1]))
        best_gain = best_x[1]
        best_gain = (min(max(best_gain, self.bounds[1][0]), self.bounds[1][1]))
        return best_exp, best_gain







