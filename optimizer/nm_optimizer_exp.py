# Created: 2025-05-26  
# Author: Gongzhe Li
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import matplotlib.pyplot as plt
import numpy as np
from optimizer.utils import ImageNoiseMetric, ImageGradientMetric, ImageEntropyMetric, MixedImageMetric


class NelderMeadOptimizerExposure:
    """
    Modified NM-Optimizer for exposure constraint
    exp_t2 / exp_t1 = g
    in each iteration, we update [exp_t2, exp_t1, g]
    """
    def __init__(self, image_formation_model, metric, bounds, max_iter=25, tol=1e-5, g=2):
        self.model = image_formation_model
        self.metric = metric
        self.bounds = bounds  # [(exp_min, exp_max), (gain_min, gain_max)]
        self.max_iter = max_iter
        self.tol = tol
        self.g = g

    def optimize(self, image):
        # Objective: negative of the quality metric (since we minimize)
        def obj(x):
             # Apply bounds to exposure
            exp_t1 = np.clip(x[0], self.bounds[0][0], self.bounds[0][1])
            exp_t2 = exp_t1 * self.g

            # Enforce exp_t2 bounds
            if exp_t2 < self.bounds[0][0] or exp_t2 > self.bounds[0][1]:
                return np.inf  # Penalize infeasible exposure
            
            
            # Allow gain to float but clamp then round for evaluation
            gain_cont = np.clip(x[1], self.bounds[1][0], self.bounds[1][1])
            gain = int(round(gain_cont))
            gain = int(min(max(gain, self.bounds[1][0]), self.bounds[1][1]))
            # Form image and compute metric
            formed = self.model.subframes_fusion(image, exp_time=[exp_t1, exp_t2], analog_gain=gain)
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
        low_exp = self.bounds[0][0]  # e.g. 1
        high_exp = self.bounds[0][1]  # e.g. 20
        low_gain = 0  # e.g. 0
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
        best_exp1 = np.clip(best_x[0], self.bounds[0][0], self.bounds[0][1])
        best_exp2 = self.g * best_exp1
        best_exp = [best_exp1, best_exp2]
        
        
        best_gain = int(round(best_x[1]))
        best_gain = int(min(max(best_gain, self.bounds[1][0]), self.bounds[1][1]))
        # best_img = self.model.subframes_fusion(image, best_exp, best_gain)
        # best_score = self.metric.evaluate(best_img)
        return best_exp, best_gain



# class GradientOptimizer:
#     """
#     Gradient-based optimizer for AEC.
#     Auto-adjusting Camera Exposure for Outdoor Robotics using Gradient Information
#     """
#     def __init__(self, image_formation_model, metric, bounds, max_iter=10, tol=1e-5):
#         self.model = image_formation_model
#         self.metric = metric
#         self.bounds = bounds  # [(exp_min, exp_max), (gain_min, gain_max)]
#         self.max_iter = max_iter
#         self.tol = tol

#     def optimize(self, image):
#         return





if __name__ == '__main__':


    opt = {
        'large_pixel': {'conversion_gain': 1 / 13.5, 'circuit_gain': 1.88, 'dark_current': 59, 'readout_noise': 27.23,
                        'quantization': 12},
        'small_pixel': {'conversion_gain': 1 / 6.9, 'circuit_gain': 1.56, 'dark_current': 23, 'readout_noise': 22.18,
                        'quantization': 12},
        'analog_gain_path': '/home/gongzheli/workspace/HDRSimulator/diffSimulator/gain.yml'
    }
    image_formation_model = ImageFormationModel(opt)


    # root_path = "/mnt/data1/hdr_sim/ISET/ISETScene/ISETScene_001_renderings"
    root_path = "/home/ligongzhe/data/ISET/HDRDataset/ISETScene_001_renderings"
    # id_name = "1113063300"
    # id_name = "1113062800"
    # id_name = "1113094119"
    # id_name = "1113105332"
    # id_name = "1113095459" # (17.75591865181923, 13)
    # id_name = "1113072023" #  (20.0, 14)
    # id_name = "1113075626" #  (10.222077891230583, 7)
    # id_name = "1113051416" # (10.24885029764846, 7)
    # id_name = "1113070448" #  (5.2191619873046875, 6)
    id_name = "1112154152"
    # id_name = "1113100547" # (14.852349497377872, 11)
    # scene_list = ['headlights', 'streetlights', 'otherlights', 'skymap']
    # scenes = []
    # for id in scene_list:
    #     exr_path = os.path.join(root_path, id_name, id_name + f"_{id}" + ".exr")
    #     exr_data = pyexr.read(exr_path, precision=pyexr.HALF)[:, :, :3]
    #     if id == 'otherlights':
    #         # r g b
    #         exr_data[:, :, 1] = exr_data[:, :, 0]
    #     exr_data = np.clip(exr_data.astype(np.float32), 0, None)
    #     scenes.append(exr_data)

    # hdr_weights = scene_adjust_dynamic_range(scenes, 1e3)
    # irradiance = scene_simulate(scenes, hdr_weights, capacity=3e3)
    # image_example = image_formation_model.subframes_fusion(irradiance, 10, 5)
    # # noise test
    # noise = np.random.normal(3, 50, img.shape).astype(np.float32)
    # img = img + noise

    # image_example = irradiance
    # image_example = ((image_example - image_example.min()) / (image_example.max() - image_example.min()))
    # plt.imshow(image_example)
    # plt.show()
    # exit(234)


    # noise_metric = ImageNoiseMetric()
    # gradient_metric = ImageGradientMetric()
    # entropy_metric = ImageEntropyMetric()
    irradiance = cv2.imread(os.path.join(root_path, f"{id_name}.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    irradiance = np.clip(irradiance, 0, None)
    # metric = MixedImageMetric()
    metric = ImageEntropyMetric()
    # print(noise_metric.evaluate(image_example))

    # NelderMeadOptimizerExposure
    # ImageEntropyMetric

    optimizer = NelderMeadOptimizerExposure(image_formation_model, metric, bounds=[(1, 20), (0, 14)], max_iter=30, tol=1e-5)
    best_params, best_score, best_img = optimizer.optimize(irradiance)
    print("Best exposure,gain:", best_params, "Quality score:", best_score)
    # plt_img(best_img)


