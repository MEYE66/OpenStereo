# Created: 2025-06-03  
# Author: Gongzhe Li
import os
import sys
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.pid_optimizer import PIDOptimizer
from optimizer.nm_optimizer import NelderMeadOptimizer
from optimizer.nm_optimizer_exp import NelderMeadOptimizerExposure
from optimizer.utils import MixedImageMetric, ImageSemanticMetric, ImageEntropyMetric, ImageGradientMetric, ImageContrastMetric
from simulator.np_simulate import ImageFormationModel


# camera_opt = {
#         'large_pixel': {'conversion_gain': 1 / 10.5, 'circuit_gain': 1.28, 'dark_current': 26, 'readout_noise': 21.23,
#                         'quantization': 12},
#         'small_pixel': {'conversion_gain': 1 / 7.9, 'circuit_gain': 1.56, 'dark_current': 18, 'readout_noise': 15.18,
#                         'quantization': 12},
#     }
camera_opt = {
        'large_pixel': {'conversion_gain': 1 / 15.5, 'circuit_gain': 1.88, 'dark_current': 33, 'readout_noise': 27.23,
                        'quantization': 12},
        'small_pixel': {'conversion_gain': 1 / 6.9, 'circuit_gain': 1.56, 'dark_current': 10, 'readout_noise': 22.18,
                        'quantization': 12},
    }

BIT8 = 256
BIT12 = 4096
BIT16 = 65535




class Worker():
    def __init__(self, optimizer, image_model, metric_model, root_path, result_path, opt):
        self.image_model = image_model(opt)
        self.metric_model = metric_model()
        self.optimizer = optimizer(self.image_model, self.metric_model, bounds=[(1, 30), (1, 14)], max_iter=30, tol=1e-5)

        self.root_path = root_path
        self.result_path = result_path
        self.id_list = os.listdir(self.root_path)
        
        # self.expos = 10
        # self.gains = 10

    def run(self, scene_path, result_path):
        # scene_name = os.path.basename(scene_path)
        # id_name = scene_name.split('.')[0]
    
        # depth_path = os.path.join(depth_path, f"{id_name}_depth.exr")
        path, id_name = os.path.split(scene_path)
        scene_path = os.path.join(path, f"{id_name}")
        file_name = id_name.split('-')[0]
        # print(f"Processing scene: {scene_path}")
        # print(f"file_nmae: {file_name}, id_name: {id_name}")
        # exit(234)
        
        
        # print(f"Processing scene: {scene_path}")
        # print(f"Depth path: {depth_path}")
        # scenes = []
        # for id in self.scene_list:
        #     exr_path = os.path.join(self.root_path, id_name, id_name + f"_{id}" + ".exr")
        #     exr_data = pyexr.read(exr_path, precision=pyexr.HALF)[:, :, :3] # type: ignore
        #     if id == 'otherlights':
        #         # r g b
        #         exr_data[:, :, 1] = exr_data[:, :, 0]
        #     exr_data = np.clip(exr_data.astype(np.float32), 0, None)
        #     scenes.append(exr_data)
        # hdr_weights = scene_adjust_dynamic_range(scenes, 3e3)
        # irradiance = scene_simulate(scenes, hdr_weights, capacity=1e3)
        # input_path = os.path.join(scene_path, f"{id_name}.tiff")
        # print(f"load image from {input_path}")
        # print(f"Processing {id_name}...")
        irradiance = cv2.imread(scene_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        irradiance = np.clip(irradiance, 0, None)
        if file_name == 'day':
            irradiance = irradiance / 2e1
        irradiance_ds = cv2.resize(irradiance, (256, 256), interpolation=cv2.INTER_LINEAR)
        try:
            best_exp, best_gain = self.optimizer.optimize(image=irradiance_ds)
            best_img = self.image_model.subframes_fusion(irradiance, best_exp, best_gain)
            # print(f"img range:{best_img.min()} ~ {best_img.max()}")
            # exit(234)
            # best_img = (best_img - best_img.min()) / (best_img.max() - best_img.min()) 
            best_img = np.clip(best_img * (BIT12-1), 0, None).astype(np.int32)
            print(f"{result_path}, {id_name}")
            cv2.imwrite(os.path.join(result_path, f"{id_name}"), best_img)
            # cv2.imwrite(os.path.join(result_path, f"{id_name}_right.png"), np_to_image(right_crop))
            result_file = os.path.join(result_path, f"{id_name}_best_params.txt")
            # exit(234)
            with open(result_file, "w") as f:
                f.write(f"best_exp: {best_exp}, best_gain: {best_gain}\n")
        except Exception as e:
            print(f"Error in optimizing {id_name}: {e}")
            # best_params, best_score, best_image = None, None, None
            error_path = os.path.join(result_path, f"{id_name}_error.txt")
            # if not os.path.exists(error_path):
            #     os.makedirs(error_path)
            with open(error_path, "w") as f:
                f.write(f"Error in optimizing {id_name}")
        # 保存最优参数和分数到txt文件


def main(mode):
    # PID AE
    # Fibonicc AE
    # Mixed AE
    # Semantic AE
    # Fixed AE

    
   
    # print(f"Processing folder: {folder}")

    src_path = f"/home/ligongzhe/data/HDRDataset/RAWtiff2" # [Pid AE] [Mixed AE] [Semantic AE] [Fibo AE]  [Neural AE]
    dst_path = f"/home/ligongzhe/data/HDRDataset/AECompare/{mode}" # [Pid AE] [Mixed AE] [Semantic AE] [Fibo AE]  [Neural AE]
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    if mode == 'pid':
    # semantic ae
        worker = Worker(optimizer=NelderMeadOptimizer, image_model=ImageFormationModel, metric_model=ImageEntropyMetric, root_path=src_path, result_path=dst_path, opt=camera_opt)
    # gradient metric not used in this experiments
    # worker = Worker(optimizer=NelderMeadOptimizer, image_model=ImageFormationModel, metric_model=ImageGradientMetric, root_path=src_path, result_path=dst_path, opt=camera_opt)
    elif mode == 'fibo':
        # fibonacci exposure
        worker = Worker(optimizer=NelderMeadOptimizerExposure, image_model=ImageFormationModel, metric_model=ImageEntropyMetric, root_path=src_path, result_path=dst_path, opt=camera_opt)
    elif mode == 'mixed':
        # mixed ae
        worker = Worker(optimizer=NelderMeadOptimizer, image_model=ImageFormationModel, metric_model=MixedImageMetric, root_path=src_path, result_path=dst_path, opt=camera_opt)        
    elif mode == 'semantic':
        # pid optimizer
        worker = Worker(optimizer=NelderMeadOptimizer, image_model=ImageFormationModel, metric_model=ImageSemanticMetric, root_path=src_path, result_path=dst_path, opt=camera_opt)

    elif mode == 'fix':
        worker = Worker(optimizer=PIDOptimizer, image_model=ImageFormationModel, metric_model=ImageContrastMetric, root_path=src_path, result_path=dst_path, opt=camera_opt)
    
    image_folders = [(os.path.join(src_path, i)) for i in os.listdir(src_path)]
    print(f"totally {len(image_folders)} folders")
    threads = multiprocessing.cpu_count() // 4
    # threads = 1
    print(f'{threads} threads')
    para = Parallel(n_jobs=threads, backend='loky')
    para(delayed(worker.run)(scene_path, dst_path) for scene_path in tqdm(image_folders))


if __name__ == '__main__':
    # mode_list = ['pid', 'fibo', 'mixed', 'semantic']  # 'pid', 'semantic', 'fibonacci', 'mixed', 
    mode_list = ['semantic']
    for mode in mode_list:
        print(f"Starting optimization with mode: {mode}")
        main(mode)

        
