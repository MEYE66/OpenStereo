# Created: 2025-06-10
# Author: Gongzhe Li
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
import pyexr
from optimizer.utils import np_to_image
from simulator.scene_simulate import scene_adjust_dynamic_range, scene_simulate, apply_si_psf


BIT8 = 2 ** 8
BIT16 = 2 ** 16
BIT24 = 2 ** 24

def minmax_norm(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)).astype(np.float32)  # Normalize to [0, 1]
    return image


class Worker():
    def __init__(self, root_path, result_path):
        self.scene_list = ['headlights', 'streetlights', 'otherlights', 'skymap']

        self.root_path = root_path
        self.result_path = result_path
        self.id_list = os.listdir(self.root_path)

    def run(self, scene_path, result_path):
        id_name = os.path.basename(scene_path)
        scenes = []
        for id in self.scene_list:
            exr_path = os.path.join(self.root_path, id_name, id_name + f"_{id}" + ".exr")
            try:
                exr_data = pyexr.read(exr_path, precision=pyexr.HALF)[:, :, :3] # type: ignore
            except pyexr.exr.ExrError as e:
                print(f"Error reading {exr_path}: {e}")
                continue
            if id == 'otherlights':
                # r g b
                exr_data[:, :, 1] = exr_data[:, :, 0]
            exr_data = np.clip(exr_data.astype(np.float32), 0, None)
            scenes.append(exr_data)
        hdr_weights = scene_adjust_dynamic_range(scenes, 3e3)
        irradiance = scene_simulate(scenes, hdr_weights, capacity=1.5e3).astype(np.float32)
        irradiance = np.clip(irradiance, 0, None)
        # irradiance = apply_si_psf(irradiance, aperture_size=2048, radius=512)
        # irradiance = cv2.resize(irradiance, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # irradiance = minmax_norm(irradiance)
        # irradiance = np.clip(irradiance * (BIT24-1), 0, (BIT24-1)).astype(np.int32)
        result_path = os.path.join(result_path, f"{id_name}.tiff") 
        cv2.imwrite(result_path, irradiance)
        


def main():
    # Gradient AE
    # Mixed AE
    # Neural AE
    folder_list = [
        'ISETScene_001_renderings',
        'ISETScene_002_renderings',
        'ISETScene_003_renderings',
        'ISETScene_004_renderings',
        'ISETScene_005_renderings',     
        'ISETScene_006_renderings',
        'ISETScene_007_renderings',
        'ISETScene_008_renderings',
        'ISETScene_009_renderings',
        'ISETScene_010_renderings',
        'ISETScene_011_renderings',
        'ISETScene_012_renderings',
        ]

    for folder in folder_list:
        print(f"Processing folder: {folder}")
        src_path = f"/home/ligongzhe/data/ISET/HDRScene/{folder}"
        dst_path = f"/home/ligongzhe/data/ISET/HDRDataset4/{folder}" # [process lights groups(.exr) into hdr(.tiff) ]
        # scene_list = ['headlights', 'streetlights', 'otherlights', 'skymap']
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        

        # fibonacci exposure 
        worker = Worker(root_path=src_path, result_path=dst_path,)
        image_folders = [(os.path.join(src_path, i)) for i in os.listdir(src_path)]
        print(f"totally {len(image_folders)} folders")
        threads = multiprocessing.cpu_count() // 4
        # threads = 1
        print(f'{threads} threads')
        para = Parallel(n_jobs=threads, backend='loky')
        para(delayed(worker.run)(scene_path, dst_path) for scene_path in tqdm(image_folders))



if __name__ == '__main__':
    main()