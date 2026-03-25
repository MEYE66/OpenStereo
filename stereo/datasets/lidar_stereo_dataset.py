import cv2
import os
import sys
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt

try:
    from stereo.datasets.dataset_template import DatasetTemplate
except ModuleNotFoundError:
    # Allow running this file directly: python stereo/datasets/carla_stereo_dataset.py
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from stereo.datasets.dataset_template import DatasetTemplate


def _load_rectified_image(path):
    ext = Path(path).suffix.lower()
    if ext != '.npy':
        raise NotImplementedError('Only .npy rectified images are supported: ' + path)

    img = np.load(path).astype(np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]
    return img


def _load_points(path):
    ext = Path(path).suffix.lower()
    if ext != '.npy':
        raise NotImplementedError('Only .npy point clouds are supported: ' + path)

    points = np.load(path).astype(np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    elif points.ndim >= 2 and points.shape[-1] == 3:
        points = points.reshape(-1, 3)
    else:
        raise ValueError('Point cloud must be shape (N, 3) or (..., 3): ' + path)
    return points


def _transform_points(points, transform_mtx):
    points = points.reshape(-1, 3)
    points = points[(points[:, 0] != 0) | (points[:, 1] != 0)]
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    points_trans = transform_mtx @ points_homo.T
    return points_trans[:3].T


def _transform_points_inverse(points, transform_mtx):
    return _transform_points(points, np.linalg.pinv(transform_mtx))


def _project_points_on_camera(points, focal_length, cx, cy, image_width=0, image_height=0):
    points = points.copy()
    z = points[:, 2]
    valid_z = z > 0
    points = points[valid_z]

    points[:, 0] = points[:, 0] * focal_length / points[:, 2] + cx
    points[:, 1] = points[:, 1] * focal_length / points[:, 2] + cy

    if image_width > 0 and image_height > 0:
        points = points[
            (points[:, 0] >= 0)
            & (points[:, 0] <= image_width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= image_height - 1)
        ]
    return points


def _plot_lidar_points(points, image_left, save_path, vmax=20000):
    # Reference: ref_code/display.py::plot_lidar_points
    if image_left.ndim != 3:
        raise ValueError('left image must be HxWxC')

    vis_img = image_left.astype(np.float32)
    max_val = float(vis_img.max())
    if max_val > 1.0:
        vis_img = np.clip(vis_img / max_val, 0.0, 1.0)

    pts = points.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, vis_img.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, vis_img.shape[0] - 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(vis_img)
    ax[0].set_title('Left Rectified Image')
    ax[0].axis('off')

    ax[1].imshow(np.zeros((vis_img.shape[0], vis_img.shape[1])), cmap='gray')
    scatter = ax[1].scatter(pts[:, 0], pts[:, 1], s=0.5, c=pts[:, 2], cmap='magma', vmax=vmax)
    fig.colorbar(scatter, ax=ax[1], label='Depth (mm)')
    ax[1].set_title('Projected LiDAR Points')
    ax[1].axis('off')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _get_image_hw(image):
    if image.ndim != 3:
        raise ValueError('left image must be 3D, got shape: ' + str(image.shape))

    # Support both HWC (numpy) and CHW (after transpose/to-tensor style transforms).
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        return int(image.shape[1]), int(image.shape[2])
    return int(image.shape[0]), int(image.shape[1])


def _build_dense_valid_mask(points, height, width, depth_thres=15000.0):
    valid = np.zeros((height, width), dtype=bool)
    if hasattr(points, 'detach'):
        pts = points.detach().cpu().numpy()
    else:
        pts = np.asarray(points)
    if pts.size == 0:
        return valid
    pts = pts.reshape(-1, 3)
    mask = pts[:, 2] < depth_thres
    if not np.any(mask):
        return valid

    u = pts[:, 0][mask].astype(np.int64)
    v = pts[:, 1][mask].astype(np.int64)

    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid_uv):
        return valid

    valid[v[valid_uv], u[valid_uv]] = True
    return valid



class LidarStereoDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        # Fixed camera intrinsics requested for ADEC real lidar-stereo data.
        self.baseline = getattr(self.data_info, 'BASELINE', 110.0)
        self.focal_length = getattr(self.data_info, 'FOCAL_LENGTH', 1323.50)
        self.cx = getattr(self.data_info, 'CX', 684.0)
        self.cy = getattr(self.data_info, 'CY', 557.0)
        self.image_width = getattr(self.data_info, 'IMAGE_WIDTH', 1440)
        self.image_height = getattr(self.data_info, 'IMAGE_HEIGHT', 928)
        self.point_scale = getattr(self.data_info, 'POINT_SCALE', 1000.0)
        self.valid_depth_thres = getattr(self.data_info, 'VALID_DEPTH_THRES', 15000.0)
        self.transform_mtx = np.array([
            [9.74168269e-01, -2.16619390e-02, -2.24781992e-01, 3.68182351e+01],
            [2.23991311e-01, -3.38457838e-02, 9.74003263e-01, 2.71851960e+02],
            [-2.87067220e-02, -9.99192285e-01, -2.81194025e-02, -2.35719906e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ], dtype=np.float32)

        self.training = (mode == 'training')

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, points_path = full_paths

        if self.training:

            left_img = _load_rectified_image(left_path)
            right_img = _load_rectified_image(right_path)

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        points = _load_points(points_path) * self.point_scale
        points = _transform_points_inverse(points, self.transform_mtx)
        points = _project_points_on_camera(
            points,
            focal_length=self.focal_length,
            cx=self.cx,
            cy=self.cy,
            image_width=self.image_width,
            image_height=self.image_height,
        )

        sample = {
            'left': left_img,
            'right': right_img,
            'points': points,
            'focal_length': self.focal_length,
            'baseline': self.baseline
        }
        if self.transform is not None:
            sample = self.transform(sample)

        h, w = _get_image_hw(sample['left'])
        sample['valid'] = _build_dense_valid_mask(
            sample['points'],
            height=h,
            width=w,
            depth_thres=self.valid_depth_thres,
        )
        sample['index'] = idx
        sample['name'] = left_path
        return sample


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test LidarStereoDataset')
    parser.add_argument('--data_root', type=str, default="/home/lgz/dataset/ADEC/real", help='Root directory of the dataset')
    parser.add_argument('--split_file', type=str, default="/home/lgz/workspace/OpenStereo/dataset_split/lidar_stereo/train~.txt", help='Path to the split file')
    parser.add_argument('--vis_out', type=str, default='lidar_points_check.png', help='Output path for lidar projection validation image')
    args = parser.parse_args()

    data_info = SimpleNamespace(
        DATA_PATH=args.data_root,
        DATA_SPLIT={
            'TRAINING': args.split_file,
            'EVALUATING': args.split_file,
            'TESTING': args.split_file,
        },
        BASELINE=110.0,
        FOCAL_LENGTH=1323.50,
        CX=684.0,
        CY=557.0,
        IMAGE_WIDTH=1440,
        IMAGE_HEIGHT=928,
        POINT_SCALE=1000.0,
        VALID_DEPTH_THRES=15000.0,
    )
    data_cfg = SimpleNamespace(
        DATA_TRANSFORM={
            'TRAINING': [],
            'EVALUATING': [],
            'TESTING': [],
        }
    )

    dataset = LidarStereoDataset(data_info=data_info, data_cfg=data_cfg, mode='training')
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[200]
    print('Sample keys:', sample.keys())
    print('Left image shape:', sample['left'].shape, sample['left'].min(), sample['left'].max())
    print('Right image shape:', sample['right'].shape, sample['right'].min(), sample['right'].max())
    print('Points shape:', sample['points'].shape)
    print('Valid mask shape:', sample['valid'].shape, sample['valid'].dtype, sample['valid'].mean())
    print('Focal length:', sample['focal_length'])
    print('Baseline:', sample['baseline'])

    _plot_lidar_points(sample['points'], sample['left'], args.vis_out)
    print('Saved lidar projection visualization to:', args.vis_out)
    
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for batch in dataloader:
    #     print('Batch keys:', batch.keys())
    #     print('Left image shape:', batch['left'].shape)
    #     print('Right image shape:', batch['right'].shape)
    #     print('Disparity shape:', batch['disp'].shape)
    #     print('Occ mask shape:', batch['occ_mask'].shape)
    #     break



