# Created: 2025-05-27  
# Author: Gongzhe Li
import cv2
import os
import math
import numpy as np
from scipy.stats import entropy as shannon_entropy
import matplotlib.pyplot as plt
import thop

try:
    import cv2.saliency  # type: ignore[attr-defined]
    _HAS_CV2_SALIENCY = True
except Exception:
    _HAS_CV2_SALIENCY = False

# from simulator.img_util import plt_img, plt_hist
import torch



# ------------------  Dataset and Image Utilities ------------------ #
def minmax_norm(image):
    """Normalize the image to the range [min_val, max_val]."""
    image = (image - np.min(image)) / (np.max(image) - np.min(image)).astype(np.float32)
    return image

def inverse_tmo(ldr_image, mu:float=1000.0, l_max: float = 3000.0)->np.ndarray:
    ldr = np.clip(ldr_image, 0, 1)
    hdr = l_max * (((1.0 + mu) ** ldr - 1.0) / mu)
    return hdr.astype(np.float32)



def radiance_scale(radiance, capacity=1.0):
    mean_val = np.mean(radiance)
    scale = capacity / (mean_val + 1e-8)
    return radiance * scale





def thop_profile(model, input_size):
    dummy_input = torch.randn(1, 3, *input_size).to(next(model.parameters()).device)
    macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
    print(f"MACs: {macs / 1e6:.2f} M, Params: {params / 1e6:.2f} M")
    # return macs, params






def np_to_image(img_np: np.ndarray, rgb2bgr=False) -> np.ndarray:
    """
    Convert a numpy array to an image format.
    :param img_np: Input image as a numpy array.
    :param rgb2bgr: Whether to convert RGB to BGR.
    :return: Converted image.
    """
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)).astype(np.float32)  # Normalize to [0, 1]
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    if rgb2bgr and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    return img_np



# def np_to_image_u16(img_np: np.ndarray, rgb2bgr=False) -> np.ndarray:
#     """
#     Convert a numpy array to an image format.
#     :param img_np: Input image as a numpy array.
#     :param rgb2bgr: Whether to convert RGB to BGR.
#     :return: Converted image.
#     """
#     bit_depth = (2 ** 16) - 1
#     img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)).astype(np.float32)  # Normalize to [0, 1]
#     img_np = np.clip(img_np * bit_depth, 0, bit_depth).astype(np.uint16)  # Scale to [0, 65535] and convert to uint16
#     if rgb2bgr and img_np.shape[2] == 3:
#         img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
#     return img_np





# ----------------------- Image Formation Model ----------------------- #
def _make_odd_int(value, mode="ceil"):
    value = math.ceil(float(value)) if mode == "ceil" else math.floor(float(value))
    value = max(1, value)
    if value % 2 == 0:
        value = value + 1 if mode == "ceil" else value - 1
    return max(1, value)


class ImageFormationModel:
    def __init__(
        self,
        time_limits=(1.0, 20.0),
        gain_limits=(1, 20),
        init_exposure=12.5,
        init_gain=10,
        nbits=8,
        gaussian_var=3e-5,
        poisson_scale=3.3e-4,
        seed=None,
        motion_blur=True,
        motion_angle_deg=10.0,
        motion_blur_params=None,
    ):
        self.time_limits = time_limits
        self.gain_limits = gain_limits
        self.init_exposure = init_exposure
        self.init_gain = init_gain
        self.nbits = int(nbits)
        self.gaussian_var = float(gaussian_var)
        self.poisson_scale = float(poisson_scale)
        self.rng = np.random.default_rng(seed)
        self.motion_blur = bool(motion_blur)
        self.motion_angle_deg = float(motion_angle_deg)
        self.motion_blur_params = {
            'velocity_scale': 0.8,
            'min_length': 1,
            'max_length': 13,
            'canvas_size': 31,
            'line_sigma': 0.55,
            'edge_softness': 0.75,
            'eps': 1e-8,
        }
        if motion_blur_params is not None:
            self.motion_blur_params.update(motion_blur_params)
        self.motion_psf = None
        self.motion_psf_info = None

    def _normalize_psf(self, kernel):
        kernel_sum = np.sum(kernel, dtype=np.float32)
        return kernel / np.maximum(kernel_sum, self.motion_blur_params['eps'])

    def _exposure_time_to_kernel_size(self, exposure_time_ms):
        params = self.motion_blur_params
        min_length = _make_odd_int(params['min_length'], mode="ceil")
        max_length = _make_odd_int(max(params['max_length'], min_length), mode="floor")
        length = int(np.ceil(float(exposure_time_ms) * float(params['velocity_scale'])))
        length = int(np.clip(length, min_length, max_length))
        if length % 2 == 0:
            length += 1
        if length > max_length:
            length -= 2
        return int(np.clip(length, min_length, max_length))

    def _generate_motion_blur_kernel(self, exposure_time_ms, angle_deg):
        params = self.motion_blur_params
        length = self._exposure_time_to_kernel_size(exposure_time_ms)
        canvas_size = _make_odd_int(max(params['canvas_size'], params['max_length']), mode="ceil")
        radius = canvas_size // 2
        coord = np.arange(-radius, radius + 1, dtype=np.float32)
        yy, xx = np.meshgrid(coord, coord, indexing='ij')

        theta = np.deg2rad(np.float32(angle_deg))
        cos_t = np.cos(theta, dtype=np.float32)
        sin_t = np.sin(theta, dtype=np.float32)
        x_parallel = xx * cos_t + yy * sin_t
        y_perp = -xx * sin_t + yy * cos_t

        half_len = max(float(length), 1.0) * 0.5
        line_profile = np.exp(-0.5 * (y_perp / float(params['line_sigma'])) ** 2, dtype=np.float32)
        endpoint_profile = 1.0 / (1.0 + np.exp(-(half_len - np.abs(x_parallel)) / float(params['edge_softness'])))
        kernel = line_profile * endpoint_profile
        return self._normalize_psf(kernel.astype(np.float32)), length

    def _apply_exposure_motion_blur(self, radiance, exposure_time_ms):
        if not self.motion_blur:
            return radiance

        kernel, motion_length = self._generate_motion_blur_kernel(exposure_time_ms, self.motion_angle_deg)
        self.motion_psf = kernel
        self.motion_psf_info = {
            'exposure_time_ms': float(exposure_time_ms),
            'angle_deg': float(self.motion_angle_deg),
            'kernel_size': int(kernel.shape[-1]),
            'length_px': motion_length,
        }
        return np.stack(
            [
                cv2.filter2D(radiance[:, :, channel_idx], -1, kernel, borderType=cv2.BORDER_REFLECT_101)
                for channel_idx in range(radiance.shape[2])
            ],
            axis=2,
        )

    def _quantize(self, image):
        max_val = float((2 ** self.nbits) - 1)
        output = np.clip(np.floor(image + 0.5), 0.0, max_val).astype(np.float32)
        out_min = float(output.min())
        out_max = float(output.max())
        if out_max - out_min < 1e-8:
            return np.zeros_like(output, dtype=np.float32)
        return (output - out_min) / (out_max - out_min)

    def __call__(self, radiance, exp_time, gain):
        radiance = np.asarray(radiance, dtype=np.float32)

        t_pred = float(exp_time)
        g_pred = float(gain)

        gauss_std = np.float32(np.sqrt(self.gaussian_var) * t_pred)
        poisson_scale = np.float32(max(self.poisson_scale * t_pred, 1e-8))

        radiance = self._apply_exposure_motion_blur(radiance, t_pred)
        radiance_t = radiance * t_pred
        poisson_lambda = np.clip(radiance_t / poisson_scale, 0.0, None)

        shot_noise = self.rng.poisson(poisson_lambda).astype(np.float32) * poisson_scale * g_pred
        readout_noise = gauss_std * self.rng.standard_normal(size=radiance.shape).astype(np.float32) * g_pred
        adc_noise = gauss_std * self.rng.standard_normal(size=radiance.shape).astype(np.float32)

        noise_radiance = shot_noise + readout_noise + adc_noise
        noise_radiance = np.clip(noise_radiance, 0.0, None)
        noise_radiance = self._quantize(noise_radiance)
        return noise_radiance

    def simulate(self, image, exp_time, analog_gain):
        return self.__call__(image, exp_time=exp_time, gain=analog_gain)






# ----------------------- Image Quality Metrics ----------------------- #

### for 12-bit image([0, 4095])
class ImageNoiseMetric:
    """
    Estimate noise level using homogeneous + unsaturated region masking
    and the Immerkaer Laplacian-based estimator.
    """
    def __init__(self, p=0.10, T_lower=0.05, T_upper=0.92):
        self.p = p

        self.T_lower = T_lower
        self.T_upper = T_upper
        # 3×3 Laplacian kernel for Immerkaer estimator
        self._Ng = np.array([[1, -2,  1],
                             [-2,  4, -2],
                             [1, -2,  1]], dtype=np.float32)

    def evaluate(self, img_np: np.ndarray) -> float:
        H, W, C = img_np.shape
        noise_vals = np.zeros(C, dtype=np.float32)

        for c in range(C):
            channel = img_np[:, :, c].astype(np.float32)
            # 1) Homogeneous region mask using Sobel gradient magnitude
            gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            # grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())  # Normalize to [0, 1]
            # grad_mag = grad_mag / grad_mag.max()  # Normalize to [0, 1]

            grad_1d = grad_mag.flatten()
            sort_grad_1d = np.sort(grad_1d)
            index = int(H * W * self.p)
            Gth = sort_grad_1d[index]  # Threshold for homogeneous region
            # Create homogeneous region mask
            homog_mask = (grad_mag <= Gth)
            # 2) Unsaturated mask for input image
            unsat_mask = ((channel >= self.T_lower) & (channel <= self.T_upper))
            # 3) Combined mask
            mask = homog_mask & unsat_mask
            # plt_img(mask.astype(np.float32))
            # # exit(234)
            # 4) Laplacian filtering
            lap = cv2.filter2D(channel, cv2.CV_32F, self._Ng)
            masked_lap = lap[mask]
            # plt_img(channel.astype(np.float32))
            # exit(234)
            Ns = masked_lap.size
            # print(f"valid pixels: {Ns}, channel: {c}, Gth: {Gth:.4f}")
            if Ns == 0:
                noise_vals[c] = 0.0  # or float('nan') if you want to flag it
            else:
                noise_vals[c] = (np.sqrt(np.pi / 2) / (6 * Ns)) * np.sum(np.abs(masked_lap))

        # float(flat_vals.mean() / (flat_vals.std() + 1e-6))  # avoid div-by-zero
        return np.mean(noise_vals / np.max(noise_vals))





class ImageGradientMetric:
    """
    Compute gradient-based index:
      - Sobel gradient magnitude → normalize & log‐map
      - Divide image into a num×num grid → compute mean per cell
      - Return ratio of mean/std of those grid values
    """
    def __init__(self, Lambda=1e3, Gamma=0.3, num=16):
        self.Lambda = Lambda
        self.Gamma = Gamma
        self.num = num

    def evaluate(self, img_np: np.ndarray) -> float:
        """
        :param img_np: range is [0, 1.], so the gradient is also in [0, 1.]
        :return:
        """

        gray = cv2.cvtColor(img_np.astype(np.float32), cv2.COLOR_RGB2GRAY)
        H, W = gray.shape

        # Ensure H and W are divisible by self.num
        H_crop = (H // self.num) * self.num
        W_crop = (W // self.num) * self.num
        gray = gray[:H_crop, :W_crop]

        # Sobel gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        G_norm = grad_mag

        # plt_img(G_norm)
        # exit(23)
        # Normalize and log mapping
        # G_norm = grad_mag / np.max(grad_mag)  # Normalize to [0, 1]
        Ng_val = np.log(self.Lambda * (1 - self.Gamma) + 1.0)

        mapped = np.zeros_like(G_norm, dtype=np.float32)
        valid_mask = G_norm >= self.Gamma
        mapped[valid_mask] = (
            np.log(self.Lambda * (G_norm[valid_mask] - self.Gamma) + 1.0) + 1e-7
        ) / Ng_val
        # Grid pooling using reshape
        h_step, w_step = H_crop // self.num, W_crop // self.num
        mapped_grid = mapped.reshape(self.num, h_step, self.num, w_step)
        block_means = mapped_grid.mean(axis=(1, 3))  # shape: (num, num)

        flat_vals = block_means.flatten()
        # return float(flat_vals.mean() / (flat_vals.std() + 1e-6))  # avoid div-by-zero
        return np.mean(flat_vals/flat_vals.max())



class ImageContrastMetric:
    def __init__(self):
        # Sigmoid函数的参数，控制归一化曲线形状
        self.alpha = 10.0  # 对比度敏感度
        self.beta = 0.2  # 中等对比度阈值

    def evaluate(self, img_np: np.ndarray):
        # 确保输入在[0,1]范围内
        img = (img_np - img_np.min()) / (img_np.max() - img_np.min()).astype(np.float32)
        # img = np.clip(img_np, 0.0, 1.0)
        # 转换为灰度图 (保持float32类型)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 计算x和y方向的Sobel梯度
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 计算梯度幅值
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 计算梯度幅值的标准差作为对比度度量
        contrast_std = np.std(magnitude)

        # 使用Sigmoid函数归一化到[0,1]范围
        normalized_contrast = 1 / (1 + np.exp(-self.alpha * (contrast_std - self.beta)))
        return normalized_contrast




class ImageEntropyMetric:
    """
    Compute Shannon-entropy of the grayscale histogram (256 bins),
    then scale by Ke.
    """
    def __init__(self, weight=0.125):
        self.weight = weight
        pass
    def evaluate(self, img_np: np.ndarray):
        gray = cv2.cvtColor((img_np).astype(np.float32), cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray.flatten(), bins=256, density=True)
        return shannon_entropy(hist) * self.weight



def minmax_norm(image):
    """Normalize the image to the range [min_val, max_val]."""
    image = (image - np.min(image)) / (np.max(image) - np.min(image)).astype(np.float32)
    return image


class ImageSemanticMetric():
    def __init__(self,):
        # self.saliency_func = cv2.saliency.StaticSaliencySpectralResidual_create()
        # StaticSaliencyFineGrained_create
        pass
    def compute_saliency(self, image):
        if not _HAS_CV2_SALIENCY:
            gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
            return np.clip(minmax_norm(gray) * 255, 0, 255).astype(np.uint8)
        saliency_func = cv2.saliency.StaticSaliencyFineGrained_create()
        flag, saliency_map = saliency_func.computeSaliency(image.astype(np.float32))
        saliency_map = np.clip(minmax_norm(saliency_map)*255, 0, 255).astype(np.uint8)  # Ensure values are in [0, 1]
        thresh_map = cv2.threshold(saliency_map, 20, 250, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return thresh_map

    def evaluate(self, image):
        saliency_map = self.compute_saliency(image)
        saliency_map = np.repeat(saliency_map[:, :, np.newaxis], 3, axis=2)
        image_saliency = saliency_map * image
        image_hist, _ = np.histogram(image_saliency.flatten(), bins=512, density=True)
        image_entropy = shannon_entropy(image_hist)
        return image_entropy



# github copilot: IROS 2019
class GradientImageMetric():
    """IROS 2019
    Camera Exposure Control for Robust Robot Vision with Noise-Aware Image Quality Assessment
    Gradient-based image metric for AEC
    """
    def __init__(self, Lambda=1e3, Gamma=0.06, num=10):
        self.metric = ImageGradientMetric(Lambda, Gamma, num)

    def evaluate(self, img_np: np.ndarray) -> float:
        return self.metric.evaluate(img_np)


class MixedImageMetric():
    """IROS 2019
    Camera Exposure Control for Robust Robot Vision with Noise-Aware Image Quality Assessment
    Mixed image metric for AEC
    """
    def __init__(self, gradient_weight=0.5, entropy_weight=0.5, noise_weight=-0.4):
        self.noise_metric = ImageNoiseMetric()
        self.gradient_metric = ImageGradientMetric()
        self.entropy_metric = ImageEntropyMetric()
        self.noise_weight = noise_weight
        self.gradient_weight = gradient_weight
        self.entropy_weight = entropy_weight
    def evaluate(self, img_np: np.ndarray) -> float:
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min())).astype(np.float32)
        noise_value = self.noise_metric.evaluate(img_np)
        gradient_value = self.gradient_metric.evaluate(img_np)
        entropy_value = self.entropy_metric.evaluate(img_np)

        return (self.noise_weight * noise_value +
                self.gradient_weight * gradient_value +
                self.entropy_weight * entropy_value)



if __name__ == '__main__':

    # Example usage
    # name = "1113082349"
    # name = "1113070448"
    # name = "1113082349"
    img = cv2.imread(f'/home/ligongzhe/data/ISET/LDRDataset/1113082349.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    noise = np.random.normal(3, 50, img.shape).astype(np.float32)
    img = img + noise
    img = np.clip(img, 0, None)  # Ensure pixel values are within [0, 4095]
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    # img = (img - np.min(img)) / (np.max(img) - np.min(img)).astype(np.float32)  # Normalize to [0, 1]
    if img is None:
        raise ValueError("Image not found or could not be read.")

    # # 3×3 Laplacian kernel for Immerkaer estimator
    # noise_kernel = np.array(
    #                 [[1, -2, 1],
    #                      [-2, 4, -2],
    #                      [1, -2, 1]], dtype=np.float32)
    #
    #
    # sobel_kernel = np.array([[1, 0, -1],
    #                          [2, 0, -2],
    #                          [1, 0, -1]], dtype=np.float32)
    # plt_img(img)
    # out = cv2.filter2D(img, cv2.CV_32F, noise_kernel)
    # # out = cv2.filter2D(img, cv2.CV_32F, sobel_kernel, )
    # plt_img(out)
    #
    noise_metric = ImageNoiseMetric()
    gradient_metric = ImageGradientMetric()
    entropy_metric = ImageEntropyMetric()
    semantic_metric = ImageSemanticMetric()
    # plt_img(img)


    noise_value = noise_metric.evaluate(img)
    print(f"Noise Level: {noise_value:.4f}")

    gradient_value = gradient_metric.evaluate(img)
    print(f"Gradient Index: {gradient_value:.4f}")


    entropy_value = entropy_metric.evaluate(img*4095)
    print(f"Entropy: {entropy_value:.4f}")
    
    semantic_value = semantic_metric.evaluate(img*4095)
    print(f"Semantic Index: {semantic_value:.4f}")