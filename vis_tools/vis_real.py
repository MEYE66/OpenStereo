import cv2
import enum
import numpy as np
from pathlib import Path
import skimage
from sympy import gamma
import sys
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from vis_tools.utils import blind_radial_undistort, undistort_radial_bgr
# from vis_tools.utils import undistort_radial_bgr


debayer = None

class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)




class Debayer5x5(torch.nn.Module):
    """Demosaicing of Bayer images using Malver-He-Cutler algorithm.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.

    Compared to Debayer2x2 this method does not use upsampling.
    Compared to Debayer3x3 the algorithm gives sharper edges and
    less chromatic effects.

    ## References
    Malvar, Henrique S., Li-wei He, and Ross Cutler.
    "High-quality linear interpolation for demosaicing of Bayer-patterned
    color images." 2004
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer5x5, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    # G at R,B locations
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [-2,  4,  8,  4, -2], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R,B at G in R rows
                    # scaled by 16
                    [ 0,  0,  1,  0,  0], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [-2,  8, 10,  8, -2], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [ 0,  0,  1,  0,  0], # noqa

                    # R,B at G in B rows
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 1,  0, 10,  0,  1], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R at B and B at R
                    # scaled by 16
                    [ 0,  0, -3,  0,  0], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [-3,  0, 12,  0, -3], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [ 0,  0, -3,  0,  0], # noqa

                    # R at R, B at B, G at G
                    # identity kernel not shown
                ]
            ).view(4, 1, 5, 5).float() / 16.0,
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            # Below, note that index 4 corresponds to identity kernel
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        planes = torch.cat(
            (planes, x), 1
        )  # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather(
            planes,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand for singleton batch dimension is faster
        )
        return torch.clamp(rgb, 0, 1)

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)





def load_data(image_path):
    data = np.load(image_path).astype(np.uint8) 
    return data


def convert_to_32bit_bayer_rg24(raw_data, width, height):
    # raw_data = np.fromfile(raw_data, dtype=np.uint8)
    raw_data = raw_data.reshape(-1, 3)
    raw_int32 = (raw_data[:, 0].astype(np.uint32) +
                (raw_data[:, 1].astype(np.uint32) << 8) +
                (raw_data[:, 2].astype(np.uint32) << 16))
    raw_int32 = raw_int32.reshape(height, width)
    raw_int32 = raw_int32 / (2**24 - 1)  # Normalize to [0, 1]
    return raw_int32.astype(np.float32)

def prosee_debayer(bayer_data):
    # bayer_data = (bayer_data - bayer_data.min()) / (bayer_data.max() - bayer_data.min())
    with torch.no_grad():
        bayer_tensor = torch.from_numpy(bayer_data).unsqueeze(0).unsqueeze(0).float()
        rgb_tensor = get_debayer()(bayer_tensor)
        rgb_image = rgb_tensor.squeeze(0).permute(1, 2, 0).numpy()
    return rgb_image


def get_debayer():
    global debayer
    if debayer is None:
        debayer = Debayer5x5(layout=Layout.RGGB)
    return debayer



def apply_awb(rgb_image):
    r_mean = np.mean(rgb_image[:, :, 0])
    g_mean = np.mean(rgb_image[:, :, 1])
    b_mean = np.mean(rgb_image[:, :, 2])

    r_gain = g_mean / (r_mean + 1e-6)
    b_gain = g_mean / (b_mean + 1e-6)

    rgb_image[:, :, 0] *= r_gain
    rgb_image[:, :, 2] *= b_gain

    rgb_image = np.clip(rgb_image, 0, 1)
    return rgb_image

def apply_ltm(rgb_image, gamma=2.2, ev=-2):
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()).astype(np.float32)
    img_linear = np.power(np.clip(rgb_image, 0.0, 1.0), gamma)
    img_linear = img_linear * (2.0 ** ev)
    img_linear = np.clip(img_linear, 0.0, 1.0)
    rgb_image = np.power(img_linear, 1.0 / gamma)
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()).astype(np.float32)
    return rgb_image




def apply_gtm(img, eps=1e-6, param=0.3):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    return out



def np_to_image(image):
    image = image / image.max()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image



def process_left(raw_data_path):
    raw_data = np.load(raw_data_path)
    bayer_data = convert_to_32bit_bayer_rg24(raw_data, raw_data.shape[1], raw_data.shape[0])
    rgb_image = prosee_debayer(bayer_data)
    rgb_image = apply_awb(rgb_image)


    rgb_image = apply_gtm(rgb_image, eps=1e-6, param=0.18)
    rgb_image = undistort_radial_bgr(rgb_image, k1=-0.34, k2=0.10, f_scale=0.70, alpha=0.0)

    rgb_image = np_to_image(rgb_image)
    return rgb_image


def process_right(raw_data_path):
    raw_data = np.load(raw_data_path)
    bayer_data = convert_to_32bit_bayer_rg24(raw_data, raw_data.shape[1], raw_data.shape[0])
    rgb_image = prosee_debayer(bayer_data)
    rgb_image = apply_awb(rgb_image)
    rgb_image = undistort_radial_bgr(rgb_image, k1=-0.34, k2=0.10, f_scale=0.70, alpha=0.0)

    rgb_image = apply_ltm(rgb_image, gamma=1, ev=2.1)
    # rgb_image = blind_radial_undistort(rgb_image,)
    rgb_image = np_to_image(rgb_image)
    return rgb_image



if __name__ == '__main__':
    # /home/lgz/dataset/ADEC/real/val_vis/Test2/14_09_45_058
    
    # root_path ="/home/lgz/dataset/ADEC/real/val/Test1/14_03_56_478"
    root_path ="/home/lgz/dataset/ADEC/real/val/Test2/14_09_45_058"
    left_img_path = root_path + "/left.npy"
    right_img_path = root_path + "/right.npy"
    get_debayer()

    

    rgb_image_left = process_left(left_img_path)
    rgb_image_right = process_right(right_img_path)
    
    
    
    
    
    # left = np.clip(left*255, 0, 255).astype(np.uint8)
    # left = np_to_image(rgb_image_left)
    cv2.imwrite('./left_image.png', cv2.cvtColor(rgb_image_left, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./right_image.png', cv2.cvtColor(rgb_image_right, cv2.COLOR_RGB2BGR))