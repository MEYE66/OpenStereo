import torch
import torch.nn as nn
import torch.nn.functional as F


class FloorDivSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val, K):
        ctx.save_for_backward(val, K)
        rem = torch.remainder(val, K)
        out = torch.floor((val - rem) / K)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        val, K = ctx.saved_tensors
        grad_val = grad_output / K
        grad_K = None  # Typically you don't backpropagate into K
        return grad_val, grad_K


class NeuralExposureController(nn.Module):
    def __init__(self, net_width=[128, 256, 512, 1024, 16], histo=True, max_exposure=10, sigmoid_scale=3.0, max_exp_time=20, max_gain=14, **kwargs):
        super(NeuralExposureController, self).__init__()
        self.net_width = net_width
        self.histo = histo
        # self.learn_delta = learn_delta
        # self.proportional_delta = proportional_delta
        self.max_exposure = max_exposure
        self.sigmoid_scale = sigmoid_scale

        self.max_exp_time = max_exp_time
        self.max_gain = max_gain

        # Define exposure branch: convolutional layers
        self.exposure_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=net_width[0], kernel_size=(4, 1), stride=4), # [B, 128, 64, 15]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=net_width[0], out_channels=net_width[1], kernel_size=(4, 1), stride=4),  # [B, 256, 16, 4]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=net_width[1], out_channels=net_width[2], kernel_size=(4, 1), stride=4), # [B, 512, 4, 1]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=net_width[2], out_channels=net_width[3], kernel_size=(4, 1), stride=4), # [B, 1024, 4, 1]
            nn.LeakyReLU(),
            # dense layer
            nn.Conv2d(in_channels=net_width[3], out_channels=net_width[4], kernel_size=1, bias=False) # # [B, 1024, 1, 1]
        )
        self.final_layer = nn.Conv2d(in_channels=net_width[4], out_channels=1, kernel_size=1, bias=False)

    def forward(self, img, **kwargs):
        img = merge_capture(img)
        exp_feat = multi_scale_histogram(img, **kwargs)  # [B, 1, H/4, W/32]
        # return exp_feat
        # print(f"mul scale histogram:{net.shape}")
        # print(f"DEBUG:histo exp_feat.mean:{exp_feat.mean()}")
        exp_feat = self.exposure_branch(exp_feat)
        # print(f"DEBUG: exposure branch output:{exp_feat.mean()}")
        output = self.final_layer(exp_feat)
        sig_input = self.sigmoid_scale * output
        output = 2 * (torch.sigmoid(sig_input) - 0.5)
        # print(f"DEBUG:after sigmoid output.mean:{output.mean()}")
        output *= torch.log(torch.tensor(self.max_exposure, device=img.device))
        # print(f"DEBUG: after log   output.mean:{output.mean()}")
        new_exp_value = torch.exp(output)
        return new_exp_value


def merge_capture(img_stack, scale_factor=4095, high_thresh=255, mid_low_thresh=256, mid_high_thresh=4094,
                  high_scale=256, mid_scale=16):
    B, _, H, W = img_stack.shape
    img_stack = (img_stack - img_stack.min()) / (img_stack.max() - img_stack.min())
    img_stack *= scale_factor
    def torch_merge(img_stack):
        img_stack = torch.round(img_stack)
        img_low, img_mid, img_high = img_stack[:, 0], img_stack[:, 1], img_stack[:, 2]

        mask_low = ~(img_mid == scale_factor)
        mask_mid = ~((img_mid >= mid_low_thresh) & (img_mid <= mid_high_thresh))
        mask_high = ~(img_mid <= high_thresh)

        img_low[mask_low] = 0
        img_mid[mask_mid] = 0
        img_high[mask_high] = 0

        img_merged = img_high / high_scale + img_mid / mid_scale + img_low
        return img_merged.view(B, 1, H, W)
    return torch_merge(img_stack)


def make_histo_sampling(
        height, width,
        offset_x=5, offset_y=4,
        block_size=42, th_x_max=40, th_y_max=25,
        crop_top=4, crop_left=4, crop_bottom=None, crop_right=None,
        bayer_x=0, bayer_y=0
):
    if crop_bottom is None:
        crop_bottom = height - 4
    if crop_right is None:
        crop_right = width - 4

    w_block = width / block_size
    h_block = height / block_size
    buffer = torch.zeros((height, width, 3), dtype=torch.uint8)

    for block_y in range(block_size // 2):
        for block_x in range(block_size // 2):
            for th_y in range(th_y_max):
                for th_x in range(th_x_max):
                    x = ((th_x + offset_x + int(block_x * w_block)) << 1) + bayer_x
                    y = ((th_y + offset_y + int(block_y * h_block)) << 1) + bayer_y
                    if x < width and y < height:
                        buffer[y, x] = torch.tensor([0, 255, 0], dtype=torch.uint8)

    mask = buffer[:, :, 1].bool()
    buffer_cropped = buffer[crop_top:crop_bottom, crop_left:crop_right]
    mask_cropped = mask[crop_top:crop_bottom, crop_left:crop_right]

    return buffer, mask, buffer_cropped, mask_cropped

def multi_scale_histogram(im_cap, ds_step=2, scales=[1, 3, 7], image_shape=[1080, 1920], log_histogram=False):
    image_height, image_width = image_shape
    slice_list = []

    for scale in scales:
        for i in range(scale):
            for j in range(scale):
                x1 = int(j / scale * image_width)
                x2 = int((j + 1) / scale * image_width)
                y1 = int(i / scale * image_height)
                y2 = int((i + 1) / scale * image_height)
                im_crop = im_cap[:, :, x1:x2:ds_step, y1:y2:ds_step]
                slice_list.append(make_histogram(im_crop))
    net = torch.stack(slice_list, dim=-1).unsqueeze(1)
    
    if log_histogram:
        net = torch.log(1 + net)
        net *= 256 / torch.sum(net, dim=1, keepdim=True)
    return net


def make_histogram(x, nbins=256):
    batch_size = x.shape[0]
    hist = []
    for i in range(batch_size):
        hist.append(torch.histc(x[i, :, :, :,], bins=nbins, min=0.0, max=1.0))
    net = torch.stack(hist, dim=0)
    net = net.float()
    net /= torch.clamp(net.sum(), min=1e-4)
    net *= nbins
    return net


