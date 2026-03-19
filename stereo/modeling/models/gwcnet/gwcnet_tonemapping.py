# @Time    : 2024/4/2 12:31
# @Author  : zhangchenming
import math
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .gwcnet_backbone import GwcNet as GwcNetBackbone
# from .gwcnet_cost_processor import GwcVolumeCostProcessor
# from .gwcnet_disp_processor import GwcDispProcessor
# from .w


repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    # Prefer absolute imports so this file can be run directly.
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.tonemappers.tonemapper import IANet,RAODNet,SANet 
except ModuleNotFoundError:
    # Fallback for environments where package root is preconfigured.
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet





class IAGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super(IAGwcNet, self).__init__(cfgs)
        self.tonemapper = IANet()  # Replace with your actual tonemapper class

    def forward(self, inputs):
        # Apply tonemapping to the input images
        inputs['left'] = self.tonemapper(inputs['left'])
        inputs['right'] = self.tonemapper(inputs['right'])
        
        # Forward pass through the original GwcNet
        return super(IAGwcNet, self).forward(inputs)
    

class RAODGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super(RAODGwcNet, self).__init__(cfgs)
        self.tonemapper = RAODNet()  # Replace with your actual tonemapper class

    def forward(self, inputs):
        # Apply tonemapping to the input images
        inputs['left'] = self.tonemapper(inputs['left'])
        inputs['right'] = self.tonemapper(inputs['right'])
        
        # Forward pass through the original GwcNet
        return super(RAODGwcNet, self).forward(inputs)
    


class SANGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super(SANGwcNet, self).__init__(cfgs)
        self.tonemapper = SANet()  # Replace with your actual tonemapper class
        state_dict = torch.load("/home/lgz/workspace/OpenStereo/CKPT/ada_canlog-00019.pt")['state_dict'] # faster rcnn weights
        self.tonemapper.load_state_dict(state_dict, strict=False)

    def forward(self, inputs):
        # Apply tonemapping to the input images
        inputs['left'] = self.tonemapper(inputs['left'])
        inputs['right'] = self.tonemapper(inputs['right'])
        
        # Forward pass through the original GwcNet
        return super(SANGwcNet, self).forward(inputs)


class GamutGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super(GamutGwcNet, self).__init__(cfgs)
        # self.tonemapper = GamutNet()  # Replace with your actual tonemapper class
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conversion_matrix = torch.tensor(
            [
                [1.660491, -0.587641, -0.072850],
                [-0.124550, 1.132900, -0.008349],
                [-0.018151, -0.100579, 1.118730],
            ],
            device=device,
            dtype=torch.float32,
        ).t()
        
        # print(self.conversion_matrix.device)
        
    def apply_mu_log(self, hdr_img, qmax=49):
        """
        Logarithmic tone mapping with a fixed mu parameter.
        """
        hdr_img = hdr_img / (qmax + 1)
        mu = 500
        tm = torch.log1p(mu * hdr_img) / math.log1p(mu)
        return torch.clamp(tm, 0, 1) 
    
    def gamut_compression(self, hdr_img):
        
        img = hdr_img.permute(0, 2, 3, 1)
        img = torch.matmul(img, self.conversion_matrix)
        img = img.permute(0, 3, 1, 2)
        return torch.clamp(img, 0, 1)

    def forward(self, inputs):
        # Apply tonemapping to the input images
        left = (inputs['left'] - inputs['left'].min()) / (inputs['left'].max() - inputs['left'].min() + 1e-8)
        right = (inputs['right'] - inputs['right'].min()) / (inputs['right'].max() - inputs['right'].min() + 1e-8)
        
        
        inputs['left'] = self.gamut_compression(self.apply_mu_log(left))
        inputs['right'] = self.gamut_compression(self.apply_mu_log(right))

        # Forward pass through the original GwcNet
        return super(GamutGwcNet, self).forward(inputs)


if __name__ == "__main__":
    # Example usage
    from types import SimpleNamespace
    cfgs = SimpleNamespace(MAX_DISP=192, USE_CONCAT_VOLUME=True, CONCAT_CHANNELS=12, DOWNSAMPLE=4, NUM_GROUPS=8)
    # model = IAGwcNet(cfgs)
    # print(model)
    
    model = GamutGwcNet(cfgs)

    input_data = {
        'left': torch.randn(1, 3, 256, 512),
        'right': torch.randn(1, 3, 256, 512),
    }
    output = model(input_data)
    print(output.keys())



