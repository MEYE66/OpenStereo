# @Time    : 2024/4/2 12:31
# @Author  : zhangchenming
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
        super().__init__(cfgs)
        self.tonemapper = IANet()  # Replace with your actual tonemapper class

    def forward(self, inputs):
        # Apply tonemapping to the input images
        inputs['left'] = self.tonemapper(inputs['left'])
        inputs['right'] = self.tonemapper(inputs['right'])
        
        # Forward pass through the original GwcNet
        return super().forward(inputs)
    

class RAODGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.tonemapper = RAODNet()  # Replace with your actual tonemapper class

    def forward(self, inputs):
        # Apply tonemapping to the input images
        inputs['left'] = self.tonemapper(inputs['left'])
        inputs['right'] = self.tonemapper(inputs['right'])
        
        # Forward pass through the original GwcNet
        return super().forward(inputs)
    


class SANGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.tonemapper = SANet()  # Replace with your actual tonemapper class
        state_dict = torch.load("/home/lgz/workspace/OpenStereo/CKPT/ada_canlog-00019.pt")['state_dict'] # faster rcnn weights
        self.tonemapper.load_state_dict(state_dict, strict=False)

    def forward(self, inputs):
        # Apply tonemapping to the input images
        inputs['left'] = self.tonemapper(inputs['left'])
        inputs['right'] = self.tonemapper(inputs['right'])
        
        # Forward pass through the original GwcNet
        return super().forward(inputs)


if __name__ == "__main__":
    # Example usage
    from types import SimpleNamespace
    cfgs = SimpleNamespace(MAX_DISP=192, USE_CONCAT_VOLUME=True, CONCAT_CHANNELS=12, DOWNSAMPLE=4, NUM_GROUPS=8)
    model = IAGwcNet(cfgs)
    # print(model)

    input_data = {
        'left': torch.randn(1, 3, 256, 512),
        'right': torch.randn(1, 3, 256, 512)
    }
    output = model(input_data)
    print(output.keys())


