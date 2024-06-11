import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch
import numpy as np

from backboned_unet import Unet


class NewUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = Unet(classes=1, position_classes=10, damage_classes=6)
        # path = 'G:/My Drive/model-last.pt'
        path = './log/20230510051527/model-last.pt'
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.unet.load_state_dict(state_dict)
        self.unet.train(False)

    def forward(self, x):
        res_list = []
        with torch.no_grad():
            pos_out, dmg_out, hp_out, seg_out = self.unet(x)

            seg_out = torch.sigmoid(seg_out)
            seg_out = seg_out.max(dim=1).values
            B, H, W = seg_out.shape
            seg_out = seg_out.reshape(B, 1, H, W)
            return pos_out, dmg_out, hp_out, seg_out
            # for i in range(len(seg_out)):
            #     # dmg_label = dmg_out[i].argmax().item()
            #     out = torch.sigmoid(seg_out)[i]
            #     out = out.max(dim=0).values
            #     out[out < 0.5] = 0
            #     out[out >= 0.5] = 1
            #     res_list.append(out)
            
            # return torch.stack(res_list)


net = NewUnet()
# net = Unet(classes=1, position_classes=10, damage_classes=6, backbone_name='resnet50')
# path = 'C:/Users/euygdun/Desktop/model-last.pt'
# state_dict = torch.load(path, map_location=torch.device('cpu'))
# net.load_state_dict(state_dict)
net.train(False)

x = torch.randn(1, 3, 480, 480)
torch_out = net(x)

torch.onnx.export(
    net,
    x,
    'UG_UNET.onnx',
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes= {
        'input': {0 : 'batch_size'},    
        'output': {0 : 'batch_size'}
    }
)

import onnxruntime

ort_session = onnxruntime.InferenceSession("UG_UNET.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)[0]

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs, rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")