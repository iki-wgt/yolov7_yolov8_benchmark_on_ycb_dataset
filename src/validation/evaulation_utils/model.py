import numpy as np
import random
import torch
import torch.nn as nn
import yaml
from models.common import Conv, DWConv

# Inspired by: https://github.com/WongKinYiu/yolov7/blob/3b41c2cc709628a8c1966931e696b14c11d6db0c/detect.py

class Model:

    def __init__(self,model_path,device):
        self.model_path = model_path
        self.model = self.__attempt_load(self.model_path,map_location=device)
        self.model.eval()
    
    def get_model(self):
        return self.model
    
    def print_model_infos(self, path): # path_to_opt.yaml
        with open(path, "r") as stream:
            opt_dic = yaml.safe_load(stream)
            print("--MODEL INFOS--")
            print(f"pretrained_weights: {opt_dic['weights']}")
            print(f"epochs: {opt_dic['epochs']}")
            print(f"batch_size: {opt_dic['batch_size']}")
            print(f"img_size: {opt_dic['img_size']}")
            print(f"workers: {opt_dic['workers']}")


    def __attempt_load(self,weights, map_location=None):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        model = Ensemble()
        for w in weights if isinstance(weights, list) else [weights]:
            ckpt = torch.load(w, map_location=map_location)  # load
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        if len(model) == 1:
            return model[-1]  # return model
        else:
            print('Ensemble created with %s\n' % weights)
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model  # return ensemble



class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


