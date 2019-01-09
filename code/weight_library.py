import torch
import torch.nn as nn
from collections import OrderedDict
import copy

#Should make sure these weights are stored on CPU?
class WeightLibrary:
    def __init__(self):
        self.state_dict = None
        self.conv_names = []
        self.fc_names = []
        self.LAYERS = 0
        self.conv_shapes = []
        self.fc_shapes = []
    
    def save_library(self, model):
        model = model.cpu()
        self.state_dict = model.state_dict()
        if self.LAYERS > 0:
            return
        self.LAYERS = 0
        cnt = 0
        for x in self.state_dict:
            state_dict_shape = self.state_dict[x].shape
            if len(state_dict_shape) == 4:
                self.conv_names.append(x)
                self.conv_shapes.append(state_dict_shape)
                self.LAYERS += 1
            elif len(state_dict_shape) == 2:
                self.fc_names.append(x)
                self.fc_shapes.append(state_dict_shape)
                self.LAYERS += 1
                
        
        self.fc_names = self.fc_names[:1]
        self.fc_shapes = self.fc_shapes[:1]
        self.LAYERS = len(self.conv_names) + len(self.fc_names)
        
    def save_weights(self, layer_idx, weights):
        if layer_idx >= len(self.conv_names):
            self.state_dict[self.fc_names[layer_idx-len(self.conv_names)]] = weights.cpu()
        else:
            self.state_dict[self.conv_names[layer_idx]] = weights.cpu()
        
    def load_weights(self, layer_idx):
        if layer_idx >= len(self.conv_names):
            return copy.deepcopy(self.state_dict[self.fc_names[layer_idx-len(self.conv_names)]])
        else:
            return copy.deepcopy(self.state_dict[self.conv_names[layer_idx]])

if __name__ == '__main__':
    from model import *
    model = Network().cuda()
    wl = WeightLibrary()
    wl.save_library(model)
    print(wl.conv_names)
    
    