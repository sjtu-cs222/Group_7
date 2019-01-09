import torch
import numpy as np
import os
from model import Network

def show_compression_rate(model):
    state_dict = model.state_dict()
    cnt_compressed = 0
    cnt_original = 0
    for x in state_dict:
        weight = state_dict[x]
        print(np.sum(weight.numpy()!=0))
        cnt_compressed += np.sum(weight.numpy()!=0)
        cnt_original += weight.numel()
    
    print('After Compression: %d'%cnt_compressed)
    print('Before Compression: %d'%cnt_original)
    print('Compression Rate: %.4f'%(float(cnt_original)/cnt_compressed))



if __name__ == '__main__':
    model = Network()
    #model.load_state_dict(torch.load('checkpoints/model_layer_1.pkl'))
    model.load_state_dict(torch.load('model.pkl'))
    show_compression_rate(model)
