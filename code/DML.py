import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()
    
    
    def forward(self, x, y):
        x = F.softmax(x.detach())
        y = F.log_softmax(y)
        #print((x*(torch.log(x)-y)).size())
        kld = torch.mean(torch.sum(x*(torch.log(x)-y),dim=1))
        #print(kld.item())
        return kld