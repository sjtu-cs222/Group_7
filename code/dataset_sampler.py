import torch.utils.data.sampler as sampler
import numpy as np

class BigSampler:
    def __init__(self, num_train, num_val):
        num_samples = num_train + num_val
        perm = np.random.permutation(num_samples)
        self.train_idx = perm[:num_train]
        self.val_idx = perm[num_train:num_val+num_train]
        self.train_sampler = ChunkSampler(self.train_idx)
        self.val_sampler = ChunkSampler(self.val_idx)
        
    def shuffle(self):
        np.random.shuffle(self.train_idx)
    
    
class ChunkSampler(sampler.Sampler): 
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, lis):
        self.lis = lis

    def __iter__(self):
        return iter(self.lis)

    def __len__(self):
        return len(self.lis)

