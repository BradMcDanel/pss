import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_tensor(t):
    #FIXME: does not match number of dimensions for pyplot
    torch.Tensor.ndim = property(lambda self: len(self.shape))
    #convert it to numpy: x.numpy()
    #convert the dimensions
    plt.imshow(t)
    return 0

def random_mask(x, keep_ratio=1.0):
    b,h,w = x.shape
    keep_point = int(round(keep_ratio*h))
    mask = torch.randint(0,h-1,(keep_point,))
    return mask

def magnitude_mask(x, keep_ratio=1.0):
    b,h,w = x.shape
    sums = torch.sum(torch.abs(x),dim=2).sort(dim=1)[1]
    return sums[:,:int(round(keep_ratio*h))]

def center_pick(x, keep_ratio=1.0):
    #FIXME: I do not know how to make a Gaussian map
    return 0

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    t = torch.zeros(1,196,768).normal_()
    print(random_mask(t,0.168)) #randomize the values' indexes in the layers
    x = torch.zeros(1,196,168).normal_()
    print(x)
    print(magnitude_mask(x,0.37)) #returns the top values but in sorted order


if __name__ == '__main__':
    print_hi('PyCharm')



