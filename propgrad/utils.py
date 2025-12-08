import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax as softmax_func
import torch
import torch.utils.data as data

def timer(func):
    """Timer directive to capture execution time.
    Usage:

    @timer
    def func():
        pass

    Args:
        func : function to wrap.
    """

    def timer_wrapper(*args, **kwargs):
        start = time.time()
        return_values = func(*args, **kwargs)
        total = time.time() - start
        print("Execution of {} took {:.3f} seconds.".format(func.__name__, total))
        return return_values

    return timer_wrapper




def subsample_plot(img_list, num=7, idx_list=None, s = 4):
    fig = plt.figure(figsize=(num*s, s))
    axs = fig.subplots(1, num)

    if idx_list is None:
        idx_list = np.linspace(0, len(img_list)-1, num, endpoint=True, dtype=int)


    subset = [img_list[idx] for idx in idx_list]

    for i, (ax, img) in enumerate(zip(axs, subset)):
        ax.imshow(img)
        ax.axis("off")
    
    fig.tight_layout()

    return fig


def perform_inference(M, D, softmax=True):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    
    M = M.to(device)
    M.eval()

    y = []
    #print(D)
    with torch.inference_mode():
        for x in D:
            x = x.to(device)

            y_hat = M(x)
            y.append(y_hat.cpu().numpy())

    logits = np.concatenate(y, axis=0)
    if softmax:
        return softmax_func(logits, axis=1)
    else:
        return logits
    

class ImageList_DS(data.Dataset):
    def __init__(self, image_list, transform):
        self.transform = transform
        self.data = image_list
        self.targets = []
    
    def set_targets(self, targets):
        if isinstance(targets, int):
            self.targets = [targets for _ in range(len(self))]
        else:
            self.targets = targets
    
    def __getitem__(self, index):
        """Gets the x and applies transformations."""
        sample = self.data[index]
        if self.transform is not None:
            sample = self.transform(sample)

        if len(self.targets) == 0:
            return sample
        else:
            return sample, self.targets[index]
    
    def __len__(self):
        return len(self.data)