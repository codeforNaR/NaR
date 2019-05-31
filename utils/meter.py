import numpy as np 

class MultAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, size=None):
        self.size=size

    def reset(self):
        self.val = np.zeros(self.size)
        self.avg = np.zeros(self.size)
        self.sum = np.zeros(self.size)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        if self.size is None:
            self.size=val.shape
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
