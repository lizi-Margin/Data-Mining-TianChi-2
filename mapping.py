import numpy as np

def encode(*args, **kwargs):
    return np.log1p(*args, **kwargs)

def decode (*args, **kwargs):
    return np.expm1(*args, **kwargs)
