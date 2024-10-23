import cupy as cp

def one_hot(cupy_array, number_of_classes):
    return cp.eye(number_of_classes)[cupy_array]
