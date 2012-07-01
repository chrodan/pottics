import numpy as np

def mean_squared_error(pred, gt):
    return np.mean(np.power(pred - gt,2))
    
def mean_absolute_error(pred, gt):
    return np.mean(np.abs(pred - gt))

def misclass_error(pred, gt):
    c_gt = np.argmax(gt, axis=0)
    c_pred = np.argmax(pred, axis=0)
    return float((c_pred != c_gt).sum()) / c_gt.size

def misclass_error_per_class(pred, gt):
    c_gt = np.argmax(gt, axis=0)
    c_pred = np.argmax(pred, axis=0)
    tmp = c_gt[c_gt != c_pred]
    c = gt.shape[0]
    if tmp.size == 0:
        return np.zeros((c,))
    else:
        return np.array(np.bincount(tmp,minlength=c) , dtype=np.double) / np.bincount(c_gt,minlength=c)
        
def kl_divergence(pred, gt):
    tmp = pred.copy()
    tmp[tmp==0] = 1e-64
    tmp = np.log(gt/tmp)
    tmp[gt == 0 ] = 0
    return np.mean(np.sum(gt*tmp, axis=0))