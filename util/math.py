'''
Created on 05.08.2011

@author: christoph
'''
import numpy as np
import util.math_cpp

def logsumexp_slow(summands, mask = None):
    """ computes ln(exp(sum(summands)) in a numerically stable manner
    values along the first dimension are summed, others not"""
    
    
    if not mask is None:
        maskcum = np.max(mask, 0)
        c = np.ma.array(summands, mask= 1 - mask).max(0).data
        s = np.exp(summands - c)
        s[mask==0] = 0
        c += np.log(s.sum(0))
        if np.min(maskcum) == 0:
            c[maskcum==0] = 0
        return c
    else:
        #ind = np.argmax(np.abs(summands),0)
        #c=summands.ravel()[ind]
        c = np.max(summands,0)
        s = np.exp(summands - c).sum(0)
        
    return np.log(s) + c

#logsumexp = util.math_cpp.logsumexp
logsumexp = logsumexp_slow
