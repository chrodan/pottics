'''
Created on 11.09.2011

@author: Christoph Dann <cdann@cdann.de>
'''
import numpy as np

def fill_nn(prediction):
    shape = prediction.shape
    dim = len(shape)
    slcs = [slice(None)]*dim

    while np.any(prediction==255): # as long as there are any False's in flag
        for i in range(dim): # do each axis
            # make slices to shift view one element along the axis
            slcs1 = slcs[:]
            slcs2 = slcs[:]
            slcs1[i] = slice(0, -1)
            slcs2[i] = slice(1, None)
            invalid = prediction==255
            # replace from the right
            repmask = np.logical_and(invalid[slcs1], ~invalid[slcs2])
            prediction[slcs1][repmask] = prediction[slcs2][repmask]
            
            invalid = prediction==255
            
            # replace from the left
            repmask = np.logical_and(invalid[slcs2], ~invalid[slcs1])
            prediction[slcs2][repmask] = prediction[slcs1][repmask]
    return prediction
