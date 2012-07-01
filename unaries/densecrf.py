# -*- coding: utf-8 -*-
"""
functions to handle unary pixel potentials of 

http://graphics.stanford.edu/projects/densecrf/unary/

Created on Fri Mar  2 16:00:29 2012

@author: Christoph Dann <cdann@cdann.de>
"""

import dataset as ds
import segments, regions
import numpy as np
import logging
from util.generic import cache
from util.binfile import load_probimage


def pixel_unary(imageset, filepattern):
    if isinstance(imageset, ds.DatasetImage):
        fn = filepattern.format(name=imageset.name, dspath=imageset.ds.path)
        pixel_unaries = load_probimage(fn)
        imgdim = imageset.top_masks.shape[:2]
        assert imageset.ds.classnum == pixel_unaries.shape[-1]
        if pixel_unaries.shape[:2] != imgdim:
            if pixel_unaries.shape[1::-1] == imgdim:
                pixel_unaries = np.transpose(pixel_unaries, axes=(1, 0, 2))
            else:
                logging.warn("Dimension mistach for dense unaries of {}".format(imageset) )
        return pixel_unaries
    else:
        return np.hstack((pixel_unary(n, filepattern=filepattern) for n in imageset))


@cache.cache
def unary_from_probfile(imageset, filepattern, with_ids=False):
    """ contructs pS from densecrf probimages"""
    if isinstance(imageset, ds.DatasetImage):
        fn1 = filepattern.format(name=imageset.name, dspath=imageset.ds.path)
        pixel_unaries = load_probimage(fn1)
        masks = imageset.top_masks  
             
        unary = np.empty((imageset.ds.classnum, masks.shape[-1]))
        
        for i in range(masks.shape[-1]):
            unary[:,i] = pixel_to_segment_prob(masks[:,:,i], pixel_unaries)
            
        assert np.allclose(np.sum(unary,axis=0), np.ones_like(np.sum(unary,axis=0)), rtol=1e-2)
        unary /= unary.sum(0)
        if with_ids:
            return unary, range(unary.shape[1])
        else:
            return unary
    else:
        return np.hstack((unary_from_probfile(n, filepattern=filepattern,with_ids=with_ids) for n in imageset))


def region_unary(x,y): 
    return region_unary_from_probfile(x, '{dspath}/densecrf_unaries/{name}.unary')


@cache.cache
def region_unary_from_probfile(img, filepattern):
    fn1 = filepattern.format(name=img.name, dspath=img.ds.path)
    pu = load_probimage(fn1)
    reg = img.regions
    regids = np.unique(reg)
    regids.sort()
    assert(np.all(regids == np.arange(len(regids))))
    if pu.shape[:2] != reg.shape and pu.shape[:2] == reg.shape[::-1]:
        pu = np.swapaxes(pu, 0, 1)
    res = np.zeros((len(regids),img.ds.classnum))            
    for i in xrange(pu.shape[0]):
        for j in xrange(pu.shape[1]):
            res[reg[i,j],:] += pu[i,j,:]
    res /= np.sum(res, axis=1)[:, np.newaxis]
    return res
    
def pixel_to_segment_prob(segment_mask, pixel_prob):
    total_num = float(np.count_nonzero(segment_mask))    
    result = np.ones(pixel_prob.shape[-1])
    if total_num == 0.:
        return result / result.sum()
    for c in range(len(result)):    
        cur = pixel_prob[:,:,c] 
        if cur.shape != segment_mask.shape and cur.shape[::-1] == segment_mask.shape:
            cur = np.swapaxes(cur, 0, 1)
        result[c] = cur[segment_mask > 0].sum() / total_num
    if not np.all(np.isfinite(result)):            
        import ipdb; ipdb.set_trace()
    return result
