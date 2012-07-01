import numpy as np
import model
import logging

import loading
from collections import defaultdict

    
def blend_unaries(image_info, unary_f, perfect_unaries):
    
    pS = unary_f(image_info) 
    if perfect_unaries > 0.0:
        pS_gt = loading.pS(image_info)
        pS = pS_gt*perfect_unaries + (1-perfect_unaries)*pS
        np.clip(pS, 0, 1, pS)
        pS /= pS.sum(0)
    return pS
def predict_max_cpmc(img_info, unary_f, w2, **kwargs):
    """ predict the cpmc segmentation that obtains the highest score according to unaries 
        i. e. argmax_{i, c} p(S_i = c)"""
      
    # load feature

    pS = unary_f(img_info, False)
    return _predict_for_ranking(img_info, pS)

def predict_top_cpmc(img_info, unary_f, w2, **kwargs):
    """ predict the top1 cpmc segmentation for the class with the highest unary score
        i. e. argmax_{c} p(S_1 = c)"""

    # load feature

    pS = unary_f(img_info, False)
    pS[:,1:] = -np.Inf
    return _predict_for_ranking(img_info, pS)

def predict_simple_voting(img_info, unary_f, w2, n_seg=100, **kwargs):
    """ predict a segmentation by sampling from pS and let each S vote for the classes """

    pS = unary_f(img_info, False)[:,:n_seg]
    segs = img_info.top_masks[:,:,:n_seg]

    w_segs = np.tensordot(segs, pS, ([2], [1]))
    img_info.prediction = np.argmax(w_segs, axis=2)
    img_info.pmap = pS



def predict_with_pixel_unaries(img_info, unary_f, w2, **kwargs):
    pixProp = unary_f(img_info, False)
    img_info.prediction = np.argmax(pixProp, 2)
    img_info.pmap = np.array([])

def predict_pixel_unaries_on_regions(img_info, unary_f, w2, **kwargs):
    pixProp = unary_f(img_info, False)
    regions = img_info.regions
    cnt = np.zeros((np.max(regions)+1, img_info.ds.classnum))
    for x in xrange(regions.shape[0]):
        for y in xrange(regions.shape[1]):
            cnt[regions[x,y],:] += pixProp[x,y,:]
    region_prediction = np.argmax(cnt, axis=1)
    img_info.prediction = np.zeros_like(regions)
    for x in xrange(regions.shape[0]):
        for y in xrange(regions.shape[1]):
            img_info.prediction[x,y] = region_prediction[regions[x,y]]
    img_info.pmap = np.array([])        

def predict_best_possible_regions(img_info, unary_f, w2, **kwargs):
    """ produces the best segmentation for the region superpixelation using ground truth information,
    i.e. the max marginal for each region"""
    def my_pR(img_info, with_ids=True):
        return loading.pR(img_info, min_size=0, only_available=True, with_ids=with_ids)
    kwargs['region_unary_f'] = my_pR
    return predict_unary_regions(img_info, unary_f, w2,**kwargs)

def predict_potts_model(img_info, unary_f, w2, prop_eps=1e-12, init="unary", **kwargs):
    pR = unary_f(img_info, False)
    regs = img_info.regions
    # assure pR > 0
    pR[pR <= prop_eps] = prop_eps
    pR = np.log(pR)
    
    if init is "unary":
        curR = np.argmax(pR, axis=1).flatten() 
    else:
        curR = np.random.randint(0,img_info.ds.classnum - 1, pR.shape[0])
  
    struct = img_info.get_neighborhood_struct()
    num_r = len(struct)
    j=0
    en=0
    while j == 0 or (not np.all(curR == lastR)):
        
        j+=1
        lastR = curR.copy()
        
        en=0        
        for i in np.random.permutation(num_r):
            r = struct.keys()[i]
            p = pR[r,:].flatten().copy()
            p -= w2 * np.bincount(curR[struct[r]], minlength=len(p))
            en += np.max(p)            
            c = np.argmax(p)
            if c != curR[r]:
                curR[r] = c

        if j > 1000:
            print "Warning, not converged"
            break
    prediction = np.ones_like(regs)*255
    for i in struct.keys():
        prediction[regs==i] = curR[i]
    img_info.pmap = None
    img_info.prediction = prediction
    
    
def predict_unary_regions(img_info, unary_f, w2, region_unary_f,**kwargs):
    """ predicts the segmentatin by classifying each region according to the maximum of a
        region unary function region_unary_f """
    pR, rid = region_unary_f(img_info, with_ids=True)
    regs = img_info.regions
    prediction = np.ones_like(regs)*255
    _, q = pR.shape
    R = np.argmax(pR, axis=0)
    for i in range(q):
        prediction[regs==rid[i]] = R[i]
    img_info.pmap = pR
    img_info.prediction = prediction
    
def _predict_for_ranking(img_info, prop_ranks, no_back=True):
    start_cls = 1 if no_back else 0
    best = np.unravel_index(np.argmax(prop_ranks[start_cls:,:]), (prop_ranks[start_cls:,:].shape))
    seg_id = best[1]
    cls = best[0]+1

    # 
    masks = img_info.top_masks
    best_prop = masks[:,:,seg_id]
    best_prop[best_prop>0] = cls
    logging.info('Predicted {name}'.format(name = img_info.name))
    
    img_info.prediction = best_prop
    img_info.prediction_class = cls
    img_info.prediction_seg_id = seg_id
    img_info.pmap = prop_ranks
 
