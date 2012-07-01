'''
Created on Aug 9, 2011

@author: Christoph Dann <cdann@cdann.de>
'''

import numpy as np
import multiprocessing
import logging
def get_confcount(segments, gt, parallel=False, num_cls=21):
    """ obtain unnormalized confusion matrix for a given list of proposals and groundtruth """
    if parallel:
        pool = multiprocessing.Pool()
        mmap = pool.map
    else:
        mmap = map
    inp = zip(segments, gt, [num_cls]*len(gt))
    rawconf = mmap(_get_cmat_mappable, inp)
    rawconf = np.dstack(rawconf).sum(2)
    assert rawconf.shape == (num_cls,num_cls)
    return rawconf

def _extract_conf(img_infos):
    if len(img_infos) == 0:
        return None
    c = img_infos[0].ds.classnum
    rawconf = np.zeros((c,c))
    
    for info in img_infos:
        rawconf += info.rawconf
    return rawconf

def get_voc_score_img(img_infos):
    """ computes the voc score for a given set of imageinfos"""
    rawconf = _extract_conf(img_infos)
    if rawconf is None:
        return None
    return voc_score_for_conf(rawconf)


def get_hamming_score_img(img_infos):
    """ computes the hamming score for a given set of imageinfos"""
    rawconf = _extract_conf(img_infos)
    if rawconf is None:
        return None
    return hamming_score_for_conf(rawconf)

def get_pixel_dist_img(imginfos):
    """ computes the number of pixels for each class based on a set of imageinfos"""
    if len(imginfos) == 0:
        return None
    c = imginfos[0].ds.classnum
    rawconf = np.zeros((c,c))
    
    for info in imginfos:
        rawconf += info.rawconf
    return rawconf.sum(1)

def voc_score_for_conf(conf):
    """ computes the voc score for a given unnormalized confidence matrix"""
    union = conf.sum(0)+conf.sum(1)-np.diag(conf)
    if np.any(union == 0):
        logging.warning('some classes do not occur in ground truth')
        union[union == 0] = 1
    acc = np.diag(conf)/union
    return np.mean(acc), acc    

def hamming_score_for_conf(conf):
    """ computes the hamming score = 1 - hamming loss for a given unnormalized confidence matrix"""
    union = conf.sum(0)
    if np.any(union == 0):
        logging.warning('some classes do not occur in ground truth')
        union[union == 0] = 1
    acc = np.diag(conf)/union
    return np.mean(acc), acc,np.array(np.trace(conf), dtype="double")/conf.sum()

def get_voc_score(segments, gt, parallel=False, num_cls=21):        
    conf = get_confcount(segments, gt, parallel, num_cls)
    return voc_score_for_conf(conf)
    

def get_confusion_mat(seg1, gt_seg, num_cls=21):
    assert np.max(seg1) <=num_cls
    assert np.min(seg1) >=0
    assert np.max(gt_seg) <=255
    assert np.min(gt_seg) >=0
    
    raw, _, _ = np.histogram2d(seg1.ravel(), gt_seg.ravel(), range(257))
    conf = raw[:num_cls, :num_cls]
    return conf, raw

def _get_cmat_mappable(inp):
    seg1, seg2, num_cls = inp
    c, _ = get_confusion_mat(seg1, seg2, num_cls)
    return c
