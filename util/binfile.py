# -*- coding: utf-8 -*-
"""
functions to load and store binary files such as probimage file format that is used by
http://graphics.stanford.edu/projects/densecrf/unary/
to store the unary potentials

Created on Fri Mar  2 15:49:34 2012

@author: christoph
"""

import numpy as np
import struct

def load_probimage(filename):
    with open(filename) as f:
        w,h,d = struct.unpack_from('!3i', f.read(12))
        num = w*h*d
        data = struct.unpack_from('!%df' % num, f.read())
    return np.array(data).reshape((h, w, d))

def export_svmstruct_dataset(ds, filename, unary_f, with_gt=True, pyramid=False, p=100):
    with open(filename.format(dspath=ds.path), "wb") as f:
        header = struct.pack("!i", len(ds))    
        f.write(header)        
        for img in ds:
            export_svmstruct(f, img, unary_f, with_gt, pyramid=pyramid, p=p)
            
def export_svmstruct_dataset_fakedseg(ds, filename, unary_f, with_gt=True):
    with open(filename.format(dspath=ds.path), "wb") as f:
        header = struct.pack("!i", len(ds))    
        f.write(header)        
        for img in ds:
            export_svmstruct_fakedseg(f, img, unary_f, with_gt)
    
def export_svmstruct(f, img, unary_f, with_gt = True, pyramid=False, p=100):
    unaries = unary_f(img)
    
    if pyramid:
        structure = img.get_pyramid_regions_segments_struct(min_size=0, with_gt=with_gt)
    else:
        structure = img.get_regions_segments_struct(min_size=0, with_gt=with_gt, p=p)
    count = np.bincount(img.regions.flatten())
    gtdist, gtregids = img.regions_pixeldist
    
    assert(unaries.shape[0] ==count.shape[0])
    header = struct.pack("{}s".format(len(img.name)+1), img.name)
    header += struct.pack("!2i", img.ds.classnum, len(structure))
    
    f.write(header)
    #print img.name, "Classes: ", img.ds.classnum, "Regions:", len(structure)
    for r, v in structure.items():
        gt = np.argmax(gtdist[:,gtregids.index(r)])
        line = struct.pack("!3i", r, count[r], gt)
        #print "Region", r, "Size", count[r], "GT", gt
        line += struct.pack("!{}f".format(img.ds.classnum), *tuple(unaries[r,:]))
        #print "unaries:", tuple(unaries[r,:])
        line += struct.pack("!{}h".format(len(v)), *tuple(v))        
        line += struct.pack("!h",-1)
        #print "adjacent to", v
        f.write(line)
       
def export_svmstruct_fakedseg(f, img, unary_f, with_gt = True):
    """ export dataset but make shure each segment only contains 1 class"""
    unaries = unary_f(img)
    structure = img.get_regions_segments_struct(min_size=0, with_gt=with_gt)
    count = np.bincount(img.regions.flatten())
    gtdist, gtregids = img.regions_pixeldist
    
    assert(unaries.shape[0] ==count.shape[0])
    header = struct.pack("{}s".format(len(img.name)+1), img.name)
    header += struct.pack("!2i", img.ds.classnum, len(structure))
    
    f.write(header)
    #print img.name, "Classes: ", img.ds.classnum, "Regions:", len(structure)
    segdict = {}
    for r, v in structure.items():
        gt = np.argmax(gtdist[:,gtregids.index(r)])
        line = struct.pack("!3i", r, count[r], gt)
        seglist = []        
        for seg in v:
            if (seg in segdict and segdict[seg] == gt) or seg not in segdict:
                seglist.append(seg)
                segdict[seg] = gt
        #print "Region", r, "Size", count[r], "GT", gt
        line += struct.pack("!{}f".format(img.ds.classnum), *tuple(unaries[r,:]))
        #print "unaries:", tuple(unaries[r,:])
        line += struct.pack("!{}h".format(len(seglist)), *tuple(seglist))        
        line += struct.pack("!h",-1)
        #print "adjacent to", v
        f.write(line)
       
      