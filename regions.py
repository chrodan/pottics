#!/usr/bin/env python
# coding=utf8
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
import segments




def region_sizes_img(region_map, visualize=False):
    
    un = np.unique(region_map)
    assert np.max(un) + 1 == un.size
    sizes = list()    
    for r in un:
        assert np.isfinite(r)
        size = np.count_nonzero(region_map == r)
        sizes.append(size)
    assert np.sum(sizes) == region_map.size
    if visualize:
        plt.figure()
        plt.hist(sizes, bins=100, log=True)
        
    return sizes

def create_region_neighborhood_struct(regions, neighborhood=np.ones((3,3), dtype=bool), region_ids=None):
    
    if region_ids is None:
        region_ids = set(np.unique(regions))
    else:
        region_ids = set(region_ids)
    result = {}
    for i in region_ids:
        mask = binary_dilation(regions==i, structure=neighborhood)
        result[i] = list((set(np.unique(regions[mask])) - set([i])) & region_ids)
    return result
def create_pyramid_regions_struct(regions, region_ids=None, levels=4):
    """produces a dict that maps pyramid segment id to a list of region ids
        if min_size is provided, only those regions that are bigger than that size are considered
        if ground_truth is given, then only regions are considered, that are not completely within an unlabeled area
    """

    if region_ids is None:
        region_ids = set(np.unique(regions))
    else:
        region_ids = set(region_ids)
    
    result = []
    
    b,h = regions.shape
    for l in range(levels):
        n = 2**l
        for i in range(n):
            for j in range(n):
                frame = regions[int(float(b) / n * (i)):int(float(b) / n * (i+1)), int(float(h) / n * (j)):int(float(h) / n * (j+1))]
                result.append(list(set(np.unique(frame)) & region_ids))

    return dict(zip(range(len(result)), result))

def ground_truth_region_ids(regions, ground_truth, min_size=0):
    sizes = region_sizes_img(regions)
    region_ids = []
    for i, x in enumerate(sizes):
        if x > min_size:
            if not ground_truth is None:
                try:
                    segments.get_class_distribution(ground_truth, regions == i)
                    region_ids.append(i)
                except UserWarning:
                    continue
            else:
                region_ids.append(i)
    return region_ids
def create_regions_segments_struct(regions, masks, min_size=0, ground_truth=None):
    """produces a dict that maps region id to a list of segment ids
        if min_size is provided, only those regions that are bigger than that size are considered
        if ground_truth is given, then only regions are considered, that are not completely within an unlabeled area
    """
    sizes = region_sizes_img(regions)
    region_ids = []
    for i, x in enumerate(sizes):
        if x > min_size:
            if not ground_truth is None:
                try:
                    segments.get_class_distribution(ground_truth, regions == i)
                    region_ids.append(i)
                except UserWarning:
                    continue
            else:
                region_ids.append(i)
    
    rs = get_all_segments(region_ids, regions, masks)
    return dict(zip(region_ids, rs))
    
def create_segments_regions_struct(regions, masks, region_ids=None):
    """produces a dict that maps segment id to a list of regions ids
        if min_size is provided, only those regions that are bigger than that size are considered
        if ground_truth is given, then only regions are considered, that are not completely within an unlabeled area
    """

    if region_ids is None:
        region_ids = set(np.unique(regions))
    else:
        region_ids = set(region_ids)
    if masks.dtype != "bool":
        masks = masks == 1
    result = {}
    for i in range(masks.shape[-1]):
        cur_ids = np.unique(regions[masks[:,:,i]])
        result[i] = list(set(cur_ids) & region_ids)
    
    return result
    
def region_struct_to_dot(struct, filename=None):
    """ creates a dot graph for a given region - segments structure and saves it in the given file, if provided"""
    edge_list = []
    seg_set = set()
    reg_set = set()
    for region, segments in struct.iteritems():
        reg_set.add("R_{}".format(region))
        for s in segments:
            seg_set.add("S_{s}".format(s=s))
            edge_list.append('''"S_{s}" -> "R_{r}"'''.format(s=s, r=region))
    
    
    result = """
    digraph region_model {{
            
        {edges}
        {{ rank=same; {segs} }}
        {{ rank=same; {regs} }}
    }}""".format(edges=";\n".join(edge_list), segs=" ".join(seg_set), regs=" ".join(reg_set))
    if not filename is None:
        with open(filename, 'w') as f:
            f.write(result)
    return result



def get_parent_segments(region, masks):
    """returns the id of the segments that overlap the given region
    @param region given as boolean 2D array [width, height]
    @param masks proposal segmentations given as [width, height, num_proposals]
"""
    ids = []
    for i in range(masks.shape[2]):
        m = masks[:, :, i]
        if not masks.dtype == bool:
            m = m == 1
        if np.any(region & m):
            ids.append(i)
    return ids

def get_all_segments(ids, regions, masks):
    """returns for each region in in ids the segment_ids of the segments that overlap the given region
    @param ids regions to consider
    @param region given as boolean 2D array [width, height]
    @param masks proposal segmentations given as [width, height, num_proposals]
"""
    return [get_parent_segments(regions == i, masks) for i in ids]

def region_size_mask(regions, visualize=False):
    """ creates for a given region image an image where each pixel has the value of the size of the region it belongs to"""
    un = np.unique(regions)
    assert np.max(un) + 1 == un.size
    values = [0] * (np.max(un) + 1)
    sizes = regions.copy()
    for r in un:
        s = np.count_nonzero(regions == r)
        sizes[regions == r] = s
        values[r] = s
    if visualize:
        plt.figure()
        plt.imshow(sizes)
        plt.set_cmap('hot')
        plt.colorbar()    
    return sizes
            


        

def produce_regions(masks, visualize=False):
    """given the proposal segmentation masks for an image as a [width, height, proposal_num]
    matrix outputs all regions in the image"""
    width, height, n_prop = masks.shape
    t = ('u8,'*int(np.math.ceil(float(n_prop) / 64)))[:-1]
    bv = np.zeros((width, height), dtype=np.dtype(t))
        
    for i in range(n_prop):
        m = masks[:, :, i]
        a = 'f%d' % (i / 64)    
        h = m * np.long(2 ** (i % 64))
        if n_prop >= 64:
            bv[a] += h
        else:
            bv += h


    un = np.unique(bv)
    regions = np.zeros((width, height), dtype="uint16")
    for i, e in enumerate(un):
        regions[bv == e] = i
    if visualize:
        plt.figure()
        plt.imshow(regions)
        plt.set_cmap('prism')
        plt.colorbar()
    return regions





