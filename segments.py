#!/usr/bin/env python
# coding=utf8
import numpy as np

import util.generic

def get_class_distributions(classmap, segments, num_classes=21, **kwargs):
	
	with util.generic.Timer("Class Distribution comp. for a set of segments"):
		_, _, num_proposals = segments.shape
		segment_ids = range(num_proposals)
		dists = np.zeros((num_classes, num_proposals))
		for i in range(num_proposals):
			try:
				dists[:, i] = get_class_distribution(classmap, segments[:, :, i] == 1, num_classes=num_classes, **kwargs)
			except UserWarning as e:
				print e
				segment_ids.remove(i)
	return dists[:,segment_ids], segment_ids


def flood(mask, value, x, y):
	"""
	floods a given mask inplace with source at (x,y) and the given value.
	"""
	shape = mask.shape
	dim = len(shape)
	slcs = [slice(None)] * dim
	old_value = mask[x,y]
	flooded = np.zeros_like(mask, dtype="bool")

	while np.any(flooded): # as long as there are any False's in flag
		flooded = np.zeros_like(mask, dtype="bool")
		for i in range(dim): # do each axis
			# make slices to shift view one element along the axis
			slcs1 = slcs[:]
			slcs2 = slcs[:]
			slcs1[i] = slice(0, -1)
			slcs2[i] = slice(1, None)
			
			# replace from the right
			repmask = np.logical_and(mask[slcs1] == value, mask[slcs2] == old_value)
			flooded[slcs1] = np.logical_or(flooded[slcs1], repmask)
			repmask = np.logical_and(mask[slcs2] == value, mask[slcs1] == old_value)
			flooded[slcs1] = np.logical_or(flooded[slcs1], repmask)
		mask[flooded] = value
	return mask

def fill_unannotated_nn(segmentation):
	shape = segmentation.shape
	dim = len(shape)
	slcs = [slice(None)]*dim

	while np.any(segmentation==255): # as long as there are any False's in flag
		for i in range(dim): # do each axis
			# make slices to shift view one element along the axis
			slcs1 = slcs[:]
			slcs2 = slcs[:]
			slcs1[i] = slice(0, -1)
			slcs2[i] = slice(1, None)
			invalid = segmentation==255
			# replace from the right
			repmask = np.logical_and(invalid[slcs1], ~invalid[slcs2])
			segmentation[slcs1][repmask] = segmentation[slcs2][repmask]
			
			invalid = segmentation==255
			
			# replace from the left
			repmask = np.logical_and(invalid[slcs2], ~invalid[slcs1])
			segmentation[slcs2][repmask] = segmentation[slcs1][repmask]
	return segmentation

def seperate_gt_segments(object_gt):
	obj_ids = np.unique(object_gt)
	lst = []
	for id in obj_ids:
		if id > 0 and id < 255:
			lst.append((object_gt == id))
	if len(lst) == 1:
		return lst[0][:,:,np.newaxis]
	else:
		return np.dstack(tuple(lst))

def fb_overlap_gt(P, gt, N, k=None, c=21, **kwargs):
	if k is None:
		res = np.zeros((c,))
		for i in range(c):
			if np.any(gt==i):
				res[i] = fb_overlap(P, gt==i, i,N, **kwargs)
		return res
	else:
		return fb_overlap(P, gt==k, k, **kwargs)

def fb_overlap(P, G, k, N, C=90 ):
	""" compute Foreground-Background Overlap defined in "Sequential F-G Ranking" with
	segment P and ground-truth segment G for class k"""
	#debug_here()
	assert G.shape == P.shape
	assert 0 <= k and k < 21
	nP = np.logical_not(P)
	nG = np.logical_not(G)
	# areas
	Ps = (P > 0).sum()
	nPs = nP.sum()
	
	
	iou = float((G & P).sum()) / (G | P).sum()
	iou_n = float((nG & nP).sum()) / (nG | nP).sum()
	t1 = np.sqrt(Ps)/np.log(Ps)/np.sqrt(N[k])*iou
	t2 = np.sqrt(nPs)/np.log(nPs)/np.sqrt(N[0])*iou_n
	return C*(t1+t2)
	
def center_of_mass(mask):
	""" computed the coordinates of the center of mass as a tuple for a given binary segment """
	res = [0]*2
	for i in range(2):
		m = np.add.accumulate(mask.sum(1-i))
		res[i] = float(np.searchsorted(m, float(m[-1])/2, 'left')+ np.searchsorted(m, float(m[-1])/2, 'right')) / 2
	return res


def get_class_distribution(classmap, segment, num_classes=21, normalized=True, unlabeled=255):
	"""produce a mapping (dict) from classnumber to occurance frequency
	within a given segment.
	@param classmap: 2D array [width, height] with integer indices for the classes
	@param segment: 2D bool array [width, height], true where the segment is present in the image
	@param normalized: if set the returned dict is normalized so that it sums to 1	
	@param unlabeled: all pixels in classmap with that label are not considered
"""
	#with util.Timer("Class Distribution computation for a single segment"):
	classes = classmap[segment]
	if not unlabeled is None:
		classes = classes[classes != unlabeled]
	if classes.size == 0:
		raise UserWarning("Segment does not cover labeled parts of the image")
	l = np.bincount(classes, minlength=num_classes)
	if normalized:
		tsum = sum(l)
		l = [v / float(tsum) for v in l]
		
		assert abs(sum(l) - 1.0) <= 0.001
		
	return l


