#!/usr/bin/env python

import os
import numpy as np
import PIL.Image
import scipy.misc
from scipy.io import matlab as ml
import os.path
import util.joblib as joblib
import types
import regions as reg
import logging
import segments


def load(imageset, load_img, **kwargs):
    if len(imageset) > 1:
        return tuple(map(None, *[load_img(n, **kwargs) for n in imageset]))
    else:
        return ( [a,] for a in  map(None, *[load_img(n, **kwargs) for n in imageset]) )



def _normalize(vector, norm):
    norms = np.array(np.apply_along_axis(lambda x: np.linalg.norm(x, norm), axis=0, arr=vector), dtype=np.double)
    norms[norms==0]=1
    return vector/norms


class DatasetImage(object):
    
    def __repr__(self):
        return "{}<{} {} {}>".format(self.__class__.__name__, self.ds.poolname, self.name, self.ds.classnum)
    
    def __getstate__(self):
        result = self.__dict__.copy()
        del result['_region_segment_struct']
        del result['_get_region_ids_gt']
                
        return result
        
    def __setstate__(self, state):
        self.__dict__ = state
        self._region_segment_struct = self.ds.cache.cache(self._regions_segments_struct, ignore=['self'])
        self._get_region_ids_gt = self.ds.cache.cache(self._get_region_ids_gt, ignore=['self'])
        
    
    def __init__(self, name, dataset):
        self.name = name
        self.ds = dataset
        self._region_segment_struct = dataset.cache.cache(self._regions_segments_struct, ignore=['self'])
        self._get_region_ids_gt = dataset.cache.cache(self._get_region_ids_gt, ignore=['self'])
        
    @property
    def top_linear_features(self):
        """ linearized CPMC features for the top 100 proposal segments"""
        fn = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'top100_norm_features_fgbg.mat')
        data = ml.loadmat(fn)
        return data['top_normfeat_fgbg']
    

    
    @property
    def top_hol_features(self, norm=1):
        """ features from the holistic paper for the top 100 proposal segments """
        fnfeat = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'holistic_features.mat')
        data = ml.loadmat(fnfeat)["holfeat"].ravel()
        featnames = ("csift_fg", "sift_fg", "csift_bg", "sift_bg", "phog_contour", "phog_edges", "phog_edges_square")
        
        seg_ids = self.get_top_proposal_ids()
        res = {}
        for i, n in enumerate(featnames):
            res[n] = data[i][:, seg_ids]
            if norm is not None:
                res[n] = _normalize(res[n], norm)
        return res


    def get_top_proposal_ids(self, i=100, force_num=True):
        fn_scores = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'scores.mat')
        sc = ml.loadmat(fn_scores)["scores"]
        seg_ids = list(np.argsort(sc.ravel())[-1:(-1-i):-1])
        if len(seg_ids) < i and force_num:
            seg_ids = (list(seg_ids)*100)[:100]
        return seg_ids
    
    
    def get_top_scores(self, i=100, force_num=True):
        fn_scores = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'scores.mat')
        sc = ml.loadmat(fn_scores)["scores"]
        scores = list(np.sort(sc.ravel())[-1:(-1-i):-1])
        if len(scores) < i and force_num:
            scores = (list(scores)*100)[:100]
        return scores
    
    @property
    def top_masks(self):
        """ top 100 proposal masks"""
        fn = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'top100_masks.mat')
        data = ml.loadmat(fn)
        return data['top_masks']
    
    @property
    def image(self):
        fn = os.path.join(self.ds.path, 'images', '{name}.jpg').format(name=self.name)
        if os.path.exists(fn):
            img = PIL.Image.open(fn)
            return np.asarray(img)
        else:
            raise Exception("Image {} not found".format(fn))
    
    @property
    def ground_truth(self):
        fn = os.path.join(self.ds.path, 'ground_truth', '{name}.png').format(name=self.name)
        if os.path.exists(fn):
            img = PIL.Image.open(fn)
            assert img.mode == 'P'
            return np.asarray(list(img.getdata()), dtype=np.uint8).reshape(img.size[1], img.size[0])
        else:
            return None
    


        
    @property
    def regions(self):
        """ regions created for the top proposals"""
        region_fn = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'top_regions.mat')    
        if os.path.exists(region_fn):
            regions = ml.loadmat(region_fn)['top_regions']
        else:    
            proposals = self.top_masks
            regions = reg.produce_regions(proposals)
            ml.savemat(region_fn, {'top_regions':regions})
            logging.debug("Storing regions in %s" % region_fn)
            if not np.any(np.isnan(regions)):
                logging.debug("All pixels are covered in one region")
        return regions
    
    @property
    def regions_pixeldist(self):
        """ returns the regions distributions for the given image."""
        regiondist_fn = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'region_dists.mat')    
        if os.path.exists(regiondist_fn):
            data = ml.loadmat(regiondist_fn)
            dist = data['dists']
            segm_ids = data["region_ids"]
        else:    
            regions = self.regions
            n_reg = np.max(regions) + 1
            gt = self.ground_truth
            prop = np.zeros((regions.shape[0], regions.shape[1], n_reg), dtype='bool')
            for i in range(n_reg):
                prop[:, :, i] = (regions == i)
            dist, segm_ids = segments.get_class_distributions(gt, prop, num_classes=self.ds.classnum)
            ml.savemat(regiondist_fn, {'dists':dist, 'region_ids': segm_ids})
        segm_ids = list(np.array(segm_ids).ravel())
        return dist, segm_ids

    def get_pyramid_segments_regions_struct(self, min_size=0, with_gt=True, levels=4):
        regid = self.get_region_ids_gt(min_size=min_size, gt_only=with_gt)  
        return reg.create_pyramid_regions_struct(self.regions,region_ids = regid, levels=levels)

    def get_neighborhood_struct(self, min_size=0, with_gt=False):
        regid = self.get_region_ids_gt(min_size=min_size, gt_only=with_gt)  
        return reg.create_region_neighborhood_struct(self.regions, region_ids=regid)
        
    def get_pyramid_regions_segments_struct(self, min_size=0, with_gt=True, levels=4):
        regids = self.get_region_ids_gt(min_size=min_size, gt_only=with_gt)  
        d = self.get_pyramid_segments_regions_struct(min_size=min_size, with_gt=with_gt)
        lst = [[]] * len(np.unique(self.regions))
        for k, v in d.iteritems():
            for regid in v:
                lst[regid] = lst[regid] + [k]
                
        return dict([(cid, lst[cid]) for cid in regids])
    
    def get_regions_segments_struct(self, min_size=0, with_gt=True, p=100):
        regids = self.get_region_ids_gt(min_size=min_size, gt_only=with_gt)  
        d = self.get_segments_region_struct(min_size=min_size, with_gt=with_gt, p=p)
        lst = [[]] * len(np.unique(self.regions))
        for k, v in d.iteritems():
            for regid in v:
                lst[regid] = lst[regid] + [k]
                
        return dict([(cid, lst[cid]) for cid in regids])

    
    def get_region_ids_gt(self, **kwargs):
        return self._get_region_ids_gt(self.name, **kwargs ) 
        
    def _get_region_ids_gt(self, name, min_size=0, gt_only=True):
        gt = self.ground_truth if gt_only else None
        return reg.ground_truth_region_ids(self.regions, gt, min_size=min_size)
        
    def _regions_segments_struct(self, name, min_size=0, with_gt=True): 
        regs = self.regions
        masks = self.top_masks
        gt = self.ground_truth if with_gt else None
        struct = reg.create_regions_segments_struct(regs, masks, min_size=min_size, ground_truth=gt)
        return struct

    def get_segments_region_struct(self, min_size=0, with_gt=True, p=100):
        """produces a dict that maps segment id to a list of region ids that the segment contain
            if min_size is provided, only those regions that are bigger than that size are considered
            if ground_truth is given, then only regions are considered, that are not completely within an unlabeled area
        """
        regid = self.get_region_ids_gt(min_size=min_size, gt_only=with_gt)  
        return reg.create_segments_regions_struct(self.regions, self.top_masks[:,:,:p], region_ids=regid)
    
    @property
    def segments_pixeldist(self):
        segmdist_fn = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'top_masks_dists.mat')    
        if os.path.exists(segmdist_fn):
            data = ml.loadmat(segmdist_fn)
            dist = data['top_masks_dist']
            segm_ids = data["segment_ids"]
        else:    
            prop = self.top_masks
            gt = self.ground_truth
            dist, segm_ids = segments.get_class_distributions(gt, prop, num_classes=self.ds.classnum)
            ml.savemat(segmdist_fn, {'top_masks_dist':dist, 'segment_ids': segm_ids})
        segm_ids = list(np.array(segm_ids).ravel())        
        return dist, segm_ids

class DagsDatasetImage(DatasetImage):
    
    @property
    def ground_truth(self):
        fn = os.path.join(self.ds.path, 'ground_truth', '{name}.regions.txt').format(name=self.name)
        gt = np.loadtxt(fn)
        gt[gt < 0] = 255
        return np.array(gt, dtype=np.uint8)
    
class MsrcDatasetImage(DatasetImage):
    
    
    
    @property
    def ground_truth(self):
        fn = os.path.join(self.ds.path, 'ground_truth', '{name}_GT.bmp').format(name=self.name)
        fn_direct = os.path.join(self.ds.path, 'ground_truth', '{name}.npy').format(name=self.name)
        if os.path.exists(fn_direct):
            return np.load(fn_direct)
        cc = np.loadtxt(os.path.join(self.ds.path, 'ground_truth', 'classcode.txt'))
        
        gt_raw = np.asarray(PIL.Image.open(fn))
        gt = np.ones(gt_raw.shape[:2], dtype=np.uint8)*255
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for c in range(cc.shape[0]):
                    if np.all(cc[c,:] == gt_raw[i,j,:]):
                        gt[i,j] = c
                        continue
        np.save(fn_direct, gt)
        return gt
    

    
class HoofedDatasetImage(DatasetImage):
    
    @property
    def image(self):
        fn = os.path.join(self.ds.path, 'images', '{name}.jpg').format(name=self.name)
        if os.path.exists(fn):
            img = PIL.Image.open(fn)
            
	    if img.mode == "L":
                logging.info("Convert Image {} to RGB mode".format(fn))
                img = img.convert("RGB")
                img.save(fn)
            return np.asarray(img)
        else:
            raise Exception("Image {} not found".format(fn))
    
    @property
    def ground_truth(self):
        fn = os.path.join(self.ds.path, 'ground_truth', '{name}_mask.ppm').format(name=self.name)
        fn_direct = os.path.join(self.ds.path, 'ground_truth', '{name}.npy').format(name=self.name)
        if os.path.exists(fn_direct):
            gt = np.load(fn_direct)
            if np.all(gt.shape != self.image.shape[:2]):
                os.unlink(fn_direct)
                logging.info("GT has wrong size, regenerate")
                gt = self.ground_truth
            return gt
        cc = np.loadtxt(os.path.join(self.ds.path, 'ground_truth', 'classcode.txt'))
        img = PIL.Image.open(fn)
        img = img.resize(self.image.shape[1::-1])
        gt_raw = np.asarray(img).copy()
        gt_raw[gt_raw > 10] = 1
        gt = np.zeros(gt_raw.shape[:2], dtype=np.uint8)
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for c in range(cc.shape[0]):
                    if np.all(cc[c,:] == gt_raw[i,j,:]):
                        gt[i,j] = c
                        continue
        np.save(fn_direct, gt)
        return gt
    
class VOCDatasetImage(DatasetImage):
    
    @property
    def object_masks(self):
        
        ob_fn = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'obj_gt_masks.mat')    
        if os.path.exists(ob_fn):
            data = ml.loadmat(ob_fn)
            masks = data['masks']
            classes = data["classes"]
            
        else:    
            masks = segments.seperate_gt_segments(self.object_ground_truth)
            gt = self.ground_truth
            dists, ids = segments.get_class_distributions(gt, masks)
            classes = np.argmax(dists, axis=0)
            assert masks.shape[2] == len(ids)
            ml.savemat(ob_fn, {'masks':masks, 'classes': classes})
        
        assert masks.shape[2] == classes.size
        return masks, classes

    @property    
    def object_ground_truth(self):
        fn = os.path.join(self.ds.path, 'ground_truth_objects', '{name}.png').format(name=self.name)
        if os.path.exists(fn):
            img = PIL.Image.open(fn)
            assert img.mode == 'P'
            return np.asarray(list(img.getdata()), dtype=np.uint8).reshape(img.size[1], img.size[0])
        else:
            return None
        
    @property
    def object_gt_hol_features(self, norm=1):
        fnfeat = os.path.join(self.ds.path, 'cpmc', 'MySegmentsMat', self.name, 'holistic_features_object_gt.mat')
        data = ml.loadmat(fnfeat)["holgtfeat"].ravel()
        featnames = ("csift_fg", "sift_fg", "csift_bg", "sift_bg", "phog_contour", "phog_edges", "phog_edges_square")
        res = {}
        for i, n in enumerate(featnames):
            res[n] = data[i]
            if norm is not None:
                res[n] = _normalize(res[n], norm)
        return res

class Dataset(object):
    
    classnum = 21
    path = ""
    poolname = ""
    classnames = ()
    image_class = VOCDatasetImage
    
    def __init__(self, names=None):
        self.cache = joblib.Memory(os.path.join(self.path,"features.cache"))
        self.pixels_per_class = self.cache.cache(self._pixels_per_class)
        self._names = names
        
    def __getstate__(self):
        result = self.__dict__.copy()
        del result['cache']
        del result['pixels_per_class']
        return result
        
    def __setstate__(self, state):
        self.__dict__ = state
        self.cache = joblib.Memory(os.path.join(self.path,"features.cache"))
        self.pixels_per_class = self.cache.cache(self._pixels_per_class)
    
    def store_names(self, filename):
        with open(filename, "w") as w:
            for i in self.names:
                w.write(i+"\n")
        
    @property
    def names(self):
        if self._names is not None:
            return self._names
        else:
            self._all_names()
    
    def _all_names(self):
        raise NotImplementedError("Subclasses should implement this")
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self,key):
        if type(key) is types.SliceType:
            return self.__class__(names = self.names[key])
        elif isinstance(key,int):
            n = self.names[key]
            return self.__getitem__(n)
        else:
            if key not in self.names:
                raise IndexError("{} is not in dataset".format(key))
            return self.image_class(key, dataset=self)
    
    def _pixels_per_class(self):
        names = self.names
        cnt = np.zeros((256,), dtype="uint64")
        for n in names:     
            gt = self[n].get_ground_truth(n)
            cnt += np.bincount(gt.ravel(), minlength=256)
        
        return cnt[:self.classnum]


class VOC2011Dataset(Dataset):
    
    path = '../../datasets/formatted/voc2011/'
    poolname = "voc2011"
    classnum = 21
    image_class = VOCDatasetImage
    classnames = ('background', 'aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor')

class DagsDataset(Dataset):
    
    poolname = "dags"
    path = '../../datasets/formatted/dags/'

    image_class = DagsDatasetImage
    classnames = ('sky', 'tree', 'road', 'grass',
                  'water', 'building', 'mountain', 'foreground-object')    
    classnum = len(classnames)
    
class HoofedDataset(Dataset):
    
    poolname = "hoofed"
    path = '../../datasets/formatted/hoofed/'
    
    image_class = HoofedDatasetImage
    classnames = ('background','cow','horse','sheep','goats','camel','deer')
    classnum = len(classnames)

class Msrc21Dataset(Dataset):
    
    poolname = "msrc21"
    path = '../../datasets/formatted/msrc21/'
    
    image_class = MsrcDatasetImage
    classnames = ('building','grass','tree','cow','sheep','sky',
                  'aeroplane','water','face','car','bicycle','flower','sign','bird', 
                  'book','chair','road','cat','dog','body','boat')
    classnum = len(classnames)
    
class MsrcDataset(Dataset):
    
    poolname = "msrc"
    path = '../../datasets/formatted/msrc/'
    
    image_class = MsrcDatasetImage
    classnames = ('void','building','grass','tree','cow','horse','sheep','sky','mountain',
                  'aeroplane','water','face','car','bicycle','flower','sign','bird', 
                  'book','chair','road','cat','dog','body','boat')
    classnum = len(classnames)
    
def _get_voc_names(part=None):
    if part is None:
        part = 'trainval'
    with open(os.path.join(VOC2011Dataset.path, 'ImageSets', 'Segmentation', part + '.txt'), 'r') as data:
            names = data.read().split()
    return names

def _get_dags_names(fold=None,part=None):
    def read_fold_names(fold):    
        with open(os.path.join(DagsDataset.path, "sets",'fold{}.txt'.format(fold)), 'r') as data:
            names = data.read().split()
        return names
    a = set([])
    for i in range(1,6):
        assert a.isdisjoint(read_fold_names(i))
        a = a.union(read_fold_names(i))

    if fold is not None and part is not None:
        if part == "train":
            names = []
            for i in range(1,6):
                if i != fold:
                    names += read_fold_names(i)
            return names
        elif part == "test":
            return read_fold_names(fold)
    else:
        with open(os.path.join(DagsDataset.path, "sets",'all.txt'), 'r') as data:
            names = data.read().split()
        return names

def _get_msrc_names(part=None):
    if part is not None:
        if part == "train":
            part_str = "Train"
        elif part == "val":
            part_str = "Validation"
        elif part == "test":
            part_str = "Test"
        with open(os.path.join(MsrcDataset.path, "sets",'{}.txt'.format(part_str)), 'r') as data:
            names = data.read().split()
        names = [n[:-4] for n in names]
    else:
        with open(os.path.join(MsrcDataset.path, "sets",'names.txt'), 'r') as data:
                names = data.read().split()
    return names

def _get_segm_plane_names(part=None):
    
    fn = {'train': 'segm_planes_and_neg_train.txt', 'test': 'segm_planes_and_neg_test.txt'}
    if part == "val": part = "test"
    if not part is None:
        tmp = fn[part]
        fn = {part: tmp}
    names = []
    for f in fn.values():    
        with open(os.path.join(VOC2011Dataset.path, f), 'r') as data:
            names += data.read().split()
    return names

voc2011_train = VOC2011Dataset(names=_get_voc_names("train"))
voc2011_trainval = VOC2011Dataset(names=_get_voc_names())
voc2011_val = VOC2011Dataset(names=_get_voc_names("val"))

voc2010_val = VOC2011Dataset(names=filter(lambda x: not x.startswith("2011"), _get_voc_names("val")))
voc2010_train = VOC2011Dataset(names=filter(lambda x: not x.startswith("2011"), _get_voc_names("train")))
voc2010_trainval = VOC2011Dataset(names=filter(lambda x: not x.startswith("2011"), _get_voc_names("trainval")))

dags = DagsDataset(names=_get_dags_names())
dags_cv_train = [DagsDataset(names=_get_dags_names(part="train", fold=i+1)) for i in range(5)]
dags_cv_test = [DagsDataset(names=_get_dags_names(part="test", fold=i+1)) for i in range(5)]

hoofed = HoofedDataset(names = [str(i) for i in range(1,201)])
hoofed_cv_test = [HoofedDataset(names = [str(i) for i in range(s+1,201,5)]) for s in range(5)]
hoofed_cv_train = [HoofedDataset(names = list(set([str(i) for i in range(1,201)]) - set(hoofed_cv_test[s].names))) for s in range(5)]

msrc21 = Msrc21Dataset(names=_get_msrc_names())
msrc21_train = Msrc21Dataset(names=_get_msrc_names("train"))
msrc21_test = Msrc21Dataset(names=_get_msrc_names("test"))
msrc21_val = Msrc21Dataset(names=_get_msrc_names("val"))

msrc = MsrcDataset(names=_get_msrc_names())
msrc_train = MsrcDataset(names=_get_msrc_names("train"))
msrc_test = MsrcDataset(names=_get_msrc_names("test"))
msrc_val = MsrcDataset(names=_get_msrc_names("val"))
