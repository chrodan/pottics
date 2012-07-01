Code for Pottics - The Potts Topic Model for Semantic Image Segmentation
========================================================================

This repository contains part of the code that was used for the experiments
in

> C.Dann, P. Gehler, S. Roth, S. Nowozin
> Pottics -- The Potts Topic Model for Semantic Image Segmentation
> DAGM 2012

Latent SVM Training
-------------------

We based the implementation of our training scheme on the SVM^struct with latent
variables of C.-N. Yu and T. Joachims. The source code may not be distributed 
without the permission of the authors. Therefore, we provide a diff located in
latentsvm/latentsvm.patch that contains the changes to the Latent SSVM implementation template
available at http://www.cs.cornell.edu/~cnyu/latentssvm/ .
To obtain our training code, download this template, apply our patch file with the
patch command (on Unix systems) and recompile the package. For instructions, see 
the README contained in the Latent SSVM template.
The input files for SVM training can be generated with the Python function
`util.binfile.export_svmstruct_dataset`.

CPMC Segmentation Generation
----------------------------

To generate the proposal segmentations, the original CPMC implementation was used.
We thank again the authors of CPMC, who made their code publicly available. You
can download it from http://sminchisescu.ins.uni-bonn.de/code/cpmc/ .
It can be used to store the best binary segmentation masks in a mat file. See 
the Data section for details, where to store the masks.


Unary Region Potentials
-----------------------

We compute the unary factor connected to each region by accumulating the 
predictions for each pixel in that region. We use precomputed predictions
for the VOC2010 dataset, that are available at
http://graphics.stanford.edu/projects/densecrf/unary/ ("compressed probabilities").
Thanks again to Philipp Krähenbühl and Vladlen Koltun for providing them.
The predictions have to be decompressed as explained on the website. For information
where to put the decompressed pixel predictions, see the Data format section.

Region Generation and Prediction
--------------------------------

The region generation and prediction of segmentations for images was implemented in Python
with Numpy. See the script `generate_paper_plots.py` for examples how to use the code, i.e. 
make predictions for the baselines or Pottics.
The implementation makes heavy use of harddisk caching. This is done either explicitely by
storing results to mat files or implicitly by the joblib library. 
For example, regions for an image are created from the segments automatically,
when you access the regions-property of an image the first time and is then stored to disk.
Whenever you access the property again, it is loaded from disk instead of recomputed:

    import dataset as ds
    
    # compute the regions for the first image of the VOC2010 dataset
    # this may take a while
    reg1 = ds.voc2010[0].regions
    
    # access the regions again. This is fast.
    reg2 = ds.voc2010[0].regions
    assert(np.all(reg1 == reg2))

Data format
-----------

A set of images that is used for training or prediction is called *dataset*. All data associated
with it are stored wihin a single directory. The location of this folder is specified in the
module `dataset` as a member of each dataset subclass (e.g.: 
`dataset.VOC2011Dataset.path = ../../datasets/formatted/voc2011`). You can adopt this path easily to
your local circumstances. We will refer to this path as *{dspath}*. Each image is identified by a 
unique name (*{name}*) within a dataset.
The following filesystem structure is assumed:
> {dspath}/images/{name}.jpg  -- the image itself
> {dspath}/ground_truth/{name}.png -- the ground truth segmentation (if available)
> {dspath}/densecrf/{name}.unary -- the decompressed unary predictions
> {dspath}/cpmc/MySegmentsMat/{name}/top100_masks.mat -- the binary segmentation masks generated by CPMC
> {dspath}/cpmc/MySegmentsMat/{name}/scores.mat -- the ranking scores for each CPMC segmentation


