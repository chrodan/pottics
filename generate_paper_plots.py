# -*- coding: utf-8 -*-
"""
Generation of paper plots
Created on Thu Apr 19 21:02:05 2012

@author: Christoph Dann <cdann@cdann.de>
"""

import matplotlib as mpl
#mpl.use('PDF')
import predict.base
import predict.post
import util.visual
import dataset as ds
import subprocess
import numpy as np
import segments
import regions
import os
import loading

import unaries.densecrf
#import prediction.evaluation

import latentsvm
import util.visual
import predict.base
import predict.evaluation

import matplotlib.pyplot as plt

name_list=("2007_000733", # person, motorbike
           "2007_001149", # monitor, sofa, pottedplant
           "2007_002597", # dog, cat
           "2007_003503", # bus, car,
           "2007_003957", # train
           "2011_003066", # chair, person, diningtable, bottle
           "2007_004459", # person aeroplane
           "2007_004537", # horse, cow, person
           "2007_005107", # bike, person
           "2007_005130", # bird
           "2008_000120", # boat
           "2008_001601") # sheep)

def prepare_jpg_plot():
    plt.clf()
    ax =  plt.axes([0,0,1,1]) 
    ax.set_axis_off()
    return ax

def save_plot(fn, path):
    plt.savefig(path+fn, bbox_inches="tight", pad_inches=0.1, quality=95)
    subprocess.call("convert -trim "+fn+" "+fn, shell=True, cwd=path)

def cb_gt(name, ax):
    
    util.visual.show_annotated_image(name, ax, alpha=0.8, show_bg=False)
    
 
    
def make_predictions(image):
    w2 = np.loadtxt("../data/voc10train_t50_c1e3.lsvm")
    predict.base.predict_pixel_unaries_on_regions(image, lambda x, y: unaries.densecrf.pixel_unary(x, '{dspath}/densecrf_unaries/{name}.unary'), w2)
    image.rawconf,_ = predict.evaluation.get_confusion_mat(image.prediction, image.ground_truth, image.ds.classnum)
    _,_,image.hacc_unary = predict.evaluation.get_hamming_score_img([image])
    image.unary_prediction = image.prediction    
    del image.prediction
    latentsvm.predict_latentsvm(image, unaries.densecrf.region_unary, w2, Psi_f=None, randinit_num=3)
    image.rawconf,_ = predict.evaluation.get_confusion_mat(image.prediction, image.ground_truth, image.ds.classnum)
    _,_,image.hacc = predict.evaluation.get_hamming_score_img([image])
    image.improvement = image.hacc- image.hacc_unary

if __name__=="__main__":
    path = '../../../dagm12/images/rider/'
    plt.figure()
    n = "2007_005331"
    img = ds.voc2010_trainval[ds.voc2010_trainval.names.index(n)]
    masks = img.top_masks
    scores = img.get_top_scores()
       
 
    num = 12    
    for i in range(num):
        ax = prepare_jpg_plot()
        mask = masks[:,:,i]
        mask[mask > 0] = 6
        util.visual.show_image_mask(img, mask, ax, alpha=0.9)     
        fn = 'segment_{:0=2d}.jpg'.format(i+1)
        util.visual.add_inner_title(ax, "Score: {:.4}".format(scores[i]), 3, size={"size":45, "color":"w"})
        save_plot(fn, path)
    regs = regions.produce_regions(masks[:,:,:num])
    ax = prepare_jpg_plot()
    ax.imshow(regs,cmap=mpl.colors.ListedColormap ( np.random.rand ( 256,3)), interpolation="nearest")
    save_plot("regions.png", path)
    
    ax = prepare_jpg_plot() 
    reg = img.regions
    ax.imshow(reg,cmap=mpl.colors.ListedColormap ( np.random.rand ( 256,3)) ,interpolation='nearest')
    ax.set_axis_off()
    save_plot("regions_all.png", path)    
    
    ax = prepare_jpg_plot() 
    util.visual.show_annotated_image(img, ax, show_bg=True, alpha=1.)    
    save_plot("gt.jpg", path)
    
    ax = prepare_jpg_plot() 
    util.visual.show_annotated_image(img, ax, show_bg=False, alpha=0.)    
    save_plot("img.jpg", path)
    
    w2 = np.loadtxt("../data/voc10train_t50_c1e3.lsvm")
    alpha_prescale = w2[0]    
    binary_prescale = w2[1]       
    topics_num = w2[2]
    alpha = w2[3]
    print alpha, alpha_prescale, binary_prescale
    w2 = w2[4:].reshape((-1, ds.voc2010_trainval.classnum))
    plt.figure(figsize=(8,3))
    plt.imshow(w2.T, interpolation="nearest")
    plt.xlabel("Topics")
    plt.ylabel("Classes")
    plt.colorbar()
    plt.savefig(path+"../t50_param.pdf")
    
    # Potts model
    plt.figure()
    d = {-15.0: 0.17103042426067733, -30.0: 0.062159744574600821, -20.0: 0.17085102071912303, -10.0: 0.18369981224948917, -0.5: 0.20696463519914904, -0.75: 0.20392326688153847, -0.3: 0.20925346346946677, -0.1: 0.21599148471137419, -5: 0.1937169597255228, -1: 0.20335897436734418}
    a = d.items()
    a.sort()
    #a.reverse()
    x,y = tuple(zip(*a))
    plt.plot(x,y)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("VOC Score")
    
    plt.savefig(path+"../potts.pdf")

    path = '../../../dagm12/images/visuals/'
    imgs = [ds.voc2010_val[i] for i in [590, 851, 688, 845, 140, 627, 591, 61, 645]]
    for img in imgs:
        make_predictions(img)
        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, show_bg=False, alpha=0.9)    
        save_plot("impr_{}_gt.jpg".format(img.name), path)

        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, annotation=img.prediction, show_bg=False, alpha=0.9)    
        save_plot("impr_{}_pottics.jpg".format(img.name), path)    
        
        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, annotation=img.unary_prediction, show_bg=False, alpha=0.9)    
        save_plot("impr_{}_unary.jpg".format(img.name), path)  
        print img.name, ":   Unary Hamming Score", img.hacc_unary, " pottics Hamming Score", img.hacc
        
    imgs = [ds.voc2010_val[i] for i in [586, 354, 262, 80, 678, 54, 755, 721]]
    for img in imgs:
        make_predictions(img)
        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, show_bg=False, alpha=0.9)    
        save_plot("worse_{}_gt.jpg".format(img.name), path)

        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, annotation=img.prediction, show_bg=False, alpha=0.9)    
        save_plot("worse_{}_pottics.jpg".format(img.name), path)    
        
        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, annotation=img.unary_prediction, show_bg=False, alpha=0.9)    
        save_plot("worse_{}_unary.jpg".format(img.name), path)  
        
        print img.name, ":   Unary Hamming Score", img.hacc_unary, " pottics Hamming Score", img.hacc
        
        
    imgs = [ds.voc2010_val[i] for i in [125, 525, 729]]
    for img in imgs:
        make_predictions(img)
        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, show_bg=False, alpha=0.9)    
        save_plot("topic_{}_gt.jpg".format(img.name), path)

        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, annotation=img.prediction, show_bg=False, alpha=0.9)    
        save_plot("topic_{}_pottics.jpg".format(img.name), path)    
        
        ax = prepare_jpg_plot() 
        util.visual.show_annotated_image(img, ax, annotation=img.unary_prediction, show_bg=False, alpha=0.9)    
        save_plot("topic_{}_unary.jpg".format(img.name), path)  
        
        print img.name, ":   Unary Hamming Score", img.hacc_unary, " pottics Hamming Score", img.hacc

