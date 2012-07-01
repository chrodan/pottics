# -*- coding: utf-8 -*-

import numpy as np

def predict_latentsvm(img_info, unary_f, w2, eps=1e-5, prop_eps=1e-6, randinit_num=5, pyramid=False, **kwargs):

    pR = unary_f(img_info, False)

    # assure pR > 0
    pR[pR <= prop_eps] = prop_eps
    
    if isinstance(w2, str):
        w2 = np.loadtxt(w2)
    alpha_prescale = w2[0]    
    binary_prescale = w2[1]       
    topics_num = w2[2]
    alpha = w2[3]
    w2 = w2[4:].reshape((-1, img_info.ds.classnum))
    assert(topics_num == w2.shape[0])
    regions = img_info.regions
    
    #region_num = len(np.unique(regions))
    if pyramid is False:
        segment_num = img_info.top_masks.shape[-1]
        struct1 = img_info.get_regions_segments_struct(min_size=0, with_gt=False)    
        struct2 = img_info.get_segments_region_struct(min_size=0, with_gt=False)
    else:
        segment_num = np.sum([2**(2*i) for i in range(pyramid)])
        struct2 = img_info.get_pyramid_segments_regions_struct(min_size=0, levels=pyramid)        
        struct1 = img_info.get_pyramid_regions_segments_struct(min_size=0, levels=pyramid)

    
    def minimize_energy(S_init=None, R_init=None):
        en = 0.
        curen = -np.inf
        i=0
        en_tot = -np.inf
        curS = S_init
        curR = R_init
            
        while (np.abs(en - curen) > eps):    
            i+=1
            if curR is not None:
                # optimize topics given regions
                ans = np.zeros((segment_num, topics_num))        
                for v, l in struct2.iteritems():
                    ans[v,:] -= binary_prescale * w2[:,curR[l]].sum(axis=1).flatten()
                        
                curS = np.argmax(ans, axis=1).flatten()        
                
            # optimize regions given topics
            # unary energy
            ans = alpha *alpha_prescale *  np.log(pR)
            # binary energy
            for v, l in struct1.iteritems():
                ans[v,:] -= binary_prescale * w2[curS[l],:].sum(axis=0).flatten()
            curR = np.argmax(ans, axis=1).flatten()
            
            curen, en = np.sum(np.max(ans, axis=1)), curen        
            en_un = alpha_prescale * alpha * np.log(pR)
            en_un = en_un[range(len(curR)),curR].sum()
        
            if curen > en_tot:
                S = curS
                R = curR
                en_tot = curen    
        return R,S, en_tot

    # initialize labels only based on unaries    
    curR = np.argmax(pR, axis=1).flatten()  
    R,S,en_tot = minimize_energy(R_init=curR)    
    source = "unary labels"
    
    for i in range(randinit_num):
        curS = np.random.random_integers(0, topics_num - 1, size=segment_num)
        R2,S2,en_tot2 = minimize_energy(S_init=curS)        
        
        if en_tot2 > en_tot:
            R = R2
            S = S2
            en_tot = en_tot2
            source = "random topics"
    
    #print "Convergence with energy", curen, "unary", en_un
    img_info.prediction = np.zeros_like(regions)
    for x in xrange(regions.shape[0]):
        for y in xrange(regions.shape[1]):
            img_info.prediction[x,y] = R[regions[x,y]]
    img_info.pmap = np.array([])
    img_info.topic_count = np.bincount(S, minlength=topics_num)
    img_info.topics = S
    img_info.energy_init = source
    
