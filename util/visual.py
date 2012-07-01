'''
Created on Aug 9, 2011

@author: cdann
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
import numpy as np
import logging
import dataset as ds
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter

import matplotlib.ticker as ticker
_colors_sep = np.ones((21,1,3))
_colors_sep[1:,:,0] = np.linspace(0,1,21)[:20, np.newaxis]
_colors_sep[0,:,2] = 0
_colors_sep = mpl.colors.hsv_to_rgb(_colors_sep)

_colors_tud = [mpl.colors.hex2color(s) for s in ["#000000",
            "#005AA9", "#0083CC", "#009D81", "#99C000", "#C9D400", 
            "#FDCA00", "#F5A300", "#EC6500", "#E6001A", "#A60084", "#721085",
            "#00ff8d", "#ff5f00", "#6c00ff", "#00ffc5", "#ff006f"                        ]]

_colors_tango = [(0,0,0),(0.988235, 0.913725, 0.309804), # Butter1
           (0.643137, 0.000000, 0.000000), # ScarletRed3
(0.929412, 0.831373, 0.000000), # Butter2
(0.305882, 0.603922, 0.023529), # Chameleon3
(0.768627, 0.627451, 0.000000), # Butter3
(0.541176, 0.886275, 0.203922), # Chameleon1
(0.360784, 0.207843, 0.400000), # Plum3
(0.807843, 0.360784, 0.000000), # Orange3
(0.450980, 0.823529, 0.086275), # Chameleon2

(0.988235, 0.686275, 0.243137), # Orange1
(0.125490, 0.290196, 0.529412), # SkyBlue3
(0.960784, 0.474510, 0.000000), # Orange2

(0.447059, 0.623529, 0.811765), # SkyBlue1

(0.756863, 0.490196, 0.066667), # Chocolate2
(0.800000, 0.000000, 0.000000), # ScarletRed2
(0.203922, 0.396078, 0.643137), # SkyBlue2

(0.678431, 0.498039, 0.658824), # Plum1
(0.560784, 0.349020, 0.007843), # Chocolate3
(0.458824, 0.313725, 0.482353), # Plum2

(0.913725, 0.725490, 0.431373), # Chocolate1

(0.937255, 0.160784, 0.160784), # ScarletRed1


(0.933333, 0.933333, 0.925490), # Aluminium1
(0.827451, 0.843137, 0.811765), # Aluminium2
(0.729412, 0.741176, 0.713725), # Aluminium3
(0.533333, 0.541176, 0.521569), # Aluminium4
(0.333333, 0.341176, 0.325490), # Aluminium5
(0.180392, 0.203922, 0.211765) # Aluminium6
 ]
clsscm_tango = mpl.colors.ListedColormap(_colors_tango, name='voc_classes_tango', N=21)
clsscm_tango.set_over((1,1,1))
clsscm_tango.set_under((0,0,0,0))

clsscm_tud = mpl.colors.ListedColormap(_colors_tud, name='voc_classes_tango', N=21)
clsscm_tud.set_over((1,1,1))
clsscm_tud.set_under((0,0,0,0))

clsscm_hue = mpl.colors.ListedColormap(np.squeeze(_colors_sep), name='voc_classes_hue', N=21)
clsscm_hue.set_over((1,1,1))
clsscm_hue.set_under((0,0,0,0))

clsscm = clsscm_hue

#function cmap = labelcolormap(N)
#
#if nargin==0
#    N=256
#end
#cmap = zeros(N,3);
#for i=1:N
#    id = i-1; r=0;g=0;b=0;
#    for j=0:7
#        r = bitor(r, bitshift(bitget(id,1),7 - j));
#        g = bitor(g, bitshift(bitget(id,2),7 - j));
#        b = bitor(b, bitshift(bitget(id,3),7 - j));
#        id = bitshift(id,-3);
#    end
#    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
#end
#cmap = cmap / 255;

def label_colormap(num=256):
    f = lambda x, n : (x >> n) % 2
    cmap = np.zeros((num, 3))

    for i in range(num):
        id = i
        r = 0
        b = 0
        g = 0
        for j in range(7,0,-1):
            r |= f(id,0) << j
            g |= f(id,1) << j
            b |= f(id,2) << j
            id = id >> 3
        cmap[i,:] = np.array([r,g,b])
    cmap /= 255
    
    cm = mpl.colors.ListedColormap(cmap, name='voc_classes_hue', N=num)
    cm.set_over((1,1,1))
    cm.set_under((0,0,0,0))
    
    return cm



def get_latex_colordefinitions(classcm, classnames ):
    res = ""
    for name, col in zip(classnames, clsscm.colors):
        print col
        res+= "\definecolor{{{name}_col}}{{rgb}}{{{col[0]:.4f},{col[1]:.4f} ,{col[2]:.4f} }}\n".format(name=name, col=col)
    return res

def show_segmentation(seg, ax, numclasses=21, **kwargs):
    
    ax.imshow(seg, cmap=label_colormap(numclasses),interpolation='bilinear',norm=mpl.colors.NoNorm(), vmin=0, vmax=255, **kwargs)
    ax.set_axis_off()

def show_annotated_image(img_info,ax=None, annotation=None,  alpha=0.6, show_bg=True):
    if annotation is None:    
        gt = img_info.ground_truth
    else:
        gt = annotation
    img = img_info.image
    if not show_bg:
        gt = np.ma.masked_where(gt == 0, gt) 
    if ax is None:
        ax = plt.gca()
    ax.imshow(img)
    show_segmentation(gt, ax, numclasses=img_info.ds.classnum, alpha=alpha)

def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

def show_image_mask(img_info, mask, ax=None, alpha=0.6, show_bg=False):     
    img = img_info.image
    if not show_bg:
        mask = np.ma.masked_where(mask == 0, mask) 
    if ax is None:
        ax = plt.gca()
    ax.imshow(img)
    show_segmentation(mask, ax, alpha=alpha)    
    
def interactive_dataset_plot(dataset):
    plt.figure()
    plt.subplots_adjust(bottom=0.2, top=0.95, right=0.8)
    
    ax = []
    
    def show(info, axes):
        for a in axes:
            a.clear()
        axes[0].imshow(info.image)
        show_segmentation(info.ground_truth, axes[1])
    
    for i in range(2):
        ax.append(plt.gcf().add_subplot(1, 2, i+1))
    show(dataset[0], ax)
    
    class Index:
        img_ind = 0
        axname = None
        def next(self, event):
            if (self.img_ind+1) in range(len(dataset)):
                self.img_ind += 1
            try:
                show(dataset[self.img_ind], ax)
                self.axname.set_text(dataset[self.img_ind].name)
                
            except:
                self.img_ind -=1
            
            
            plt.draw()
    
        def prev(self, event):
            if (self.img_ind-1) in range(len(dataset)):
                self.img_ind -= 1
            try:
                show(dataset[self.img_ind], ax)
                self.axname.set_text(dataset[self.img_ind].name)
            except:
                self.img_ind +=1
                
            
           

    cb = Index()
    cb.axname = plt.figtext(0.1, 0.05, dataset[0].name, fontsize=24)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(cb.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(cb.prev)
    plt.show()
    
def interactive_predictions_plot(img_infos):
    plt.figure()
    plt.subplots_adjust(bottom=0.2, top=0.95, right=0.8)
    
    ax = []
    
    def show(info, axes):
        for a in axes:
            a.clear()
        axes[0].imshow(info.image)
        show_segmentation(info.ground_truth, axes[1], numclasses=info.ds.classnum)
        show_segmentation(info.prediction, axes[2], numclasses=info.ds.classnum)
        cb = mpl.colorbar.ColorbarBase(axes[3], cmap=label_colormap(info.ds.classnum), norm=mpl.colors.NoNorm())
        cb.set_ticks(np.arange(info.ds.classnum))
        cb.set_ticklabels(info.ds.classnames)
    for i in range(4):
        ax.append(plt.gcf().add_subplot(2, 2, i+1))
    show(img_infos[0], ax)
    
    class Index:
        img_ind = 0
        axname = None
        def next(self, event):
            if (self.img_ind+1) in range(len(img_infos)):
                self.img_ind += 1
            try:
                show(img_infos[self.img_ind], ax)
                self.axname.set_text(img_infos[self.img_ind].name)
                
            except:
                self.img_ind -=1
            
            
            plt.draw()
    
        def prev(self, event):
            if (self.img_ind-1) in range(len(img_infos)):
                self.img_ind -= 1
            try:
                show(img_infos[self.img_ind], ax)
                self.axname.set_text(img_infos[self.img_ind].name)
            except:
                self.img_ind +=1
                
            
           

    cb = Index()
    cb.axname = plt.figtext(0.1, 0.05, img_infos[0].name, fontsize=24)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(cb.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(cb.prev)
    plt.show()
    

def image_interactive_plot(names, *callbacks):

    plt.figure()
    plt.subplots_adjust(bottom=0.2, top=0.95, right=0.8)
    #cax = plt.axes([0.8, 0.2, 0.2, 0.8])
    #im = cax.imshow(np.arange(21).reshape(3,7),cmap=clsscm, alpha=0.0,interpolation='bilinear',norm=mpl.colors.NoNorm(), vmin=0, vmax=255)
    #cb = mpl.colorbar.Colorbar(cax, im)
    #cb.set_ticks(range(21))
    #cb.set_ticklabels(ds.classnames)
    n = len(callbacks)
    y = int(np.ceil(np.sqrt(n)))
    x = int(np.ceil(float(n) / y))
    ax = []
    for i, c in enumerate(callbacks):
        ax.append(plt.gcf().add_subplot(x, y, i+1))
        c(names[0], ax[i])
    class Index:
        img_ind = 0
        axname = None
        def next(self, event):
            if (self.img_ind+1) in range(len(names)):
                self.img_ind += 1
            try:
                for i, c in enumerate(callbacks):
                    ax[i].clear()
                    c(names[self.img_ind], ax[i])
                self.axname.set_text(names[self.img_ind])
                
            except:
                self.img_ind -=1
            
            
            plt.draw()
    
        def prev(self, event):
            if (self.img_ind-1) in range(len(names)):
                self.img_ind -= 1
            try:
                for i, c in enumerate(callbacks):
                    ax[i].clear()
                    c(names[self.img_ind], ax[i])
                self.axname.set_text(names[self.img_ind])
            except:
                self.img_ind +=1
                
            
           

    cb = Index()
    cb.axname = plt.figtext(0.1, 0.05, names[0], fontsize=24)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(cb.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(cb.prev)
    plt.show()

    
def show_3dplot(ax, res,x,y, xname, yname, zname):


    val = [x,y]
        
    Z = np.ma.masked_invalid(res)
    X,Y = np.meshgrid(np.linspace(0,1,len(val[0])), np.linspace(0,1,len(val[1])))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.get_cmap("BrBG"), antialiased=True, alpha=0.5)

    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    ax.w_xaxis.set_major_locator(ticker.FixedLocator(np.linspace(0,1,len(val[0]))))
    ax.w_xaxis.set_ticklabels(val[0])

    ax.w_yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0,1,len(val[1]))))
    ax.w_yaxis.set_ticklabels(val[1])

    ax.fmt_ydata = ax.fmt_xdata = lambda x : str(x)


def insert_figtitle(fig, text):
        fig.text(0.5, 0.05,text,horizontalalignment='center')
def prepare3dfig(fignum, subplotnum, title):
    fig = plt.figure(fignum)
    #fig.clear()
    fig.dpi=75
    insert_figtitle(fig, title)
    myaxes = []
    x = np.ceil(np.sqrt(subplotnum))
    y = np.ceil(subplotnum/x)
    for i in xrange(subplotnum):
        myaxes.append(fig.add_subplot(y, x, i, projection='3d'))
    return fig, myaxes