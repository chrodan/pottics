'''
Created on 05.08.2011

@author: christoph
'''

import datetime

import warnings
from IPython.core.debugger import Tracer
debug_here = None#Tracer()



from .joblib import Memory
cache = Memory("../cache")

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

def cross(*sets):
    wheels = map(iter, sets) # wheels like in an odometer
    digits = [it.next() for it in wheels]
    while True:
        yield digits[:]
        for i in range(len(digits)-1, -1, -1):
            try:
                digits[i] = wheels[i].next()
                break
            except StopIteration:
                wheels[i] = iter(sets[i])
                digits[i] = wheels[i].next()
        else:
            break

def format_class_table(array, colnames, classnames):
    sep = "\t"
    c=21
    assert len(colnames) == array.shape[1]
    assert c == array.shape[0]
    
    # header
    res = "class"+sep+"pos"
    for n in colnames:
        res += sep+n
    
    
    # data
    for i in range(c):
        res+="\n"+classnames[i]+sep+str(i)
        for j in range(len(colnames)):
            res+=sep+"{}".format(array[i,j])
    return res

class Timer(object):
    def __init__(self, name=None, print_enter=True):
        self.name = name
        self.print_enter = print_enter
    def __enter__(self):
        self.tstart = datetime.datetime.now()
        if self.print_enter and self.name:
            print 'Start [%s]' % self.name
            
    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % str(datetime.datetime.now() - self.tstart)
