import os, pickle, warnings, itertools
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse
import hcpalign_utils as hutils
from hcpalign_utils import sizeof, ospath
import matplotlib.pyplot as plt, hcp_utils as hcp, tkalign_utils as utils
from tkalign_utils import full, values2ranks as ranks, regress as reg, identifiability as ident
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import spearmanr as sp
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt


z1= hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/0-50_rest_0-10.npy')
z2= hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/0-50_rest_40-50.npy')

b1,f1,a1 = utils.load_f_and_a(z1)
b2,f2,a2 = utils.load_f_and_a(z2)

align_nparcs=300
nparcs=301
nblocks=5

def func1(a):
    return np.transpose( np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs)) , (0,1,3,2)) #Now a is n_subs_test * n_subs_test * nparcs * nblocksperparc

c1=func1(a1)
c2=func1(a2)

a1n=utils.subtract_nonscrambled_from_a(c1)
a2n=utils.subtract_nonscrambled_from_a(c2)


def removenans(arr):
    arr = arr[~np.isnan(arr)]
    return arr
def get_vals(a):
    array=[]
    for i in range(nparcs):
        values = removenans(a[:,:,i,:])
        array.append(values)
    return np.array(array)
def tstat(a):
    array=[]
    for i in range(a.shape[0]):
        t,p=ttest_1samp(a[i,:],0)
        array.append(t)
    return array


a1x=get_vals(a1n)
a2x=get_vals(a2n)

a1t=tstat(a1x)
a2t=tstat(a2x)

a1m=np.mean(a1x,axis=-1)
a2m=np.mean(a2x,axis=-1)

a1nN= [utils.count_negs(a1n[:,:,i,:]) for i in range(nparcs)]
a2nN= [utils.count_negs(a2n[:,:,i,:]) for i in range(nparcs)]

print(np.corrcoef(a1t,a2t))
print(np.corrcoef(a1m,a2m))
print(np.corrcoef(a1nN,a2nN))




"""
p=hutils.surfplot('ddd',plot_type='open_in_browser')
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)
p.plot(a1t @ align_parc_matrix,savename='ddd')
"""

"""
a1nN= [utils.count_negs(a1n[:,:,i,:]) for i in range(nparcs)]
a2nN= [utils.count_negs(a2n[:,:,i,:]) for i in range(nparcs)]
print(np.corrcoef(a1nN,a2nN))
plt.scatter(a1nN,a2nN)
plt.show()
"""
