import hcp_utils as hcp
import numpy as np
import os
from sklearn.svm import LinearSVC
import hcpalign_utils as hutils
from joblib import Parallel, delayed
#from my_surf_pairwise_alignment import MySurfacePairwiseAlignment, LowDimSurfacePairwiseAlignment

c=hutils.clock()
hcp_folder=hutils.hcp_folder
intermediates_path=hutils.intermediates_path
results_path=hutils.results_path

subs=hutils.subs[slice(0,2)]
nsubs = np.arange(len(subs)) #number of subjects
MSMAll=False

post_decode_smooth=hutils.make_smoother_100610(0)

nalign,string = hutils.get_movie_or_rest_data(subs,'movie',runs=[0],fwhm=0,clean=True,MSMAll=MSMAll)
ndecode,decode_string = hutils.get_task_data(subs,hutils.tasks[0:7],MSMAll=MSMAll)
nlabels = [np.array(range(i.shape[0])) for i in ndecode] 

parcellation_string='S300'
clustering = hutils.parcellation_string_to_parcellation(parcellation_string)
classifier=LinearSVC(max_iter=10000,dual='auto')     


"""
aligner=MySurfacePairwiseAlignment(alignment_method='scaled_orthogonal', clustering=clustering,n_jobs=-1,reg=0)  #faster if fmralignbench/surf_pairwise_alignment.py/fit_parcellation uses processes not threads
aligner.fit(nalign[0],nalign[1])
aligner.transform(ndecode[0])

from fmralign.surf_pairwise_alignment import SurfacePairwiseAlignment
aligner2 = SurfacePairwiseAlignment(alignment_method='scaled_orthogonal',clustering=clustering,alignment_kwargs ={'scaling':True})
aligner2.fit(nalign[0],nalign[1])
"""

"""
from my_template_alignment import MyTemplateAlignment, get_template  
aligners= MyTemplateAlignment('scaled_orthogonal',clustering=clustering,n_jobs=1,n_iter=2,scale_template=False,template_method=1,reg=0)
aligners.fit(nalign) 
ndecode_new = aligners.transform(ndecode[0],0)
"""

print(f'{c.time()}: Done loading data')


from fmralign.template_alignment import TemplateAlignment 

if False:
    aligners= TemplateAlignment('scaled_orthogonal',clustering=clustering,n_jobs=1,n_iter=2,scale_template=False,include_current_subject=True,alignment_kwargs={'scaling':True})
    aligners.fit(nalign) 
    ndecode_new = aligners.transform(ndecode[0],0)
    print(ndecode_new[0:2,0])
    print(aligners.estimators[0].fit_[0].R[0:2,0:2]) 

    print(f'{c.time()}: Done fitting alignment 1')


from fmralign import template_alignment
aligners2= TemplateAlignment('scaled_orthogonal',clustering=clustering,alignment_kwargs={'scaling':True})
aligners2.make_template(nalign,n_iter=1,do_level_1=False,normalize_imgs='zscore',normalize_template='zscore',remove_self=False,level1_equal_weight=False)
aligners2.fit_to_template(nalign)
ndecode_new2 = aligners2.transform(ndecode[0],0)
print(ndecode_new2[0:2,0])
print(aligners2.estimators[0].fit_[0].R[0:2,0:2])

print(f'{c.time()}: Done fitting alignment 2')

#aligners.template=get_template(clustering,nalign)
#aligners.fit_to_template(nalign)


"""
#Generic brain plot for figure
import hcpalign_utils as hutils
import numpy as np
p=hutils.surfplot('')
x=hutils.makesurfmap([])

for i in np.arange(0,1,0.2):
    x[:]=i
    p.plot(x,vmin=0,vmax=1,cmap='Greys')
"""

"""
n=[5,10,20,50,100]
anat=[.68, .84, .88, .91, .93]
temp=[.81,.91,.87,.88,.89]
temp_niter1 = [.79, .91, .94, .93, .95]

temp_restFC=[0.88,0.89,0.92] #don't have 0 and 100 subs values yet

import matplotlib.pyplot as plt
import matplotlib

fig,ax=plt.subplots(1)
ax.plot(n[1:],anat[1:],'k-o',markersize=8)
ax.plot(n[1:],temp_niter1[1:],'r-o',markersize=8)
#ax.plot(n[1:-1],temp_restFC,'b-o',markersize=8)
ax.set_xlabel('Number of subjects')
ax.set_ylabel('Classification accuracy')
#ax.legend(['Standard co-registration', 'Functional alignment - movie', 'Functional alignment - rsfMRI'])
ax.legend(['Standard co-registration', 'Functional alignment'])
ax.set_xscale('log')
ax.set_xticks(n[1:])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()
"""

'''
import hcp_utils as hcp
import matplotlib.pyplot as plt
import numpy as np
import os, pickle, warnings, itertools
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse
import hcpalign_utils as hutils
from hcpalign_utils import ospath
import matplotlib.pyplot as plt, tkalign_utils as tutils
from tkalign_utils import values2ranks as ranks, regress as reg, identifiability as ident, count_negs
from joblib import Parallel, delayed

print(hutils.memused())
c=hutils.clock()


import socket
hostname=socket.gethostname()
if hostname=='DESKTOP-EGSQF3A': #home pc
    tckfile= 'tracks_5M_sift1M_200k.tck' #'tracks_5M_sift1M_200k.tck','tracks_5M.tck' 
else: #service workbench
    tckfile='tracks_5M_1M_end.tck'

sift2=not('sift' in tckfile) #default True
MSMAll=False

pre_hrc_fwhm = 0
align_nparcs=300
align_labels=hutils.Schaefer(align_nparcs)
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)

sub = hutils.subs[0]

print(f'{c.time()}: Get highres connectomes and downsample',end=", ")
par_prefer_hrc='threads'        
hrs = hutils.get_highres_connectomes(c,[sub],tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefer_hrc,n_jobs=-1)
hr=hrs[0]

print(f'{c.time()}: Get geodesic distances')
from make_gdistances_full import get_saved_gdistances_full
gdists = get_saved_gdistances_full(sub,'midthickness')
gdistsL = gdists[hcp.struct.cortex_left,hcp.struct.cortex_left].astype(np.float32)
print(f'{c.time()}: Get geodesic distances done', end=", ")


def fulldistance_to_gaussian(fwhm):
    from Connectome_Spatial_Smoothing import CSS as css
    sigma=css._fwhm2sigma(fwhm)    
    gaussian = np.exp(-(gdistsL**2 / (2 * (sigma ** 2))))
    from sklearn.preprocessing import normalize
    gaussian=normalize(gaussian,norm='l1',axis=0) #so each column has sum 1.
    return gaussian


for pre_hrc_fwhm in [2,3,7,10]:
    smoother=sparse.load_npz(ospath(f'{hutils.intermediates_path}/smoothers/100610_{pre_hrc_fwhm}_0.01.npz')).astype(np.float32) 
    print(f'{c.time()}: {pre_hrc_fwhm}, Smooth hrc')
    hr2 = hutils.smooth_highres_connectomes([hr],smoother)[0].astype(np.float16)
    hr2L = hr2[hcp.struct.cortex_left,:][:,hcp.struct.cortex_left].astype(np.float32).toarray()

    print(f'{c.time()}: {pre_hrc_fwhm}, Make scatter')
    n = int(1e5) #get this many random points. Remove non-zero weights and plot
    n2 = int(1e3) #Of the non-zero weights, only plot this many
    inds_i = np.random.randint(0,29696,n) #get n random integers, each between 0 and 29696
    inds_j = np.random.randint(0,29696,n)
    dists = gdistsL[inds_i,inds_j]
    wts = hr2L[inds_i,inds_j]
    fig,ax=plt.subplots()
    non_zeros = np.nonzero(wts)
    dists = dists[non_zeros]
    wts = wts[non_zeros]
    inds = np.random.randint(0,len(dists),n2)

    logwts = np.log10(wts)
    ax.scatter(dists[inds],logwts[inds],s=2)

    #ax.scatter(dists[non_zeros],wts[non_zeros],s=2) 
    ax.set_xlabel('Geodesic distance')
    ax.set_ylabel('Log10(Weight)')
    ax.set_title(f'smooth fwhm={pre_hrc_fwhm}, n={n}')

    #fit linear regression to predict wts based on dists, and return R^2
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(dists[inds].reshape(-1,1),logwts[inds])
    print(f'{pre_hrc_fwhm}: R^2 = {reg.score(dists[inds].reshape(-1,1),logwts[inds]):.2f}')


print(f'{c.time()} done')
plt.show(block=False)
'''