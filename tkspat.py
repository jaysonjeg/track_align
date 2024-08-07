import os, pickle, warnings, itertools
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse
import hcpalign_utils as hutils
from hcpalign_utils import sizeof, ospath
import matplotlib.pyplot as plt, hcp_utils as hcp, tkalign_utils as tutils
from tkalign_utils import full, values2ranks as ranks, regress as reg, identifiability as ident
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import spearmanr as sp
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

align_nparcs=300
nparcs=301
nblocks=5

def get_a(z):
    b,a=tutils.load_a(z)
    return a
def reshape(a):
    return np.transpose( np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs)) , (0,1,3,2)) #Now a is n_subs_test * n_subs_test * nparcs * nblocksperparc
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
def count_negs_for_each_parc(a):
    return [tutils.count_negs(a[:,:,i,:]) for i in range(nparcs)]
def corr(x):
    #Correlations between columns in x, averaged across all pairs of columns (ignoring corr between a col and itself)
    values = np.corrcoef(x)
    non_ones = values!=1
    return np.mean(values[non_ones])

zs=[f'0-50_rest_{string}.npy' for string in ['0-10','10-20','20-30','30-40','40-50']]

zs = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/old/maxhpmult/{i}') for i in zs]

a = [get_a(i) for i in zs]

an=[reshape(tutils.subtract_nonscrambled_from_a(i)) for i in a]

anc=np.concatenate(an,axis=0) 

ax=[get_vals(i) for i in an]
ancx=get_vals(anc)

at=[tstat(i) for i in ax]
anct=tstat(ancx)

am=[np.mean(i,axis=-1) for i in ax]
ancm=np.mean(ancx,axis=-1)

anN=[count_negs_for_each_parc(i) for i in an]
ancN = count_negs_for_each_parc(anc)


print(np.corrcoef(at))
print(np.corrcoef(am))
print(np.corrcoef(anN))


#Plot spatial distribution of struct-func linkage
p=hutils.surfplot(hutils.results_path,plot_type='open_in_browser')
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)
ancN_cortex = ancN @ align_parc_matrix
p.plot(ancN_cortex,savename='ancN',vmin = min(ancN_cortex))
p.vmin=None
p.plot([-i for i in ancm] @ align_parc_matrix,savename='-ancm')
p.plot([-i for i in anct] @ align_parc_matrix,savename='-anct')

#Plot parcel sizes as # of vertices (?confounder)
parc_sizes=np.array(align_parc_matrix.sum(axis=1)).squeeze()
parc_sizes[0]=parc_sizes.mean()
p.plot(parc_sizes @ align_parc_matrix, savename='parc_sizes')

#Plot aligner scale factor (?confounder)
"""
alignfile = 'hcpalign_movie_temp_scaled_orthogonal_10-4-7_TF_0_0_0_FFF_S300_False'
aligner_file = f'{hutils.intermediates_path}/alignpickles/{alignfile}.p'
all_aligners = pickle.load( open( ospath(aligner_file), "rb" ))
scales = np.vstack( [[all_aligners.estimators[i].fit_[nparc].scale for nparc in range(nparcs)] for i in range(len(all_aligners.estimators))] )   
scales_mean=scales.mean(axis=0)    
log_scales_mean=np.log10(scales_mean)
log_scales_mean_adj = log_scales_mean - log_scales_mean.min()
p.plot(log_scales_mean_adj @ align_parc_matrix, savename='log_scales_mean_adj')  
"""

#Plot vertex areas (?confounder)
path='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\old_intermediates\\vertex_areas\\vert_area_100610_white.npy'
vertex_areas=np.load(path)
total_vertex_areas_parc = align_parc_matrix @ vertex_areas 
mean_vert_areas = total_vertex_areas_parc / parc_sizes
p.plot(mean_vert_areas @ align_parc_matrix,'mean_vert_areas_parc')
p.plot(total_vertex_areas_parc @ align_parc_matrix,'total_vertex_areas_parc')

#Plot mesh SFM

path='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\old_intermediates\\SFM\\SFM_white_sub100610.pickle'
import pickle
sfms=pickle.load(open(path,"rb"))[0]
curv_av=[np.trace(i) for i in sfms]
curv_av_abs=np.abs(curv_av)
curv_gauss=[np.linalg.det(i) for i in sfms]
#p.plot(curv_av_abs,'curv_av_abs')
curv_av_parc = ( align_parc_matrix @ curv_av ) / parc_sizes
curv_av_abs_parc = ( align_parc_matrix @ curv_av_abs ) / parc_sizes


#Are spatial variations in SC-func linkage associated with these confounders?
"""
print(f'Corr between ancN and scales_mean is {np.corrcoef(ancN,scales_mean)[0,1]}')
print(f'Corr between ancN and parc_sizes is {np.corrcoef(ancN,parc_sizes)[0,1]}')
print(f'Corr between ancN and mean_vert_areas is {np.corrcoef(ancN,mean_vert_areas)[0,1]}')
print(f'Corr between ancN and total_parcel_area is {np.corrcoef(ancN,total_vertex_areas_parc)[0,1]}')
print(f'Corr between ancN and average curvature is {np.corrcoef(ancN,curv_av_parc)[0,1]}')
print(f'Corr between ancN and abs-value of average curvature is {np.corrcoef(ancN,curv_av_abs_parc)[0,1]}')
"""



parcellation_string = 'S300'
align_labels=hutils.parcellation_string_to_parcellation(parcellation_string)
align_parc_matrix=hutils.parcellation_string_to_parcmatrix(parcellation_string)
nparcs=align_parc_matrix.shape[0]

import getmesh_utils
subject_id = '100610'
mesh = getmesh_utils.get_verts_and_triangles(subject_id,'white')
p.mesh = mesh

sulc = hutils.get_sulc(subject_id)


nsubjects=1
axis = 0
corrs = np.zeros((nsubjects,nparcs),dtype=np.float32) #corr bw Rsums and hrdu for given subject and parcel
for subject in range(nsubjects):
    for nparc in range(nparcs):
        areas = vertex_areas[align_labels==nparc]
        Dsums = sulc[align_labels==nparc]
        corrs[subject,nparc] = np.corrcoef(areas,Dsums)[0,1]

print(np.sum(corrs<0)/len(np.squeeze(corrs)))
plt.hist(np.squeeze(corrs))
plt.show(block=False)