"""
env py390
Script for investigating gyral bias in fMRI
See bottom of tkalign.py for additional code for plotting node strength (no. of streamlines) in diffusion data

TODO:
Replace nextcorr with corr to all neighbours
Try correlation with 10th closest vertex or 10 closest vertices, or all vertices within X mm radius, instead of nearest vertex
Is the bias due to distance between vertices or independent of that? Get spatial map of spatial autocorrelation when we look at correlation with all vertices within X mm radius
Look for gyral bias in volume data

"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
import biasfmri_utils as butils

project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
surface = 'fsLR32k' #'fsaverage5','fsLR32k'


'''
def text_file_to_npy(filepath):
    """
    Open a text file. Save the numbers as a numpy array in a .npy file
    """
    data = np.loadtxt(filepath+'.txt',dtype=np.float32)
    np.save(filepath,data)

[text_file_to_npy(ospath(f'{project_path}/fsaverage10k_timeseries_0{i}')) for i in [0,1,2]]
[text_file_to_npy(ospath(f'{project_path}/fsLR32k_timeseries_0{i}')) for i in [0,1,2]]
[text_file_to_npy(ospath(f'{project_path}/fsaverage_10k_medial_wall_{hemi}_masked')) for hemi in ['lh','rh']]
'''

#Plot fake data on surface
if surface == 'fsaverage5':
    vertices,faces  = butils.get_fsaverage_mesh('white')
    mask = butils.get_fsaverage5_mask()
    random_data_prefix = 'fsaverage10k_timeseries_'
elif surface == 'fsLR32k':
    import hcp_utils as hcp
    vertices,faces = hcp.mesh['pial']
    mask = hutils.get_fsLR32k_mask()
    random_data_prefix = 'fsLR32k_timeseries_'

mesh = (vertices,faces)
p = hutils.surfplot('',mesh=mesh,plot_type = 'open_in_browser')

random_data_path = ospath(f'{project_path}/{random_data_prefix}00.npy')
im = np.load(random_data_path).astype(np.float32)
nearest_vertices,nearest_distances = butils.get_nearest_vertices(mesh,mask)


import tkalign_utils as tutils
im_nextcorr = tutils.get_corr_with_neighbour(nearest_vertices,im)
im_nextcorr[np.isnan(im_nextcorr)] = 0 #replace NaNs with 0s

im_nextcorr[im_nextcorr<0] = 0

p.plot(butils.fillnongray(im[500,:],mask))
p.plot(butils.fillnongray(im_nextcorr,mask))
p.plot(butils.fillnongray(nearest_distances,mask))
plt.scatter(nearest_distances,im_nextcorr,1)
plt.show(block=False)

if surface =='fsaverage5':
    #Plot curvature and sulcal depth
    sulc = butils.get_fsaverage_scalar('sulc')
    curv = butils.get_fsaverage_scalar('curv')
    area = butils.get_fsaverage_scalar('area')
    thick = butils.get_fsaverage_scalar('thick')
    for scalar in [sulc,curv,area,thick]:
        p.plot(scalar)
    print('Correlations between ims_nextcorr and:')
    print(f'sulc: {np.corrcoef(im_nextcorr,sulc[mask])[0,1]:.2f}')
    print(f'curv: {np.corrcoef(im_nextcorr,curv[mask])[0,1]:.2f}')
    print(f'area: {np.corrcoef(im_nextcorr,area[mask])[0,1]:.2f}')
    print(f'thick: {np.corrcoef(im_nextcorr,thick[mask])[0,1]:.2f}')

assert(0)




### SETTABLE PARAMETERS ###
sub_slice = slice(0,3)
img_type = 'rest' #'movie' or 'rest'
MSMAll = False
this_parc = 1 #which parcel for within-parcel analysis

### FUNCTIONS ###

def make_random_fmri_data(nsubjects=50,ntimepoints=1000,shape=(91,109,91)):
    """
    Generate random fMRI data for nsubjects, each with shape given by shape+[ntimepoints]. Use random numbers drawn from a normal distribution.
    Parameters
    ----------
    nsubjects : int
    ntimepoints : int
    shape : tuple
        e.g. (91,100,91)
    """
    return [np.random.randn(*shape,ntimepoints) for i in range(nsubjects)]


def fourier_randomize(img):
    """
    Fourier transform each vertex time series, randomize the phase of each coefficient, and inverse Fourier transform back. Use the same set of random phase offsets for all vertices. This will randomize while preserving power spectrum of each vertex time series, and the spatial autocorrelation structure of the data.
    Parameters
    ----------
    img : 2D numpy array (ntimepoints,nvertices)
    """
    nvertices = img.shape[1]
    ntimepoints = img.shape[0]
    fourier = np.fft.rfft(img,axis=0)
    rand_uniform_numbers = np.tile(np.random.rand(fourier.shape[0],1),(nvertices))
    random_phases = np.exp(2j*np.pi*rand_uniform_numbers)
    fourier = fourier*random_phases
    img = np.fft.irfft(fourier,axis=0)
    return img

def parcellate(img,parc_matrix):
    """
    Parcellate a 2D image (ntimepoints,nvertices) into a 2D image (ntimepoints,nparcels) by averaging over vertices in each parcel
    Parameters
    ----------
    img : 2D numpy array (ntimepoints,nvertices)
    parc_matrix : 2D numpy array (nparcels,nvertices) with 1s and 0s    
    """
    parcel_sums = img @ parc_matrix.T #sum of all vertices in each parcel
    nvertices_sums=parc_matrix.sum(axis=1) #no. of vertices in each parcel
    nvertices_sum_expanded = np.tile(nvertices_sums,(1,img.shape[0])).T
    return parcel_sums/nvertices_sum_expanded

### PREPARATION ###
np.random.seed(0)
p=hutils.surfplot('',plot_type='open_in_browser')
c=hutils.clock()
hcp_folder=hutils.hcp_folder
intermediates_path=hutils.intermediates_path
results_path=hutils.results_path

subjects=hutils.all_subs[sub_slice]
nsubjects = len(subjects) #number of subjects

parc_string='S300'
parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
parc_matrix = hutils.parcellation_string_to_parcmatrix(parc_string)
nparcs = parc_labels.max()+1

print(f'{c.time()}: Get fMRI data')      
runs_x = [0]
runs_y = [1]
imsx,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs_x,fwhm=0,clean=True,MSMAll=MSMAll)
imsy,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs_y,fwhm=0,clean=True,MSMAll=MSMAll)

print(f'img_type: {img_type}, MSMAll: {MSMAll}, subs {sub_slice}, parc {parc_string}, test is runs {runs_x}, retest is runs {runs_y}')


ims = imsx+imsy #concatenate lists containing first and second run data. ims will be [sub0run0,sub1run0,...,sub0run1,sub1run1,...]

"""
print(f'{c.time()}: Fourier randomize')
ims = Parallel(n_jobs=-1,prefer='processes')(delayed(fourier_randomize)(img) for img in ims)
"""

print(f'{c.time()}: Parcellate')    
ims_parc = Parallel(n_jobs=-1,prefer='processes')(delayed(parcellate)(im,parc_matrix) for im in ims) #list (nsubjects) of parcellated time series (ntimepoints,nparcels)
ims_singleparc = [im[:,parc_labels==1] for im in ims] #list(nsubjects) of single-parcel time series (ntimepoints,nverticesInParcel)


print(f'{c.time()}: Calculate FC')   
ims_pfc = Parallel(n_jobs=-1,prefer='processes')(delayed(np.corrcoef)(im.T) for im in ims_parc) #list (nsubjects) of parcellated FC Matrices (nparcels,nparcels)
ims_sfc = Parallel(n_jobs=-1,prefer='processes')(delayed(np.corrcoef)(im.T) for im in ims_singleparc) #list (nsubjects) of within-single-parcel FC Matrices (nvertices,nvertices)

ims_pfcv = [i.ravel() for i in ims_pfc] #list (nsubjects) of parcellated FC matrices vectorized, shape (nparcls*nparcels,)
ims_sfcv = [i.ravel() for i in ims_sfc] #list (nsubjects) of single parcel FC matrices vectorized, shape (nverticesInParcel*nverticesInParcel,)

#re-split into run 1 and run 2
ims_pfcvx = ims_pfcv[0:nsubjects]
ims_pfcvy = ims_pfcv[nsubjects:]
ims_sfcvx = ims_sfcv[0:nsubjects]
ims_sfcvy = ims_sfcv[nsubjects:]

import tkalign_utils as tutils

#Get correlation between test and retest scans, across all subject pairs, for parcellated FC
corrs = tutils.ident_plot(ims_pfcvx,'test',ims_pfcvy,'retest',normed=False) 
plt.show(block=False)
print(f'Parcellated FC: test-retest identifiability is {tutils.identifiability(corrs):.2f}%')

#Correlation between test and retest scans, across all subject pairs, for single-parcel FC
corrs = tutils.ident_plot(ims_sfcvx,'test',ims_sfcvy,'retest',normed=False)
plt.show(block=False)
print(f'Single-parcel FC: test-retest identifiability is {tutils.identifiability(corrs):.2f}%')


### GET SPATIAL MAPS ###

import getmesh_utils
import tkalign_utils as tutils
meshes = [getmesh_utils.get_verts_and_triangles(subject,'white') for subject in subjects]
sulcs = [hutils.get_sulc(i) for i in subjects] #list (subjects) of sulcal depth maps

print(f'{c.time()}: Get maxcorr maps', end=", ")
ims_maxcorr = Parallel(n_jobs=-1,prefer='processes')(delayed(tutils.get_max_within_parcel_corrs)(img,parc_labels,nparcs) for img in ims) #list (subjects) of maxcorr maps. For each vertex, maximum correlation with any other vertex in the same parcel
print(f'{c.time()}: Get nearest vertices', end=", ")  
temp = Parallel(n_jobs=-1,prefer='processes')(delayed(hutils.get_nearest_vertices)(meshes[i]) for i in range(nsubjects))
nearest_vertices,nearest_distances = [list(item) for item in zip(*temp)]
print(f'{c.time()}: corr neighbour start', end=", ")   
ims_nextcorr = Parallel(n_jobs=-1,prefer='processes')(delayed(tutils.get_corr_with_neighbour)(nearest_vertices[i],ims[i]) for i in range(nsubjects))

fig,axs = plt.subplots(2)
ax = axs[0]
ax.scatter(nearest_distances[0],ims_nextcorr[0],1)
ax.set_xlabel('Distance to nearest vertex')
ax.set_ylabel('Correlation with nearest vertex')
ax = axs[1]
ax.scatter(nearest_distances[0],sulcs[0],1)
ax.set_xlabel('Distance to nearest vertex')
ax.set_ylabel('Sulcal depth')
plt.show(block=False)


"""
print(f'{c.time()}: Get temporal autocorrs', end=", "
ims_autocorr = Parallel(n_jobs=-1,prefer='processes')(delayed(hutils.temporal_autocorr)(i,1) for i in ims)     
print(f'{c.time()}: Get vertex areas start', end=", ")
vertex_areas = Parallel(n_jobs=-1,prefer='processes')(delayed(hutils.get_vertex_areas)(meshes[i]) for i in range(nsubjects))
"""

### FURTHER ANALYSES ###
'''
structdata = sulcs
funcdata = ims_nextcorr

cc = np.zeros((nsubjects,nsubjects,nparcs),dtype=np.float32) #corrs bw structdata from one subject and funcdata from another subject, for a given parcel
for subject_D in range(nsubjects):
    for subject_R in range(nsubjects):
        for nparc in range(nparcs):
            structdatum = structdata[subject_D][parc_labels==nparc]
            funcdatum = funcdata[subject_R][parc_labels==nparc]
            correlation = np.corrcoef(structdatum,funcdatum)[0,1]
            cc[subject_D,subject_R,nparc] = correlation
            """
            if subject_D==subject_R and nparc<5: #TRY THIS. Scatterplot of structdatum and funcdata. Rows are subjects, columns are parcels
                axs[subject_D,nparc].scatter(structdatum,funcdatum,1)
                axs[subject_D,nparc].set_title(f"r={correlation:.2f}")
            """
#fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.2, wspace=0.2)
#plt.show(block=False)
corrs = np.diagonal(cc).T #corrs bw structdata and funcdata for each subject (same subject for both) and parcel
corrsm = np.mean(corrs,axis=0)
cci=tutils.identifiability(cc)
'''

### COOL PLOTS ###
norm = lambda x: hutils.subtract_parcelmean(x,align_labels)
for mesh_sub in [0]:
    for func_sub in range(nsubjects):
        print(f"sub {func_sub} on mesh {mesh_sub}", end = ", ")
        p.mesh = meshes[mesh_sub]
        p.plot(sulcs[func_sub],cmap='bwr')
        p.plot((ims_nextcorr[func_sub]),cmap='inferno')
        p.plot(ims_maxcorr[func_sub],cmap='inferno')