"""
Script for investigating gyral bias in fMRI
Use env py390
Shows the bias using noise data in a single subject

See bottom of tkalign.py for additional code for plotting node strength (no. of streamlines) in diffusion data
Visualization might be best with surface_visual==inflated or white, subtract_parcelmeans_for_visual=True


"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
import biasfmri_utils as butils
import nilearn.plotting as plotting
import nibabel as nib
import pickle
import brainmesh_utils as bmutils

c = hutils.clock()

#Set paths
hcp_folder=hutils.hcp_folder
intermediates_path=hutils.intermediates_path
results_path=hutils.results_path
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
biasfmri_intermediates_path = ospath(f'{project_path}/intermediates')

### SETTABLE PARAMETERS ###
mesh_template = 'fsLR32k' #'fsaverage5','fsLR32k'
which_subject = '100610' #'100610','standard'
which_subject_visual = which_subject #which_subject or a specific subject eg 102311
surface = 'midthickness' #which surface for calculating neighbour distances, e.g. 'white','inflated','pial','midthickness'
surface_visual = 'white' #which surface for visualization
folder='MNINonLinear' #More parameters for getmesh_utils.get_verts_and_triangles. 'MNINonLinear' (default) or 'T1w'
version='fsaverage_LR32k' #'native', 'fsaverage_LR32k' (default) or '164k'

noise_source = 'surface' #'volume' or 'surface'. 'volume' means noise data in volume space projected to 'surface'. 'surface' means noise data generated in surface space
smooth_data_fwhm = 0 #mm of surface smoothing. Try 0 or 2
MSMAll=False
which_neighbours = 'local' #'local','nearest','distant'
distance_range=(3,5) #Only relevant if which_neighbours=='distant'. Geodesic distance range in mm, e.g. (0,4), (2,4), (3,5), (4,6)
subtract_parcelmeans_for_visual = False #whether surface plots will subtract parcel-specific mean values. This is useful for looking at within-parcel correlations and can improve visualization of gyral bias
load_neighbours = False
save_neighbours = False


print(f"mesh_template: {mesh_template} \nsurface: {surface} \nwhich_subject: {which_subject} \nsurface_visual: {surface_visual} \nwhich_subject_visual: {which_subject_visual} \nnoise_source: {noise_source} \nsmooth_data_fwhm: {smooth_data_fwhm} \nwhich_neighbours: {which_neighbours} \nsubtract_parcelmeans_for_visual: {subtract_parcelmeans_for_visual}\n")



### Get mesh ###
print(f'{c.time()}: Get mesh start')
if mesh_template == 'fsaverage5':
    mask = butils.get_fsaverage5_mask()
elif mesh_template == 'fsLR32k':
    mask = hutils.get_fsLR32k_mask()
if which_subject == 'standard':
    if mesh_template == 'fsaverage5':
        vertices,faces = butils.get_fsaverage_mesh(surface)
        vertices_visual,faces_visual  = butils.get_fsaverage_mesh(surface_visual)
        noise_data_prefix = 'fsaverage10k_timeseries_00'
    elif mesh_template == 'fsLR32k':
        import hcp_utils as hcp
        vertices,faces = hcp.mesh[surface]
        vertices_visual,faces_visual = hcp.mesh[surface_visual]
        noise_data_prefix = 'fsLR32k_timeseries_00'
elif which_subject != 'standard':
    if mesh_template == 'fsLR32k':
        vertices,faces = bmutils.hcp_get_mesh(which_subject,surface,folder=folder,version=version)
        vertices_visual,faces_visual = bmutils.hcp_get_mesh(which_subject_visual,surface_visual,folder=folder,version=version)
        if not(folder=='MNINonLinear' and version=='fsaverage_LR32k'):
            print('Setting mask to all 1s')
            mask = np.zeros(vertices.shape[0],dtype=bool)
            mask[:]=True
        noise_data_prefix = f'{which_subject}.32k_fs_LR.func'
    else:
        assert('Do not currently have fsaverage5 mesh for individual subjects')
p = hutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')
#mesh = (hutils.cortex_64kto59k(vertices),hutils.cortex_64kto59k_for_triangles(faces)) #downsample from 64k to 59k

mesh = (vertices[mask],butils.triangles_removenongray(faces,mask))

### Get neighbour distances ###

neighbour_vertices, neighbour_distances = butils.get_subjects_neighbour_vertices(c, which_subject,surface,mesh, biasfmri_intermediates_path, which_neighbours, distance_range, load_neighbours, save_neighbours,MSMAll)

'''
#See on other meshes other than fsLR32k (e.g. Native, 164k, T1w folder)
from scipy import stats
print(f'{vertices.shape[0]} vertices')
print(vertices[0:2,:])
sulc = butils.get_sulc(which_subject,version=version)[mask]
fig,axs=plt.subplots(2)
ax = axs[0]
ax.hist(neighbour_distances,bins=100)
ax = axs[1]
ax.scatter(sulc,neighbour_distances,1,alpha=0.05,color='k')
ax.set_title(f'Correlation: {np.corrcoef(sulc,neighbour_distances)[0,1]:.3f}')
p.plot(butils.fillnongray(sulc,mask),cmap='inferno')
p.plot(butils.fillnongray(neighbour_distances,mask),cmap='inferno')
plt.show(block=False)
assert(0)
'''

### GET NOISE DATA AND CALCULATE NEIGHBOUR CORRELATIONS ###
print(f'{c.time()}: Get noise data start')
if noise_source=='volume':
    #Get noise data in volume space which has been projected to surface
    noise_data_path = ospath(f'{project_path}/{noise_data_prefix}.npy')
    noise = np.load(noise_data_path).astype(np.float32) #noise data in volume space projected to surface (vol 2 surf)
    if noise.shape[1] > mesh[0].shape[0]:
        print("Removing non-gray vertices from noise fMRI data")
        noise = noise[:,mask]
elif noise_source=='surface':
    #Generate noise data in surface space
    noise = np.random.randn(1000,mesh[0].shape[0]).astype(np.float32)

if smooth_data_fwhm>0:
    print(f'{c.time()}: Smooth data start')
    #smoother = hutils.make_smoother(which_subject,smooth_data_fwhm)
    #noise = smoother(noise)
    fwhm_values_for_gdist = np.array([3,5,10]) #fwhm values for which geodesic distances have been pre-calculated
    fwhm_for_gdist = fwhm_values_for_gdist[np.where(fwhm_values_for_gdist>smooth_data_fwhm)[0][0]] #find smallest value greater than fwhm in the above list
    skernel = butils.get_smoothing_kernel(which_subject,surface,fwhm_for_gdist,smooth_data_fwhm,MSMAll,mesh_template)
    noise = butils.smooth(noise,skernel)


print(f'{c.time()}: Get corr with neighbours start')
noise_adjcorr = butils.get_corr_with_neighbours(neighbour_vertices,noise)
noise_adjcorr[np.isnan(noise_adjcorr)] = 0 #replace NaNs with 0s
#noise_adjcorr[noise_adjcorr<0] = 0 #set negative correlations to 0
print(f'{c.time()}: Get corr with neighbours end')

if mesh_template == 'fsLR32k':
    if which_subject == 'standard':
        sulc = -hcp.mesh.sulc[mask]
    else:
        sulc = butils.get_sulc(which_subject)[mask]
if subtract_parcelmeans_for_visual:
    parc_string='S300'
    parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
    parc_matrix = hutils.parcellation_string_to_parcmatrix(parc_string)
    nparcs = parc_labels.max()+1
    noise_adjcorr = butils.subtract_parcelmean(noise_adjcorr,parc_matrix)
    neighbour_distances = butils.subtract_parcelmean(neighbour_distances,parc_matrix)
    if mesh_template == 'fsLR32k' and which_subject !='standard':
        sulc = butils.subtract_parcelmean(sulc,parc_matrix)

### PLOTTING ###

#p.plot(butils.fillnongray(im[500,:],mask))
p.plot(butils.fillnongray(noise_adjcorr,mask),cmap='inferno')
p.plot(butils.fillnongray(neighbour_distances,mask),cmap='inferno')

"""
#Visualize with different subjects' surfaces
neighbour_distances_full = butils.fillnongray(neighbour_distances,mask)
for which_subject_visual in ['standard','100610']:
    if which_subject_visual == 'standard':
        import hcp_utils as hcp
        vertices_visual,faces_visual = hcp.mesh[surface_visual]
    else:
        import getmesh_utils
        vertices_visual,faces_visual = bmutils.hcp_get_mesh(which_subject_visual,surface_visual)
    p.mesh = (vertices_visual,faces_visual)
    p.plot(neighbour_distances_full)
    p.plot(sulc)
"""

fig,axs=plt.subplots(2,2)
ax=axs[0,0]
ax.scatter(neighbour_distances,noise_adjcorr,1,alpha=0.05,color='k')
ax.set_xlabel('Inter-vertex distance (mm)')
ax.set_ylabel('Correlation with neighbours')
ax.set_title(f'Correlation: {np.corrcoef(neighbour_distances,noise_adjcorr)[0,1]:.3f}')
if mesh_template == 'fsLR32k':
    ax=axs[0,1]
    #p.plot(butils.fillnongray(sulc,mask))
    ax.scatter(sulc,neighbour_distances,1,alpha=0.05,color='k')
    ax.set_xlabel('Sulcal depth')
    ax.set_ylabel('Inter-vertex distance (mm)')
    ax.set_title(f'Correlation: {np.corrcoef(sulc,neighbour_distances)[0,1]:.3f}')
    ax=axs[1,0]
    ax.scatter(sulc,noise_adjcorr,1,alpha=0.05,color='k')
    ax.set_xlabel('Sulcal depth')
    ax.set_ylabel('Correlation with neighbours')
    ax.set_title(f'Correlation: {np.corrcoef(sulc,noise_adjcorr)[0,1]:.3f}')
fig.tight_layout()

if mesh_template =='fsaverage5':
    #Plot curvature and sulcal depth
    sulc = butils.get_fsaverage_scalar('sulc')
    curv = butils.get_fsaverage_scalar('curv')
    area = butils.get_fsaverage_scalar('area')
    thick = butils.get_fsaverage_scalar('thick')
    for scalar in [sulc,curv,area,thick]:
        p.plot(scalar)
    print('Correlations between ims_nextcorr and:')
    print(f'sulc: {np.corrcoef(noise_adjcorr,sulc[mask])[0,1]:.2f}')
    print(f'curv: {np.corrcoef(noise_adjcorr,curv[mask])[0,1]:.2f}')
    print(f'area: {np.corrcoef(noise_adjcorr,area[mask])[0,1]:.2f}')
    print(f'thick: {np.corrcoef(noise_adjcorr,thick[mask])[0,1]:.2f}')
    print('Correlations between neighbour_distances and:')
    print(f'sulc: {np.corrcoef(neighbour_distances,sulc[mask])[0,1]:.2f}')
    print(f'curv: {np.corrcoef(neighbour_distances,curv[mask])[0,1]:.2f}')


plt.show(block=False)
assert(0)


##### PART TWO #####

### SETTABLE PARAMETERS ###
sub_slice = slice(0,3)
img_type = 'rest' #'movie' or 'rest'
MSMAll = False
this_parc = 1 #which parcel for within-parcel analysis


### PREPARATION ###
np.random.seed(0)
p=hutils.surfplot('',plot_type='open_in_browser')
c=hutils.clock()


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
ims = Parallel(n_jobs=-1,prefer='processes')(delayed(butils.fourier_randomize)(img) for img in ims)
"""

print(f'{c.time()}: Parcellate')    
ims_parc = Parallel(n_jobs=-1,prefer='processes')(delayed(butils.parcel_mean)(im,parc_matrix) for im in ims) #list (nsubjects) of parcellated time series (ntimepoints,nparcels)
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

import tkalign_utils as tutils
meshes = [bmutils.hcp_get_mesh(subject,'white') for subject in subjects]
sulcs = [butils.get_sulc(i)[mask] for i in subjects] #list (subjects) of sulcal depth maps

print(f'{c.time()}: Get maxcorr maps', end=", ")
ims_maxcorr = Parallel(n_jobs=-1,prefer='processes')(delayed(tutils.get_max_within_parcel_corrs)(img,parc_labels,nparcs) for img in ims) #list (subjects) of maxcorr maps. For each vertex, maximum correlation with any other vertex in the same parcel
print(f'{c.time()}: Get nearest vertices', end=", ")  
temp = Parallel(n_jobs=-1,prefer='processes')(delayed(butils.get_neighbour_vertices)(meshes[i]) for i in range(nsubjects))
nearest_vertices,nearest_distances = [list(item) for item in zip(*temp)]
print(f'{c.time()}: corr neighbour start', end=", ")   
ims_nextcorr = Parallel(n_jobs=-1,prefer='processes')(delayed(butils.get_corr_with_neighbours)(nearest_vertices[i],ims[i],parallelize=False) for i in range(nsubjects))

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