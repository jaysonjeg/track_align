"""
Code for looking at the consequences of gyral bias in fMRI data
Script for investigating gyral bias in fMRI
Use env py390
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
import hcp_utils as hcp

c = hutils.clock()

#Set paths
hcp_folder=hutils.hcp_folder
intermediates_path=hutils.intermediates_path
results_path=hutils.results_path
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
biasfmri_intermediates_path = ospath(f'{project_path}/intermediates')

### SETTABLE PARAMETERS
sub_slice = slice(0,2)
real_or_noise = 'real' # 'real' or 'noise
this_parc = 1 #which parcel for within-parcel analysis
parc_string='S300'
surface = 'midthickness' #which surface for calculating distances, e.g. 'white','inflated','pial','midthickness'
which_subject_visual = '100610' #which subject for visualization. '100610', 'standard'
surface_visual = 'white'

### PARAMETERS FOR NOISE OR REAL DATA
subjects=hutils.all_subs[sub_slice]
nsubjects = len(subjects)
print(f'{c.time()}: subs {sub_slice}, {real_or_noise}, parc {parc_string}, surface {surface}')
if real_or_noise == 'noise':
    noise_source = 'surface' #'volume' or 'surface'. 'volume' means noise data in volume space projected to 'surface'. 'surface' means noise data generated in surface space
    smooth_noise_fwhm = 2 #mm of surface smoothing. Try 0 or 2
    ntimepoints = 1000 #number of timepoints
elif real_or_noise == 'real':
        img_type = 'rest' #'movie' or 'rest'
        MSMAll = False
        runs_x = [0]
        runs_y = [1]
        print(f'{c.time()}: img_type: {img_type}, MSMAll: {MSMAll}, test is runs {runs_x}, retest is runs {runs_y}')

import getmesh_utils
if which_subject_visual =='standard':
    p=hutils.surfplot('',mesh = hcp.mesh[surface_visual], plot_type='open_in_browser')
else:
    vertices_visual,faces_visual = getmesh_utils.get_verts_and_triangles(which_subject_visual,surface_visual)
    p = hutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')

### GET DATA
mask = hutils.get_fsLR32k_mask() #boolean mask of gray matter vertices. Excludes medial wall
parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
parc_matrix = hutils.parcellation_string_to_parcmatrix(parc_string)
nparcs = parc_labels.max()+1

print(f'{c.time()}: Get meshes')
import getmesh_utils
meshes = [getmesh_utils.get_verts_and_triangles(subject,surface) for subject in subjects]
meshes = [(hutils.cortex_64kto59k(vertices),hutils.cortex_64kto59k_for_triangles(faces)) for vertices,faces in meshes] #downsample from 64k to 59k
all_vertices, all_faces = zip(*meshes)
sulcs = [butils.get_sulc(i) for i in subjects] #list (subjects) of sulcal depth maps

print(f'{c.time()}: Get fMRI data')   
if real_or_noise == 'noise':
    if noise_source=='volume':
        #Get noise data in volume space which has been projected to surface
        def get_vol2surf_noisedata(which_subject):
            noise_data_prefix = f'{which_subject}.32k_fs_LR.func'
            noise_data_path = ospath(f'{project_path}/{noise_data_prefix}.npy')
            noise = np.load(noise_data_path).astype(np.float32) #noise data in volume space projected to surface (vol 2 surf)
            if noise.shape[1] > 59412:
                print("Removing non-gray vertices from noise fMRI data")
                noise = noise[:,mask]
            return noise     
        ims = Parallel(n_jobs=-1,prefer='threads')(delayed(get_vol2surf_noisedata)(subject) for subject in subjects)
        assert(0) #need two of each of these noise sources
    elif noise_source=='surface':
        #Generate noise data in surface space
        ims = [np.random.randn(ntimepoints,59412).astype(np.float32) for i in range(nsubjects*2)]
    if smooth_noise_fwhm>0: #smooth the noise data using geodesic distances from subject-specific meshes
        fwhm_values_for_gdist = np.array([3,5,10]) #fwhm values for which geodesic distances have been pre-calculated
        fwhm_for_gdist = fwhm_values_for_gdist[np.where(fwhm_values_for_gdist>smooth_noise_fwhm)[0][0]] #find smallest value greater than fwhm in the above list
        print(f'{c.time()}: Generate smoothers start')
        skernels = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.get_smoothing_kernel)(subject,surface,fwhm_for_gdist,smooth_noise_fwhm) for subject in subjects)
        print(f'{c.time()}: Smooth noise data start')
        ims = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.smooth)(im,skernel) for im,skernel in zip(ims,skernels*2))
        print(f'{c.time()}: Smoothing finished')
elif real_or_noise == 'real':
    imsx,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs_x,fwhm=0,clean=True,MSMAll=MSMAll)
    imsy,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs_y,fwhm=0,clean=True,MSMAll=MSMAll)
    ims = imsx+imsy #concatenate lists containing first and second run data. ims will be [sub0run0,sub1run0,...,sub0run1,sub1run1,...]

assert(0)

### Bias in fMRI-based parcellation (do left hemisphere alone)

do_bias_parcellation = False
if do_bias_parcellation:

    nsubject = 0
    n_clusters = 100

    imgt = ims[nsubject].T
    sulc = sulcs[nsubject] #could put (nsubjects+1) to compare to the sulc map of a different subject

    faces = meshes[nsubject][1]
    print(f'{c.time()}: Faces to structural adjacency matrix')
    connectivity,edges = butils.faces2connectivity(faces)

    ngrayl = len(hcp.vertex_info.grayl) #left hemisphere only
    imgt = imgt[0:ngrayl,:]
    sulc_left = sulc[0:ngrayl] #left-sided sulcal depth map
    connectivity = connectivity[:,0:ngrayl][0:ngrayl,:]    
    edges_left = butils.find_edges_left(edges, ngrayl)

    print(f'{c.time()}: Agglomerative clustering')
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', connectivity=connectivity, linkage='ward').fit(imgt)
    labels = clustering.labels_

    #Remove edges which are outside the left hemisphere

    #Optional: use standard mesh sulc vaule, and pre-computed parcellation labels
    """
    sulc = -hcp.mesh.sulc[mask] #because this standard sulc values are flipped
    sulc_left = sulc[0:ngrayl] #left-sided sulcal depth map
    labels = hutils.parcellation_string_to_parcellation('S100')[0:ngrayl] #left-sided parcellation
    """

    print(f'{c.time()}: Find distance from parcel boundaries for each vertex')
    border,border_bool = butils.get_border_vertices(ngrayl,edges_left,labels)
    border_full = np.zeros(59412)
    border_full[0:ngrayl] = border
    parcels_full = np.zeros(59412)
    parcels_full[0:ngrayl] = labels
    sulc_border = sulc_left[border_bool] #sulcal depth values at the border of parcels
    sulc_nonborder = sulc_left[~border_bool] #sulcal depth values not at the border of parcels

    from scipy import stats
    ttest = stats.ttest_ind(sulc_border,sulc_nonborder)
    cohen_d = butils.get_cohen_d(sulc_border,sulc_nonborder)
    print(f"Sulcal depth at parcel borders {np.mean(sulc_border):.3f} vs non-border vertices {np.mean(sulc_nonborder):.3f}: cohens d {cohen_d:.3f}, t({ttest.df})={ttest.statistic:.3f}, p={ttest.pvalue:.3f}")

    p.plot(sulc)
    p.plot(border_full,cmap='bwr_r') #Plot border points
    p.plot(parcels_full,cmap='prism') #Plot parcellation

    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(figsize=(5,5))
    ax=axs
    vp = ax.violinplot([sulc_border,sulc_nonborder],showmeans=True,showmedians=False)
    ax.set_ylabel('Sulcal depth')
    ax.set_xticks([1,2],labels=['Border','Non-border'])
    ax.set_xlabel('Vertex location')
    """
    axs[1].scatter(sulc_left,border,1,alpha=0.05)
    axs[1].set_xlabel('Sulcal depth')
    axs[1].set_ylabel('Distance from parcel boundary')
    """
    fig.tight_layout()
    plt.show()

    assert(0)


### Parcellate
print(f'{c.time()}: Parcellate')    
ims_parc = Parallel(n_jobs=-1,prefer='processes')(delayed(butils.parcel_mean)(im,parc_matrix) for im in ims) #list (nsubjects) of parcellated time series (ntimepoints,nparcels)
ims_singleparc = [im[:,parc_labels==1] for im in ims] #subjects' single-parcel time series (ntimepoints,nverticesInParcel)
sulcs_parc = [butils.parcel_mean(sulc,parc_matrix) for sulc in sulcs] #subjects' parcellated sulcal depth maps (nparcels,)
sulcs_singleparc = [sulc[parc_labels==1] for sulc in sulcs] #subjects' single-parcel sulcal depth maps (nverticesInParcel,)

### Will the largest PCA components be biased towards the sulci?
do_pca = False
if do_pca:
    #calculate PCA components for each array in ims separately 
    print(f'{c.time()}: PCA start')
    from sklearn.decomposition import PCA
    nsubject = 0
    img = ims[nsubject]
    sulc = sulcs[nsubject]
    pca = PCA(n_components=50)
    pca.fit(img)
    pca.components_.shape #shape (ncomponents,nvertices)

    corrs_with_sulc = []
    for component in pca.components_:
        #p.plot(component)
        corrs_with_sulc.append(np.corrcoef(component,sulc)[0,1])
    print(f"How many components are gyral biased? {np.sum(np.array(corrs_with_sulc)>0)}/{len(corrs_with_sulc)}")
    assert(0)


### Parcel-mean values might be a bit biased towards the sulci within each parcel

do_parcelmean = False
if do_parcelmean:
    print(f'{c.time()}: Parcel-mean values might be a bit biased towards the sulci within each parcel')
    def corr_between_vertex_and_parcelmean(img, img_parc, parc_labels):
        """
        Given time series for all vertices, and parcel-mean time series, compute the correlation between each vertex and its parcel's mean
        Parameters:
        ----------
        img: np.array (ntimepoints,nvertices)
            Time series for all vertices
        img_parc: np.array (ntimepoints,nparcels)
            Parcel-mean time series
        parc_labels: np.array (nvertices)
            Parcel labels for each vertex
        Returns:
        -------
        np.array (nvertices,)
            Correlation between each vertex and its parcel's mean
        """
        nvertices = img.shape[1]
        corrs = np.zeros(nvertices)
        for i in range(nvertices):
            corrs[i] = np.corrcoef(img[:,i],img_parc[:,parc_labels[i]])[0,1]
        return corrs

    corrs_vert_parc = Parallel(n_jobs=-1,prefer='threads')(delayed(corr_between_vertex_and_parcelmean)(img,img_parc,parc_labels) for img,img_parc in zip(ims,ims_parc)) #correlations between each vertex and its parcel's mean

    if which_subject_visual !='standard':
        cc2=butils.smooth(corrs_vert_parc[0],skernels[0]) 
    else:
        cc2 = corrs_vert_parc[0]

    cc2n = butils.subtract_parcelmean(cc2,parc_matrix)
    sulc0n = butils.subtract_parcelmean(sulcs[0],parc_matrix)

    p.plot(cc2)
    p.plot(sulc0n)

    np.corrcoef(cc2n,sulc0n)[0,1]   
    assert(0)

### Get neighbours and correlations
do_neighbours = True
if do_neighbours:
    which_neighbours = 'local' #'local','nearest','distant'
    distance_range=(3,5) #Only relevant if which_neighbours=='distant'. Geodesic distance range in mm, e.g. (0,4), (2,4), (3,5), (4,6)
    load_neighbours = True
    save_neighbours = False
    func = lambda subject,mesh: butils.get_subjects_neighbour_vertices(c, subject,surface,mesh,biasfmri_intermediates_path, which_neighbours, distance_range,load_neighbours, save_neighbours)
    print(f'{c.time()}: Get neighbour vertices')  
    temp = Parallel(n_jobs=-1,prefer='threads')(delayed(func)(subject,mesh) for subject,mesh in zip(subjects,meshes))
    all_neighbour_vertices, all_neighbour_distances = zip(*temp)
    print(f'{c.time()}: Get corr neighbour start', end=", ")   
    ims_adjcorr = Parallel(n_jobs=2,prefer='processes')(delayed(butils.get_corr_with_neighbours)(all_neighbour_vertices[i],ims[i],parallelize=False) for i in range(nsubjects))
    ims_adjcorr_parc = [butils.parcel_mean(im,parc_matrix) for im in ims_adjcorr] #subjects' parcel-mean local neighbourhood correlations (nparcels,)
    ims_adjcorr_singleparc = [im[parc_labels==1] for im in ims_adjcorr] #subjects' single-parcel local neighbourhood correlations (nverticesInParcel,)

    p.plot(ims_adjcorr[0])
    p.plot(ims_adjcorr[1])
    p.plot(butils.subtract_parcelmean(ims_adjcorr[0],parc_matrix))
    p.plot(butils.subtract_parcelmean(ims_adjcorr[1],parc_matrix))


    ### Is there a bias in the group mean local neighbourhood correlations?
    do_groupmean = False
    if do_groupmean:
        neighbour_distances_mean = np.mean(all_neighbour_distances,axis=0) #mean of all subjects' local neighbourhood distances
        ims_adjcorr_mean=np.mean(ims_adjcorr,axis=0) #mean of all subjects' local neighbourhood correlations
        p.plot(butils.fillnongray(ims_adjcorr_mean,mask))
        p.plot(butils.fillnongray(neighbour_distances_mean,mask))

    #Correlation between sulc and local correlations. Looking at parcel-mean data, or single-parcel data
    do_correlations=True
    if do_correlations:
        corrs_sulc_adjcorr_parc = [np.corrcoef(sulc,adjcorr)[0,1] for sulc,adjcorr in zip(sulcs_parc,ims_adjcorr_parc)]
        corrs_sulc_adjcorr_singleparc = [np.corrcoef(sulc,adjcorr)[0,1] for sulc,adjcorr in zip(sulcs_singleparc,ims_adjcorr_singleparc)]

        print('Correlation (within-subject, across parcels) of (parcel-averaged) sulcal depth vs local correlations')
        print(corrs_sulc_adjcorr_parc)

        print('Correlation (within-subject, within-parcel, across vertices) of sulcal depth vs local correlations')
        print(corrs_sulc_adjcorr_singleparc)

### Analyse reliability and identifiability of functional connectivity
do_FC = False
if do_FC:
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