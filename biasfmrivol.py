"""
env py390
Test for gyral bias in neighbourhood correlations in fMRI, in volume space
"""


import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
import biasfmri_utils as butils
import nilearn.plotting as plotting
import nibabel as nib

c = hutils.clock()

#Set paths
hcp_folder=hutils.hcp_folder
intermediates_path=hutils.intermediates_path
results_path=hutils.results_path
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"

"""
This code gets neighbourhood correlation map for a single subject's fMRI data, masked by a gray matter mask
Input data is single subject's resting state fMRI data as a nifti file
First we get that subject's aparc+aseg.nii.gz and find the 'cutoff value' to convert this segmentation map to a boolean mask of gray matter
Then we get the neighbourhood correlation map for the fMRI data, masked by the gray matter mask, and save it for visualization
"""
subject = '100610'
directions = ['LR','RL']
runs = ['1']
corrs_filename = f"corrs_{subject}_{''.join(directions)+''.join(runs)}"
corrs_volume_path = ospath(f'{project_path}\\intermediates\\restfmri_volume_corr\\{corrs_filename}.nii.gz')
corrs_vol2surf_folder = ospath(f'{project_path}/intermediates/restfmri_volume_corr_surf')

make_corrs_volume = False
if make_corrs_volume:
    nifti_filepaths = [ospath(f'{hcp_folder}/{subject}/MNINonLinear/Results/rfMRI_REST{run}_{direction}/rfMRI_REST{run}_{direction}_hp2000_clean.nii.gz') for direction in directions for run in runs]
    print(f'{c.time()}: Load images')
    nifti_images = [nib.load(filepath) for filepath in nifti_filepaths]
    #now concatenate these nifti images
    nifti_image = nib.concat_images(nifti_images,axis=3)

    segmentation_filepath = ospath(f'{hcp_folder}/{subject}/MNINonLinear/aparc+aseg.nii.gz')
    mask_image = nib.load(segmentation_filepath)
    from nilearn import image
    mask_image_resampled = image.resample_img(mask_image,nifti_image.affine)

    view_different_cutoffs = False
    if view_different_cutoffs:
        view = plotting.view_img(mask_image_resampled)
        view.open_in_browser()
        for cutoff in [20,50,100,200,500]:
            mask_image_resampled_bool = image.math_img(f"img>{cutoff}",img=mask_image_resampled)
            view = plotting.view_img(mask_image_resampled_bool)
            view.open_in_browser()
        assert(0)

    cutoff = 100
    mask_image_resampled_bool = image.math_img(f"img>{cutoff}",img=mask_image_resampled)

    print(f'{c.time()}: Get neighbourhood corrs start')
    corrs = butils.get_corr_with_neighbours_nifti(nifti_image,mask_image = mask_image_resampled_bool)
    print(f'{c.time()}: Get neighbourhood corrs end')

    nib.save(corrs,corrs_volume_path) #save neigbourhood correlations as a .nii.gz file
    view = plotting.view_img(corrs,bg_map = mask_image_resampled_bool)
    view.open_in_browser() #can also view using mrview from MRTrix

    plotting.plot_epi(corrs,display_mode='z',cut_coords=[0],cmap='gray')
    plotting.plot_stat_map(corrs,display_mode='z', cut_coords=[0])
    plt.show(block=False)

make_corrs_vol2surf = False
if make_corrs_vol2surf:
    surface_loc = ospath(f"{hcp_folder}/{subject}/MNINonLinear/fsaverage_LR32k")

    for hemi in ['L','R']:
        corrs_vol2surf_path = ospath(f'{corrs_vol2surf_folder}/{corrs_filename}_{hemi}.midthickness.32k_fs_LR.func.gii')
        string = f"wb_command -volume-to-surface-mapping {corrs_volume_path} {surface_loc}/{subject}.{hemi}.midthickness.32k_fs_LR.surf.gii {corrs_vol2surf_path} -ribbon-constrained {surface_loc}/{subject}.{hemi}.white.32k_fs_LR.surf.gii {surface_loc}/{subject}.{hemi}.pial.32k_fs_LR.surf.gii"
        os.system(string)

use_corrs_vol2surf = True
if use_corrs_vol2surf:
    import hcp_utils as hcp

    ### PARAMETERS
    which_subject_visual = subject
    surface_visual = 'white'

    ### SETUP
    import getmesh_utils
    if which_subject_visual =='standard':
        p=hutils.surfplot('',mesh = hcp.mesh[surface_visual], plot_type='open_in_browser')
    else:
        vertices_visual,faces_visual = getmesh_utils.get_verts_and_triangles(subject,surface_visual)
        p = hutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')
    parc_string = 'S300'
    parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
    parc_matrix = hutils.parcellation_string_to_parcmatrix(parc_string)

    ### GET THE DATA
    mask = hutils.get_fsLR32k_mask() 
    ngraysLR = [len(hcp.vertex_info.grayl),len(hcp.vertex_info.grayr)]
    corrs_vol2surf_path = ospath(f'{corrs_vol2surf_folder}/{corrs_filename}_L.midthickness.32k_fs_LR.func.gii')
    dataL=nib.load(corrs_vol2surf_path).darrays[0].data
    corrs_vol2surf_path = ospath(f'{corrs_vol2surf_folder}/{corrs_filename}_R.midthickness.32k_fs_LR.func.gii')
    dataR=nib.load(corrs_vol2surf_path).darrays[0].data
    im = np.concatenate([dataL,dataR])[mask]

    im[im==0] = np.nan #Set zeros values (only medial parahippocampal area) to nans
    valid = ~np.isnan(im) #The other nan values are those without a close-enough volume voxel

    ### PLOTTING
    sulc = butils.get_sulc(subject)
    fig,ax=plt.subplots()
    ax.scatter(sulc[valid],im[valid],1,alpha=0.1,color='black')
    ax.set_xlabel('Sulcal depth')
    ax.set_ylabel('Local correlation')
    ax.set_title(f'Correlation is {np.corrcoef(sulc[valid],im[valid])[0,1]:.3f}')

    im[np.isnan(im)] = np.nanmean(im) #for surface plot, replace nans with mean value
    p.plot(im,cmap='inferno')
    plt.show(block=False)







