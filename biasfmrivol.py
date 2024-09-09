"""
Test for gyral bias in neighbourhood correlations in fMRI, in volume space
Use env py390
"""


import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import hcp_utils as hcp
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
import biasfmri_utils as butils
import nilearn.plotting as plotting
import nibabel as nib
import generic_utils as gutils
import cortex_utils as cutils

c = gutils.clock()

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

rests=['REST1_LR','REST1_RL','REST2_LR','REST2_RL']
runs = [0,1]

for subject in hutils.all_subs[slice(0,10)]:
    print(f'{c.time()}: Start subject {subject}')

    ### PARAMETERS
    which_subject_visual = subject
    surface_visual = 'inflated'

    ### SETUP
    import brainmesh_utils as bmutils
    if which_subject_visual =='standard':
        p=cutils.surfplot('',mesh = hcp.mesh[surface_visual], plot_type='open_in_browser')
    else:
        vertices_visual,faces_visual = bmutils.hcp_get_mesh(subject,surface_visual)
        p = cutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')

    corrs_filename = f"corrs_{subject}_runs{''.join([str(i) for i in runs])}"
    corrs_volume_path = ospath(f'{project_path}\\intermediates\\restfmri_volume_corr\\{corrs_filename}.nii.gz')
    corrs_vol2surf_folder = ospath(f'{project_path}/intermediates/restfmri_volume_corr_surf')

    mask = cutils.get_fsLR32k_mask() 
    sulc = butils.hcp_get_sulc(subject)[mask]

    get_nifti = True
    if get_nifti:
        #nifti_filepaths = [ospath(f'{hcp_folder}/{subject}/MNINonLinear/Results/rfMRI_REST{run}_{direction}/rfMRI_REST{run}_{direction}_hp2000_clean.nii.gz') for direction in directions for run in runs]
        nifti_filepaths = [ospath(f'{hcp_folder}/{subject}/MNINonLinear/Results/rfMRI_{rests[run]}/rfMRI_{rests[run]}_hp2000_clean.nii.gz') for run in runs]

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
        cutoff = 100 #default 100
        mask_image_resampled_bool = image.math_img(f"img>{cutoff}",img=mask_image_resampled)


        #Parameters for functional clustering
        method='ward'
        n_parcels=100
        smoothing_fwhm=8
        to_mask='F' #'T' to mask as usual, or 'F' (default) to calculate a mask within nilearn
        clust_folder = ospath(f'{project_path}/intermediates/clustering_volume')
        clust_vol2surf_folder = ospath(f'{project_path}/intermediates/clustering_volume_surf')
        clust_filename = f"clust_{method[0]}{n_parcels}{to_mask}{smoothing_fwhm}mm_r{''.join([str(i) for i in runs])}_{subject}"
        clust_path = ospath(f'{clust_folder}\\{clust_filename}.nii.gz')

        do_clustering = True
        if do_clustering:
            print(f'{c.time()}: Clustering start')
            from nilearn.regions import Parcellations
            from nilearn import plotting
            if to_mask=='T':
                mni_mask=mask_image_resampled_bool
                mask_strategy=None
            else:
                mni_mask=None
                mask_strategy='whole-brain-template' #defualt whole-brain-template
            clustering = Parcellations(method=method,n_parcels=n_parcels,mask=mni_mask,mask_strategy=mask_strategy,smoothing_fwhm=smoothing_fwhm,n_jobs=-1,standardize='zscore_sample') 
            clustering.fit([nifti_image])
            print(f'{c.time()}: Clustering done')
            labels = clustering.labels_img_
            #plotting.plot_roi(labels,cmap='prism')
            #plt.show(block=False)
            data = labels.get_fdata()
            #print([np.sum(data==i) for i in np.unique(data)]) #show number of voxels in each parcel
            nib.save(labels,clust_path) #save as .nii.gz

        use_clustering=False
        if use_clustering:
            im = butils.mni_to_fslr(clust_path,mask,method='nearest')
            p.plot(im,cmap='prism')


        make_corrs_volume = False
        if make_corrs_volume:
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

    use_corrs_vol2surf = False
    if use_corrs_vol2surf:
        parc_string = 'S300'
        parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
        parc_matrix = hutils.parcellation_string_to_parcmatrix(parc_string)

        ### GET THE DATA
        im,valid = butils.data_gii2array(mask,corrs_vol2surf_folder,corrs_filename)
        """
        #ngraysLR = [len(hcp.vertex_info.grayl),len(hcp.vertex_info.grayr)]
        corrs_vol2surf_path = ospath(f'{corrs_vol2surf_folder}/{corrs_filename}_L.midthickness.32k_fs_LR.func.gii')
        dataL=nib.load(corrs_vol2surf_path).darrays[0].data
        corrs_vol2surf_path = ospath(f'{corrs_vol2surf_folder}/{corrs_filename}_R.midthickness.32k_fs_LR.func.gii')
        dataR=nib.load(corrs_vol2surf_path).darrays[0].data
        im = np.concatenate([dataL,dataR])[mask]
        im[im==0] = np.nan #Set zeros values (only medial parahippocampal area) to nans
        valid = ~np.isnan(im) #The other nan values are those without a close-enough volume voxel
        """

        to_normalize = True
        if to_normalize:
            smoother = hutils.get_searchlight_smoother(sub=subject,surface='midthickness',radius_mm=15) #102311
            normalize = lambda x: x - smoother(x)
            im[np.isnan(im)] = np.nanmean(im)
            im=normalize(im)
            sulc = normalize(sulc)

        ### PLOTTING
        from scipy.stats import pearsonr

        plt.rcParams.update({'font.size': 15})
        fig,ax=plt.subplots(figsize=(4,4))
        ax.scatter(sulc[valid],im[valid],1,alpha=0.1,color='k')
        ax.set_xlabel('Sulcal depth')
        ax.set_ylabel('Local correlation')
        corr,pval=butils.corr_with_nulls(sulc,im,mask,method='spin_test',n_perm=1000)
        ax.set_title(f'Correlation is {corr:.3f}, p-value is {pval:.3f}')
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        fig.tight_layout()

        im[np.isnan(im)] = np.nanmean(im) #for surface plot, replace nans with mean value
        p.plot(im,cmap='bwr')
        plt.show(block=False)

        """
        ngrayl = len(hcp.vertex_info.grayl) #left hemisphere only
        im_left = im[0:ngrayl]
        verts,faces = bmutils.hcp_get_mesh_hemi(subject,'L',surface_visual)
        mask_left = hutils.get_fsLR32k_mask(hemi='L')
        verts_left = verts[mask_left,:]
        faces_left = butils.triangles_removenongray(faces,mask_left)
        p_left = hutils.surfplot('',mesh = [verts_left,faces_left], plot_type='open_in_browser')
        p_left.plot(im_left,cmap='inferno')
        """