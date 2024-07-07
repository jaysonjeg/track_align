#Make geodesic distances and save it
import os
import numpy as np
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from scipy.io import savemat
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse

import generic_utils as gutils
c=gutils.clock()

"""
if mesh_template=='fsLR32k':
    if sub=='standard':
        import hcp_utils as hcp
        surf_file_L = ospath(f"{os.path.dirname(hcp.__file__)}/data/S1200.L.{surface}{MSMstring}.32k_fs_LR.surf.gii")
        surf_file_R = ospath(f"{os.path.dirname(hcp.__file__)}/data/S1200.R.{surface}{MSMstring}.32k_fs_LR.surf.gii")
    else:
        surf_file_L=ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.L.{}{}.32k_fs_LR.surf.gii".format(hcp_folder,sub,sub,surface,MSMstring))
        surf_file_R=ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.R.{}{}.32k_fs_LR.surf.gii".format(hcp_folder,sub,sub,surface,MSMstring))
elif mesh_template=='fsaverage5':
    assert(sub=='standard')
    from nilearn import datasets
    surf_file_L = ospath(f"{os.path.dirname(datasets.__file__)}/data/fsaverage5/{surface}_left.gii.gz")
    surf_file_R = ospath(f"{os.path.dirname(datasets.__file__)}/data/fsaverage5/{surface}_right.gii.gz")
"""

"""
    if mesh_template=='fsLR32k':
        mesh_string = ''
        if sub=='standard':
            if surface=='sphere':
                assert(MSMAll==False)
            else: 
                assert(MSMAll==True)
    if mesh_template== 'fsaverage5':
        mesh_string = 'fsaverage5'
        assert(MSMAll==False)
    if MSMAll:
        MSMstring = '_MSMAll'
    else:
        MSMstring = ''

"""

def get_mesh_string(mesh_template, sub, surface, MSMAll):
    if mesh_template == 'fsLR32k':
        mesh_string = ''
        if sub == 'standard':
            if surface == 'sphere':
                assert(MSMAll == False)
            else:
                assert(MSMAll == True)
    elif mesh_template == 'fsaverage5':
        mesh_string = 'fsaverage5'
        assert(MSMAll == False)
    else:
        mesh_string = ''
    if MSMAll:
        MSMstring = '_MSMAll'
    else:
        MSMstring = ''
    return mesh_string, MSMstring

def get_surf_files(mesh_template, sub, surface, MSMstring, hcp_folder):
    if mesh_template == 'fsLR32k':
        if sub == 'standard':
            import hcp_utils as hcp
            surf_file_L = ospath(f"{os.path.dirname(hcp.__file__)}/data/S1200.L.{surface}{MSMstring}.32k_fs_LR.surf.gii")
            surf_file_R = ospath(f"{os.path.dirname(hcp.__file__)}/data/S1200.R.{surface}{MSMstring}.32k_fs_LR.surf.gii")
        else:
            surf_file_L = ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.L.{}{}.32k_fs_LR.surf.gii".format(hcp_folder, sub, sub, surface, MSMstring))
            surf_file_R = ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.R.{}{}.32k_fs_LR.surf.gii".format(hcp_folder, sub, sub, surface, MSMstring))
    elif mesh_template == 'fsaverage5':
        assert(sub == 'standard')
        from nilearn import datasets
        surf_file_L = ospath(f"{os.path.dirname(datasets.__file__)}/data/fsaverage5/{surface}_left.gii.gz")
        surf_file_R = ospath(f"{os.path.dirname(datasets.__file__)}/data/fsaverage5/{surface}_right.gii.gz")
    return surf_file_L, surf_file_R

def get_gdistances(sub,surface,fwhm=None,epsilon=0.01,max_dist=None,hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200",intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/intermediates',load_from_cache=True,save_to_cache=True,MSMAll=False,mesh_template='fsLR32k'):
    """
    Get geodesic distances based on a surface mesh. Can either specify max_dist, or specify fwhm and epsilon
    Parameters:
    ----------
    sub: string
        e.g. '102311'
    surface: string
        e.g. midthickness, white, etc
    fwhm: float
    epsilon: float
    max_dist: float
    hcp_folder: string
    intermediates_path: string
    load_from_cache: boolean
        to load previously saved file from 'intermediates' folder
    save_to_cache: boolean
        to save the calculated geodesic distances to the 'intermediates' folder
    MSMAll: boolean
    mesh_template: string
        'fsLR32k','fsaverage5'
    """
    mesh_string, MSMstring = get_mesh_string(mesh_template, sub, surface, MSMAll)

    if max_dist is None:
        save_path=ospath(f'{intermediates_path}/geodesic_distances/gdist_{mesh_string}{surface}_fwhm{fwhm}_eps{epsilon}_sub{sub}{MSMstring}')
        sigma=css._fwhm2sigma(fwhm)
        max_dist=css._max_smoothing_distance(sigma, epsilon)
    else:
        save_path=ospath(f'{intermediates_path}/geodesic_distances/gdist_{mesh_string}{surface}_max{max_dist}mm_sub{sub}{MSMstring}')
    if load_from_cache and os.path.exists(f'{save_path}.npz'):
        return sparse.load_npz(f'{save_path}.npz')
    else:  
        print(f"{c.time()}: Not in cache: Generating geodesic distances {sub} {surface} fwhm{fwhm}mm max_dist{max_dist}mm")
        surf_file_L, surf_file_R = get_surf_files(mesh_template, sub, surface, MSMstring, hcp_folder)
        cortical_local_geodesic_distances = css._get_cortical_local_distances(surf_file_L, surf_file_R, max_dist) 
        if save_to_cache:
            sparse.save_npz(f'{save_path}.npz',cortical_local_geodesic_distances)
        return cortical_local_geodesic_distances

def get_searchlights(sub='102311',surface='midthickness',radius_mm=15):
    """
    Return searchlights. If cached file doesn't already exist, then make it. Save this list of searchlights as a pickle object
    Replaces make_gdistances_full.make_and_search_searchlights and hcpalign_utils.searchlights
    Parameters:
    ----------
    sub: string
        e.g. '102311'
    surface: string
        e.g. midthickness, white, etc
    radius_mm: float
        only look at vertices within this many mm of the source vertex
    Returns:
    ----------
    parcels: list
        list of searchlights, one centred at each vertex. Each searchlight is a list of vertices within radius_mm of the centre vertex
    """
    import pickle
    save_path=hutils.ospath(f'{hutils.intermediates_path}/searchlightparcellation/xparc_{sub}_{surface}_{radius_mm}mm.p')
    if os.path.exists(save_path):
        return pickle.load(open(ospath(save_path), "rb" ))
    else:
        print(f"Not in cache: Generating searchlights {sub} {surface} radius {radius_mm}mm")
        gdists = get_gdistances(sub,surface,fwhm=None,max_dist=radius_mm,mesh_template='fsLR32k',load_from_cache=True,save_to_cache=True)
        gdists_bool = gdists.astype(bool).toarray()
        parcels = [np.where(gdists_bool[i,:])[0].astype('int32') for i in range(gdists.shape[0])]
        del gdists_bool
        pickle.dump(parcels,open(save_path,"wb"))
        return parcels

if __name__=='__main__':
    for surface in ['midthickness']:
        for max_dist in [15]:
            for sub in ['100610','102311']: #hutils.all_subs[0:10]: #['100610','102311','102816','104416','105923','108323','109123','111312','111514','114823']: #['standard']
                print([surface,sub,max_dist])  
                #x=get_gdistances(sub,surface,fwhm=3,mesh_template='fsLR32k',load_from_cache=False,save_to_cache=True,epsilon=0.01)
                x=get_gdistances(sub,surface,fwhm=None,max_dist=max_dist,mesh_template='fsLR32k',load_from_cache=False,save_to_cache=True)
                print(f'{c.time()}: Done')

