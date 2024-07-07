"""
Generate smoothing kernels for fs32kLR surface for subject-specific meshes

THIS SCRIPT WAS NOT ACTUALLY USED. IT IS PROBABLY A WORK IN PROGRESS
"""

import hcpalign_utils, os
from hcpalign_utils import ospath
from scipy.io import savemat
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse
from get_gdistances import get_gdistances

def get_smoothers(sub,surface,fwhm,hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200",intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/intermediates',load_from_cache=True,save_to_cache=True):
    """
    Get smoothing kernels based on a surface mesh
    Parameters:
    ----------
    sub: string
        e.g. '102311'
    surface: string
        e.g. midthickness, white, etc
    load_from_cache: boolean
        to load previously saved file from 'intermediates' folder
    save_to_cache: boolean
        to save the calculated geodesic distances to the 'intermediates' folder
    """
    save_path=ospath(f'{intermediates_path}/smoothers/smoother_{surface}_fwhm{fwhm}_sub{sub}')
    if load_from_cache and os.path.exists(f'{save_path}.npz'):
        return sparse.load_npz(f'{save_path}.npz')
    else:  
        print(f"{c.time()} Not in cache: Generating smoothers")
        c=hcpalign_utils.clock()

        gdistances = get_gdistances(sub,surface,fwhm=10,epsilon=0.01,load_from_cache=True,save_to_cache=True)

        surf_file_L=ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.L.{}.32k_fs_LR.surf.gii".format(hcp_folder,sub,sub,surface))
        surf_file_R=ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.R.{}.32k_fs_LR.surf.gii".format(hcp_folder,sub,sub,surface))
       
        if save_to_cache:
            sparse.save_npz(f'{save_path}.npz',smoother)
        return smoother

if __name__=='__main__':
    for surface in ['white','midthickness']:
        for fwhm in [2]:
            for sub in ['100610','102311','102816']: #,'104416','105923','108323','109123','111312','111514','114823']: 
                print([surface,sub,fwhm])  
                get_smoothers(sub,surface,fwhm)
