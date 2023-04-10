#Make geodesic distances and save it
import hcpalign_utils, os
from hcpalign_utils import ospath
from scipy.io import savemat
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse

def get_gdistances(sub,surface,fwhm,epsilon=0.01,hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200",intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/intermediates',load_from_cache=True,save_to_cache=True):
    """
    Get geodesic distances based on a surface mesh
    sub e.g. '102311'
    surface can be midthickness, white, etc
    load_from_cache: to load previously saved file from 'intermediates' folder
    save_to_cache: to save the calculated geodesic distances to the 'intermediates' folder
    """
    save_path=ospath(f'{intermediates_path}/geodesic_distances/gdist_{surface}_fwhm{fwhm}_eps{epsilon}_sub{sub}')
    if load_from_cache and os.path.exists(f'{save_path}.npz'):
        return sparse.load_npz(f'{save_path}.npz')
    else:  
        c=hcpalign_utils.clock()
        surf_file_L=ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.L.{}.32k_fs_LR.surf.gii".format(hcp_folder,sub,sub,surface))
        surf_file_R=ospath("{}/{}/MNINonLinear/fsaverage_LR32k/{}.R.{}.32k_fs_LR.surf.gii".format(hcp_folder,sub,sub,surface))
       
        #print(c.time())
        sigma=css._fwhm2sigma(fwhm)
        cortical_local_geodesic_distances = css._get_cortical_local_distances(surf_file_L, surf_file_R, css._max_smoothing_distance(sigma, epsilon)) 
        #print(c.time())
        if save_to_cache:
            sparse.save_npz(f'{save_path}.npz',cortical_local_geodesic_distances)
            #savemat(f'{save_path}.mat',{'data':smoother})
        return cortical_local_geodesic_distances

if __name__=='__main__':
    for surface in ['white','inflated']:
        for fwhm in [10]:
            for sub in ['100610','102311','102816','104416','105923','108323','109123','111312','111514','114823']: 
                print([surface,sub,fwhm])  
                get_gdistances(sub,surface,fwhm)

