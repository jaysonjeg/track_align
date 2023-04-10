#Make geodesic distances and save it. Full matrix for all coordinates
import hcpalign_utils, os
from hcpalign_utils import ospath
from scipy.io import savemat
import nibabel as nib
nib.imageglobals.logger.level = 40  #suppress pixdim error msg
import numpy as np
import gdist
import hcp_utils as hcp
from scipy import sparse

c=hcpalign_utils.clock()
hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200"


def get_saved_gdistances_full_hemi(sub,surface,hemi,hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200",intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/intermediates'):
    """
    Get full geodesic distances based on a surface mesh
    sub e.g. '102311'
    surface can be midthickness, white, etc
    hemi is 'L' or 'R'
    """
    save_path=ospath(f'{intermediates_path}/geodesic_distances/gdist_full_{sub}.{hemi}.{surface}.32k_fs_LR')
    if os.path.exists(f'{save_path}.npy'):
        return np.load(f'{save_path}.npy')
    else:
        print(f'{save_path}.npy not found')

def get_saved_gdistances_full(sub,surface,hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200",intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/intermediates'):
    L=get_saved_gdistances_full_hemi(sub,surface,'L')
    R=get_saved_gdistances_full_hemi(sub,surface,'R')
    w=np.zeros((59412,59412),dtype=L.dtype)
    w[:]=np.inf #so that interhemispheric distances are infinite
    cutoff = len(hcp.vertex_info.grayl)
    w[0:cutoff,0:cutoff]=L
    w[cutoff:,cutoff:]=R
    return w


def process_i(i,nvertices_target,vertices,triangles):
    source_indices=np.array([i])
    target_indices=np.array(range(i+1 , nvertices_target)).astype('int32') #everything above source_indices. ie computing lower half of the symmetric distance matrix                   
    distances=gdist.compute_gdist(vertices,triangles,source_indices,target_indices=target_indices)
    return distances.astype(np.float16)

if __name__=='__main__':
    method = 2 #methods 1 and 2
    use_multiproc=True
    p=hcpalign_utils.surfplot('',plot_type='open_in_browser')
    for surface_type in ['midthickness','inflated']:
        for hemi in ['L','R']:
            for sub in ["100610","102311"]:
                
                print(f'{sub}, {surface_type}, {hemi}')
                save_path=ospath(f'/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/intermediates/geodesic_distances/gdist_full_{sub}.{hemi}.{surface_type}.32k_fs_LR')
                surf_file=ospath(f"{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.{hemi}.{surface_type}.32k_fs_LR.surf.gii")

                surface=nib.load(surf_file)
                vertices=surface.darrays[0].data.astype(float) #array (nvertices,3)
                triangles=surface.darrays[1].data #array (ntriangles,3)

                #remove non-cortical medial wall
                if hemi=='L':
                    hemi_grays=hcp.vertex_info.grayl
                elif hemi=='R':
                    hemi_grays=hcp.vertex_info.grayr
                vertices=vertices[hemi_grays]
                triangles=hcpalign_utils.cortex_64kto59k_for_triangles(triangles,hemi=hemi)
                
                if method==1:
                    print(f'start at {c.time()}')
                    r_sparse=gdist.local_gdist_matrix(vertices,triangles)
                    print(f'done at {c.time()}')
                    r_sparse = r_sparse.astype('float32')
                    r=r_sparse.toarray()
                    savemat(f'{save_path}.mat',{'data':r_sparse.toarray()})
                    savemat(f'{save_path}_sparse.mat',{'data':r_sparse})
                
                if method==2:
                    
                    nvertices=vertices.shape[0]
                    r=np.zeros((nvertices,nvertices),dtype=np.float16)
                    print(f'start at {c.time()}')
                    if not(use_multiproc):
                        """
                        8h per hemisphere
                        """
                        for i in range(nvertices):
                            print(i)
                            if i!=0 and i%100==0: 
                                print(f"{i}/{nvertices} at {c.time()}")
                                hcpalign_utils.memused()
                            r[i+1 :,i]=process_i(i,nvertices,vertices,triangles)
                    else:
                        """
                        1h per hemisphere
                        10: 7sec
                        100: 11sec (20sec with 3 procs)
                        1000: 62sec 
                        3000: 176s
                        10000: 593s
                        20000: 1314s
                        """
                        import multiprocessing as mp
                        with mp.Pool(10) as p:
                            nvertices,vertices,triangles
                            temp = p.starmap(process_i,[[i,nvertices,vertices,triangles] for i in range(nvertices)])
                        print(f'{c.time()}: starting to fill in array')
                        for i in range(nvertices):
                            r[i+1 :,i] = temp[i]                                    
                    np.save(f"{save_path}.npy",r+r.T)
                    #savemat(f'{save_path}.mat',{'data':r}) 
                

