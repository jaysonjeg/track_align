"""
Get surface mesh from HCP data. Find second fundamental form (of curvature) and new coordinate system at each vertex, and save these
"""

import nibabel as nib
nib.imageglobals.logger.level = 40  #suppress pixdim error msg
import numpy as np
import os
import hcp_utils as hcp
import hcpalign_utils
from hcpalign_utils import ospath
import pickle
from scipy.io import savemat
from scipy import sparse
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\Jayson\\Google Drive\\PhD\\Project_Hyperalignment\\Code\\curvpack\\curvpack")

"""
import trimesh
from curvpack.curvpack import CurvatureCwF,CurvatureWpF,CurvatureCubic,CurvatureISF
"""

c=hcpalign_utils.clock()

def get_surface(surface_gifti_file):
    surface=nib.load(surface_gifti_file)
    vertices=surface.darrays[0].data #array (nvertices,3)
    triangles=surface.darrays[1].data #array (ntriangles,3)
    return vertices,triangles

def get_verts_and_triangles(sub,surface_type):
    hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200"

    L_surf_file=ospath(f"{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.L.{surface_type}.32k_fs_LR.surf.gii")
    R_surf_file=ospath(f"{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.R.{surface_type}.32k_fs_LR.surf.gii")

    Lvertices,Ltriangles=get_surface(L_surf_file) #L triangles are numbered 0 to 32491
    Rvertices,Rtriangles=get_surface(R_surf_file) #R triangles are originally numbered 0 to 32491
    Rtriangles_new=Rtriangles+Lvertices.shape[0] #R triangles now numbered 32492 to 64983

    all_vertices_64k=np.vstack((Lvertices,Rvertices)) #array (64984,3)
    all_triangles_64k=np.vstack((Ltriangles,Rtriangles_new))

    return all_vertices_64k,all_triangles_64k

def get_verts_and_triangles_59k(sub,surface_type):
    all_vertices_64k,all_triangles_64k=get_verts_and_triangles(sub,surface_type)
    all_vertices_59k=hcpalign_utils.cortex_64kto59k(all_vertices_64k) #array (59412,3)
    all_triangles_59k=hcpalign_utils.cortex_64kto59k_for_triangles(all_triangles_64k)

    vertices=all_vertices_59k
    faces=all_triangles_59k
    
    return vertices,faces

def filter_allowed_columns(csc_array,columns_to_include):
    """
    columns_to_include is an array of logicals (True,False) which says which columns of csc_array to include
    Non-included columns of csc_array will end up with 1 in row that's on the diagonal, and 0s elsewhere
    """
    arrayT = sparse.lil_matrix(csc_array.T) #transpose for speed
    for row in range(arrayT.shape[0]): 
        if not(columns_to_include[row]):
            arrayT.rows[row] = [row]
            arrayT.data[row] = [1]
    return sparse.csc_matrix(arrayT.T) #untranspose it back

def apply_gyral_mask(smoother,intermediates_path,surface_type,gyrus_threshold_curvature,sub):
    gyral_mask_save_file = ospath(f'{intermediates_path}/gyral_mask/gyral_mask_{surface_type}_MaxCurvOver{gyrus_threshold_curvature}_sub{sub}.npy')
    gyral_mask = np.load(gyral_mask_save_file)           
    return filter_allowed_columns(smoother,gyral_mask)    



def gyralsmoothing(hrs, sub, surface_type='white',smoother_type = 'a', fwhm_y=3,fwhm_x=None, fwhm=10, use_gyral_mask=False,gyrus_threshold_curvature=0.3, interp_from_gyri=False):
    """
    Parameters relevant to smoother_type 'd' only
        fwhm: used to determine radius of circular distance
        interp_from_gyri: use xy distance matrices where non-gyral values are interpolated from gyral values
    
    """
    from Connectome_Spatial_Smoothing import CSS as css
    intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/intermediates'
    if smoother_type in ['a','b','c']:
        smoother=sparse.load_npz(ospath(f'{intermediates_path}/gyral_smoother/GS_{surface_type}_fwhm{fwhm_y}_sub{sub}_dirmin_nrows59412.npz'))
    elif smoother_type in ['d']:
        smoo_string = f'SMOO{surface_type}_fwhm{fwhm}'
        if interp_from_gyri: mask_string=f'MASKT{surface_type}_{gyrus_threshold_curvature}'
        else: mask_string = 'MASKF'
        xy_smoo_string = f'{smoo_string}_{mask_string}_{sub}'
        xydist_save_prefix = ospath(f'{intermediates_path}/xy_smoothing/xy_dists/dist_{xy_smoo_string}')
        xydist_string_X = f'{xydist_save_prefix}_X.npz'
        xydist_string_Y = f'{xydist_save_prefix}_Y.npz'
        xdists = sparse.load_npz(xydist_string_X)
        ydists = sparse.load_npz(xydist_string_Y)
        assert(fwhm>=fwhm_y and fwhm>=fwhm_x)
        smoother=hcpalign_utils._xy_distances_to_2Dgaussian(xdists,ydists,fwhm_x,fwhm_y)

    if use_gyral_mask:
        smoother = apply_gyral_mask(smoother,intermediates_path,surface_type,gyrus_threshold_curvature,sub) 
    return css.smooth_high_resolution_connectome(hrs,smoother)

def coordinate_transform(coords_in,matrix):
    """
    coords_in is 1 x 2
    marix is 2 x 3
    """
    return matrix.T @ coords_in
def coordinate_transform_each_point(coords_array_in,up,vp):
    """
    coords_array_in is array(n,2) containing n points in 2-dimensional coordinate system
    up and vp (each is array(n,3)) contain the 3D direction vectors for each of the original 2-dim direction vectors
    """
    nvertices=coords_array_in.shape[0]
    coords_array_out=np.zeros((nvertices,3),dtype=float)
    for nvertex in range(nvertices):
        coords_in = coords_array_in[nvertex,:]
        axes=np.array(np.vstack([up[nvertex,:],vp[nvertex,:]]))
        coords_array_out[nvertex,:] = coordinate_transform(coords_in,axes)
    return coords_array_out

def get_vertices_along_a_plane(mesh,plane_normal=[0,0,1],plane_origin=[0,0,0]):
    """
    First determine the intersection of a plane with a surface mesh
    Then return all the vertices that belong to faces which are on this intersection
    For plane_normal and plane_origin, the axes are (right, anterior, superior)
    """
    lines,face_index=trimesh.intersections.mesh_plane(mesh,plane_normal,plane_origin,return_faces=True)
    faces_intersecting_plane=mesh.faces[face_index,:]
    return faces_intersecting_plane.ravel()  

### FUNCTIONS FOR VISUALISING GYRAL SMOOTHERS ###

def pad_right(array,target_ncols=59412):
    """
    Pad a sparse csc 'array' with empty columns on the right, until number of columns is target_ncols
    """
    new_indptr=np.pad(array.indptr,(0,target_ncols-array.shape[1]),"edge")
    return sparse.csc_matrix((array.data,array.indices,new_indptr),shape=(array.shape[0],target_ncols))

def array_copy(array):
    if type(array)==sparse.csc_matrix:
        return sparse.csc_matrix.copy(array)
    elif type(array)==sparse.csr_matrix:
        return sparse.csr_matrix.copy(array)
    elif type(array)==np.ndarray:
        return np.copy(array)

def allones(array):
    """
    Replace all data points with value 1. Useful for visualising full extent of a gaussian kernel
    """
    array2=array_copy(array)
    array2.data = np.ones(array2.data.shape,array2.data.dtype)
    return array2

if __name__=='__main__':
    for sub in ['102311']:
        for surface_type in ['midthickness']:
            
            save_path=ospath(f'/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/intermediates/SFM/SFM_{surface_type}_sub{sub}')
            
            vertices,faces=get_verts_and_triangles_59k(sub,surface_type)

            VertexSFM,VertNormals,up,vp=CurvatureWpF.CalcCurvature(vertices,faces) 
            with open(f'{save_path}.pickle', 'wb') as f:
                pickle.dump([VertexSFM,VertNormals,up,vp], f)
            print(f'{sub} {surface_type} done at {c.time()}')