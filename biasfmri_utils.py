import numpy as np
import hcpalign_utils as hutils
from hcpalign_utils import ospath

hcp_folder=hutils.hcp_folder
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"



def text_file_to_npy(filepath):
    """
    Open a text file. Save the numbers as a numpy array in a .npy file
    """
    data = np.loadtxt(filepath+'.txt',dtype=np.float32)
    np.save(filepath,data)


'''
from hcpalign_utils import ospath
[text_file_to_npy(ospath(f'{project_path}/fsaverage10k_timeseries_0{i}')) for i in [0,1,2]]
[text_file_to_npy(ospath(f'{project_path}/fsLR32k_timeseries_0{i}')) for i in [0,1,2]]
[text_file_to_npy(ospath(f'{project_path}/fsaverage_10k_medial_wall_{hemi}_masked')) for hemi in ['lh','rh']]
[text_file_to_npy(ospath(f'{project_path}/{i}.32k_fs_LR.func')) for i in ['100610','102311']]

'''

### Functions for fsvaverage5 mesh

def get_fsaverage5_mask():
    """
    Returns a boolean array indicating, for each vertex in fsaverage5 surface, whether it is gray matter (1) or medial wall (0)
    """
    from hcpalign_utils import ospath
    lh_masked=np.load(ospath(f'{project_path}/fsaverage_10k_medial_wall_lh_masked.npy'))
    rh_masked=np.load(ospath(f'{project_path}/fsaverage_10k_medial_wall_rh_masked.npy'))
    fsaverage5_mask = np.hstack((lh_masked,rh_masked)).astype(bool) #list of booleansindicating whether vertex is gray matter (1) or medial wall (0)
    #fsaverage5_gray = np.where(fsaverage5_mask==1)[0] #list of indices of gray matter vertices
    return fsaverage5_mask

def get_fsaverage_scalar_hemi(string,hemisphere,meshname='fsaverage5'):
    from nilearn import datasets
    import nibabel as nib
    fsaverage = datasets.fetch_surf_fsaverage(mesh=meshname)
    filename = fsaverage[f'{string}_{hemisphere}'] 
    gifti = nib.load(filename)
    return gifti.darrays[0].data

def get_fsaverage_scalar(string,meshname='fsaverage5'):
    """
    Return scalar data for a mesh
    Parameters:
    string : str
        e.g. 'curv', area, pial, white, sulc, thick
    mesh: str
        e.g. 'fsaverage5'
    """
    Lscalar = get_fsaverage_scalar_hemi(string,'left')
    Rscalar = get_fsaverage_scalar_hemi(string,'right')
    scalar = np.hstack((Lscalar,Rscalar))
    return scalar

def get_fsaverage_mesh_hemi(string,hemisphere,meshname='fsaverage5'):
    from nilearn import datasets
    import nibabel as nib
    fsaverage = datasets.fetch_surf_fsaverage(mesh=meshname)
    filename = fsaverage[f'{string}_{hemisphere}'] 
    mesh = nib.load(filename)
    vertices, faces = mesh.darrays[0].data, mesh.darrays[1].data
    return vertices,faces
def get_fsaverage_mesh(string,meshname='fsaverage5'):
    """
    Return vertices and faces for a mesh
    Parameters:
    string : str
        e.g. 'curv', area, pial, white, sulc, thick
    mesh: str
        e.g. 'fsaverage5'
    """
    Lvert,Lfaces = get_fsaverage_mesh_hemi(string,'left')
    Rvert,Rfaces = get_fsaverage_mesh_hemi(string,'right')
    vertices = np.vstack((Lvert,Rvert))
    faces = np.vstack((Lfaces,Rfaces+Lvert.shape[0]))
    return vertices,faces

#Functions for dealing with surface meshes

def fillnongray(arr,mask,fillvalue=0):
    """
    Takes a 1D array of fMRI grayordinates and returns the values on the vertices of the left cortex mesh which is neccessary for surface visualization. 
    The unused vertices are filled with a constant (zero by default).
    E.g. stepping up from 18k to 20k vertices 
    Like hcp_utils.cortex_data()
    """
    out = np.zeros(len(mask))
    out[:]=fillvalue
    out[mask] = arr
    return out

def removenongray(mask):
    """
    List of 20k cortex mesh vertices, with their mapping onto 18k vertices in fsaverage5. Vertices not present in 59k version are given value 0
    """   
    fsaverage5_gray = np.where(mask==1)[0] #list (18k gray matter vertices) of indices (in full mesh)
    temp = np.zeros(len(mask),dtype=int)
    for index,value in enumerate(fsaverage5_gray):
        temp[value]=index
    return temp

def triangles_removenongray(triangles,mask):
    """
    Given a list of triangles in a 64984-vertex mesh, return a list of triangles in a 64984-vertex mesh with only gray matter vertices. First remove the rows containing non-gray vertices, then renumber the vertices so they are 0 to nGrayVertices
    triangles: array (n,3) list of triangles in a 64984-vertex mesh
    mask: boolean array (64984,) indicating which vertices are gray matter
    """
    gray = np.where(mask)[0]
    temp=np.isin(triangles,gray)
    temp2=np.all(temp,axis=1)
    triangles_v2 = triangles[temp2,:]
    triangles_v3 = removenongray(mask)[triangles_v2] #renumbering
    return triangles_v3

def get_sulc(subject_id,version='fsaverage_LR32k'):
    """
    version: 'native', 'fsaverage_LR32k' (default) or '164k'
    """
    if version=='fsaverage_LR32k':
        sulc_path = ospath(f'{hcp_folder}/{subject_id}/MNINonLinear/fsaverage_LR32k/{subject_id}.sulc.32k_fs_LR.dscalar.nii')
    elif version=='native':
        sulc_path = ospath(f'{hcp_folder}/{subject_id}/MNINonLinear/Native/{subject_id}.sulc.native.dscalar.nii')
    elif version=='164k':
        sulc_path = ospath(f'{hcp_folder}/{subject_id}/MNINonLinear/{subject_id}.sulc.164k_fs_LR.dscalar.nii')
    else: 
        assert(0)
    return hutils.get(sulc_path).squeeze()

def filepath_to_parcellation(filepath):
    import nibabel as nib
    x=nib.load(filepath)
    labels=x.darrays[0].data
    return labels
def get_Yeo17Networks_fsLR32k():
    file_left = ospath(f"{project_path}/intermediates/parcellations/Yeo_JNeurophysiol11_17Networks.32k.L.label.gii")
    labels_left = filepath_to_parcellation(file_left)
    file_right = ospath(f"{project_path}/intermediates/parcellations/Yeo_JNeurophysiol11_17Networks.32k.R.label.gii")
    labels_right = filepath_to_parcellation(file_right)
    return np.hstack([labels_left,labels_right])
def get_brodmann_fsLR32k():
    prefix = 'BAprobatlas.32k.R.func'
    prefix = 'BA_handArea.32k.L.label'
    file_left = ospath(f"{project_path}/intermediates/parcellations/{prefix}.gii")
    labels_left = filepath_to_parcellation(file_left)
    file_right = ospath(f"{project_path}/intermediates/parcellations/{prefix}.gii")
    labels_right = filepath_to_parcellation(file_right)
    return np.hstack([labels_left,labels_right])

#Functions for neighbour vertices and correlations

def _get_all_neighbour_vertices(mesh,mask):
    """
    Given a mesh, return for each vertex, a list of neighbouring vertices for each vertex (variable neighbour_vertices), and mean distance to its neighbours (variable mean_neighbour_distance). 
    Parameters:
    ----------
    mesh: tuple (vertices,triangles)
        vertices: array (nvertices,3)
        triangles: array (ntriangles,3)
    mask: boolean array (nvertices,) or None
    Returns:
    --------
    neighbour_vertices: list of lists
        For each vertex, a list of neighbouring vertices
    mean_neighbour_distance: array (nvertices,)
        Mean distance to neighbours for each vertex    
    """
    from itertools import combinations
    vertices=mesh[0]
    faces=mesh[1]
    neighbour_vertices=[[] for i in range(len(vertices))]
    #neighbour_distances_sum=np.zeros((len(vertices),2),dtype=np.float32)
    neighbour_distances = [[] for i in range(len(vertices))]
    i=0
    for face in faces:
        i+=1
        if i%30000==0:
            print(f'Face {i} of {len(faces)}')
        #Each triangle is a 3-tuple of vertices. For each vertex in the triangle, add the other two vertices to its list of neighbours. Add the distance to the other two vertices to the distance array. This procedure counts each edge twice, because each edge is in two triangles
        for vertex1,vertex2 in combinations(face,2):
            distance = np.linalg.norm(vertices[vertex1]-vertices[vertex2])
            if vertex2 not in neighbour_vertices[vertex1]:
                neighbour_vertices[vertex1].append(vertex2)
                neighbour_distances[vertex1].append(distance)
            if vertex1 not in neighbour_vertices[vertex2]:
                neighbour_vertices[vertex2].append(vertex1)
                neighbour_distances[vertex2].append(distance)

    if mask is not None:
        map_mesh_to_graymesh = removenongray(mask)

        neighbour_vertices = np.array(neighbour_vertices,dtype=object)
        neighbour_vertices = [map_mesh_to_graymesh[i] for i in neighbour_vertices] 
        neighbour_vertices = np.array(neighbour_vertices,dtype=object)
        neighbour_vertices  = neighbour_vertices[mask]

        neighbour_distances = np.array(neighbour_distances,dtype=object)
        neighbour_distances = neighbour_distances[mask]
    return neighbour_vertices,neighbour_distances 

'''
def get_neighbour_vertices(c, mesh, which_neighbours, distance_range=None,MSMAll=False):
    if which_neighbours == 'distant':
        from get_gdistances import get_gdistances
        print(f'{c.time()}: Get geodesic distances start')
        d = get_gdistances(which_subject,surface,10,MSMAll=MSMAll) #d is a sparse matrix in compressed sparse row format
        print(f'{c.time()}: Get vertices at distance range start')
        neighbour_vertices, neighbour_distances = get_vertices_in_distance_range(distance_range,d)
        nVerticesWithZeroNeighbours = np.sum(np.array([len(i) for i in distant_vertices])==0)
        print(f'{nVerticesWithZeroNeighbours} vertices with 0 neighbours in distance range {distance_range}')
        assert(nVerticesWithZeroNeighbours==0)
    elif which_neighbours == 'local':
        print(f'{c.time()}: Get neighbour vertices start')
        neighbour_vertices,neighbour_distances = _get_all_neighbour_vertices(mesh,None)       
        #nearest_neighbour_distances = np.array([np.min(i) for i in all_neighbour_distances])
        #nearest_neighbour_indices = np.array([np.argmin(i) for i in all_neighbour_distances])
        #nearest_neighbour_vertices = [i[j] for i,j in zip(all_neighbour_vertices,nearest_neighbour_indices)]
    return neighbour_vertices, neighbour_distances
'''

import pickle
def get_subjects_neighbour_vertices(c, subject,surface,mesh, biasfmri_intermediates_path, which_neighbours, distance_range, load_neighbours, save_neighbours,MSMAll=False):
    """
    For each vertex belonging to this subject, find neighbouring vertices and distances, and optionally save to file
    Save neighbour vertex indices in verts.pkl (list (len 59412) of lists)
    Save neighbour vertex distances in dists.pkl (list (len 59412) of lists)
    Save the mean distance to neighbours in dists.npy (np.array of shape 59412)
    """
    if MSMAll:
        MSMstring='_MSMAll'
    else:
        MSMstring=''
    if which_neighbours=='distant':
        dist_string=f'_{distance_range[0]}to{distance_range[1]}'
    else:
        dist_string = ''
    neighbour_verts_path = ospath(f'{biasfmri_intermediates_path}/neighbour_vertices/adj_{subject}_{surface}{MSMstring}_{which_neighbours}{dist_string}_verts.pkl')
    neighbour_dists_path = ospath(f'{biasfmri_intermediates_path}/neighbour_vertices/adj_{subject}_{surface}{MSMstring}_{which_neighbours}{dist_string}_dists.pkl')
    #neighbour_dists_mean_path = ospath(f'{biasfmri_intermediates_path}/neighbour_vertices/adj_{subject}_{surface}{MSMstring}_{which_neighbours}_dists.npy')
    if subject=='standard': 
        load_neighbours = False
        save_neighbours = False
    if load_neighbours:
        print(f'Loading neighbour vertices in {subject}')
        with open(neighbour_verts_path,'rb') as f:
            neighbour_vertices = pickle.load(f)
        with open(neighbour_dists_path,'rb') as f:
            neighbour_distances = pickle.load(f)
        #neighbour_distances_mean = np.load(neighbour_dists_mean_path)
    else:
        print(f'Finding neighbour vertices in {subject}')
        #neighbour_vertices, neighbour_distances = get_neighbour_vertices(c, mesh, which_neighbours, distance_range,MSMAll=MSMAll)
        if which_neighbours == 'distant':
            from get_gdistances import get_gdistances
            print(f'{c.time()}: Get geodesic distances start')
            d = get_gdistances(subject,surface,10,MSMAll=MSMAll) #d is a sparse matrix in compressed sparse row format
            print(f'{c.time()}: Get vertices at distance range start')
            neighbour_vertices, neighbour_distances = get_vertices_in_distance_range(distance_range,d)
            nVerticesWithZeroNeighbours = np.sum(np.array([len(i) for i in neighbour_vertices])==0)
            print(f'{nVerticesWithZeroNeighbours} vertices with 0 neighbours in distance range {distance_range}')
        elif which_neighbours == 'local':
            print(f'{c.time()}: Get neighbour vertices start')
            neighbour_vertices,neighbour_distances = _get_all_neighbour_vertices(mesh,None)       
        if save_neighbours:
            print(f'Saving neighbour vertices in {subject}')
            with open(neighbour_verts_path,'wb') as f:
                pickle.dump(neighbour_vertices,f)
            with open(neighbour_dists_path,'wb') as f:
                pickle.dump(neighbour_distances,f)
            #np.save(neighbour_dists_mean_path,neighbour_distances_mean)
    neighbour_distances_mean = np.array([np.mean(i) for i in neighbour_distances])
    return neighbour_vertices, neighbour_distances, neighbour_distances_mean

def get_vertices_in_distance_range(distance_range,csr_matrix):
    """
    For each vertex, return a list of vertices within a particular geodesic distance range away from the original vertex, and their corresponding actual distances
    Parameters:
    ------------
    distance_range: tuple, e.g. (2,3)
        minimum and maximum distance away from current vertex
    csr_matrix: compressed sparse row matrix containing distances between vertex pairs
    Returns:
    ------------
    vertices_at_distance: list (nvertices)
        Each element is a list of vertex indices that are in the specified distance range
    distances: list(nvertices)
        Each element is a list of geodesic distances (mm), corresponding to elements within lists within vertices_at_distance
    """
    nvertices = csr_matrix.shape[0]
    vertices_at_distance = [[] for i in range(nvertices)]
    distances = [[] for i in range(nvertices)]
    for nrow in range(nvertices):
        #if nrow%1000==0: print(nrow)
        drow = csr_matrix[nrow,:]
        for i in range(len(drow.indices)):
            distance = drow.data[i]
            if (distance >= distance_range[0]) and (distance <= distance_range[1]):
                vertices_at_distance[nrow].append(drow.indices[i]) 
                distances[nrow].append(distance.astype(np.float32))
    return vertices_at_distance, distances


def get_corr_with_neighbours(nearest_vertices_array,time_series,parallelize=True):
    """
    For each vertex, find correlation between its time series and that of its neighbours
    Parameters:
    ----------
    nearest_vertices_array: Can either be:
        1) Array of shape (nvertices,) containing the index of nearest vertex to each vertex, or 
        2) List/array (nvertices,) where each element is a list of neighbouring vertices
        In scenario 2, return the mean (over neighbouring vertices) correlation between source vertex and each neighbour
    time_series: array of shape (ntimepoints,nvertices)
    parallelize: bool
    Returns:
    ----------
    result: array of shape (nvertices,)
        (mean) correlation between source vertex and its neighbours
    """

    from joblib import Parallel, delayed

    if type(nearest_vertices_array[0]) in [np.ndarray,list]:
        scenario = 2
    else:
        scenario = 1

    nvertices=len(nearest_vertices_array)
    corrs = [i[:] for i in nearest_vertices_array] #of same size as nearest_vertices_array. Stores correlations
    corrs_mean = np.zeros(nvertices,dtype=np.float32)

    if parallelize: 
        def yield_chunks(nearest_vertices_array,nchunks):
            chunk_size = 1 + nvertices//nchunks
            for i in range(0, nvertices, chunk_size):
                chunk_indices = range(i, min(i + chunk_size, nvertices))
                #print(chunk_indices)
                yield chunk_indices
        def compute_correlation(time_series,nearest_vertices_array,scenario,chunk_indices):
            chunked_corrs = [i[:] for i in nearest_vertices_array[slice(chunk_indices.start,chunk_indices.stop)]]
            chunked_corrs_mean = np.zeros(len(chunk_indices),dtype=np.float32)
            for source_vertex_index,source_vertex in enumerate(chunk_indices):
                source_vertex_time_series = time_series[:, source_vertex]
                """
                if scenario == 1:
                    target_vertex_time_series = time_series[:, nearest_vertices_array[source_vertex]]
                    result[source_vertex_index] = np.corrcoef(source_vertex_time_series, target_vertex_time_series)[0, 1]
                
                elif scenario == 2:
                """
                target_vertex_time_series = time_series[:, np.array(nearest_vertices_array[source_vertex])]
                list_of_corrs = [np.corrcoef(source_vertex_time_series, target_vertex_time_series[:, j])[0, 1] for j in range(target_vertex_time_series.shape[1])]
                chunked_corrs[source_vertex_index] = list_of_corrs
                chunked_corrs_mean[source_vertex_index] = np.mean(list_of_corrs)
            return chunked_corrs_mean, chunked_corrs
        temp = Parallel(n_jobs=-1, prefer='processes')(delayed(compute_correlation)(time_series,nearest_vertices_array,scenario,chunk_indices) for chunk_indices in yield_chunks(nearest_vertices_array, 12))
        chunked_corrs_mean, chunked_corrs = zip(*temp)
        corrs_mean = np.concatenate(chunked_corrs_mean)
        import itertools
        corrs = list(itertools.chain.from_iterable(chunked_corrs))
        return corrs_mean, corrs

    else:
        for i in range(nvertices):
            if i % 10000 == 0:
                print(f"Vertex {i} of {nvertices}")  
            source_vertex_time_series = time_series[:, i]
            """
            if scenario == 1:
                target_vertex_time_series = time_series[:, nearest_vertices_array[i]]
                corrs_mean[i] = np.corrcoef(source_vertex_time_series, target_vertex_time_series)[0, 1]
            elif scenario == 2:
            """
            target_vertex_time_series = time_series[:, np.array(nearest_vertices_array[i])]
            temp = [np.corrcoef(source_vertex_time_series, target_vertex_time_series[:, j])[0, 1] for j in range(target_vertex_time_series.shape[1])]
            corrs[i] = temp
            corrs_mean[i] = np.mean(temp)
        return corrs_mean, corrs

def get_corr_with_neighbours_nifti(nifti_image, mask_image = None):
    """
    For each vertex, find correlation between its time series and that of its neighbours, averaged across its immediate neighbours. Exclude voxels outside the background mask. First initialize the result as an empty nibabel image or numpy array. Then loop through each voxel. For each voxel, consider the immediately adjacent 6 voxels (up, down, left, right, front, back) as its neighbours. For each neighbour, find the correlation between its time series and that of the source voxel. Average these 6 correlations and update the result array. Finally, return the result image.  
    Parameters:
    ----------
    nifti_image: nibabel image object in MNI space with dimensions (x,y,z,time)
    mask_image: nibabel image object or filepath, or None
    Returns: 
    ----------
    result: nibabel image object in MNI space with dimensions (x,y,z)
        Contains the mean correlation between each vertex and its neighbours
    """

    """
    from nilearn import maskers
    masker = maskers.NiftiMasker(standardize=True,standardize = 'zscore_sample')
    img2 = masker.fit_transform(nifti_image)
    """

    if mask_image is None:
        from nilearn import masking
        mask = masking.compute_background_mask(nifti_image).get_fdata().astype(bool)
    else:
        mask = mask_image.get_fdata()
    data = nifti_image.get_fdata().astype(np.float32)
    print('got data')

    x, y, z, _ = data.shape  # Ignore the time dimension for the output dimensions
    # Initialize the result array with zeros
    result = np.zeros((x, y, z),dtype=np.float32)
    # Iterate through each voxel in the spatial dimensions
    for i in range(x):
        print(i)
        for j in range(y):
            for k in range(z):
                if mask[i,j,k]: #skip voxels outside the brain mask               
                    correlations = []
                    # Iterate through each neighbour (6 neighbours: up, down, left, right, front, back)
                    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        # Check if neighbour is within bounds and inside the brain
                        if 0 <= ni < x and 0 <= nj < y and 0 <= nk < z and mask[ni, nj, nk]:
                            # Calculate the correlation between the voxel and its neighbour
                            corr = np.corrcoef(data[i,j,k,:], data[ni,nj,nk,:])[0,1]
                            correlations.append(corr)
                    if correlations: #ensure non-empty
                        result[i,j,k] = np.mean(correlations)
    # Create a new NIfTI image from the result array
    import nibabel as nib
    result_img = nib.Nifti1Image(result, affine=nifti_image.affine)
    return result_img



#Other functions

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

def parcel_mean(img,parc_matrix,vertex_area=None):
    """
    Parcellate a 2D image (ntimepoints,nvertices) into a 2D image (ntimepoints,nparcels) by averaging over vertices in each parcel
    Parameters
    ----------
    img : 2D numpy array (ntimepoints,nvertices) or 1D array (nvertices)
    parc_matrix : 2D numpy array (nparcels,nvertices) with 1s and 0s   
    vertex_area: 1D numpy array (nvertices,) or None
        Normalize values by vertex area 
    """
    if vertex_area is not None:
        print('Normalizing values by vertex areas')
        img = img / vertex_area

    parcel_sums = img @ parc_matrix.T #sum of all vertices in each parcel
    nvertices_sums=parc_matrix.sum(axis=1) #no. of vertices in each parcel
    if img.ndim==1:
        return parcel_sums / np.squeeze(np.array(nvertices_sums))
    elif img.ndim==2:
        nvertices_sum_expanded = np.tile(nvertices_sums,(1,img.shape[0])).T
        return np.squeeze(np.array(parcel_sums/nvertices_sum_expanded))

def subtract_parcelmean(data,parc_matrix):
    """
    Given a data vector, calculate and subtract the parcel-mean value from each vertex. First, find the mean value for each parcel. Then, multiply by parc_matrix to get the parcel-mean value for each vertex. Then, subtract this from the data. Finally, add mean value of whole-brain to each vertex
    Parameters:
    ------------
    data: np.array, shape (nvertices,)
    parc_matrix: np.array, shape (nvertices,nparcs)
    Returns:
    ------------
    np.array, shape (nvertices,)
    """
    parc_means = parcel_mean(np.reshape(data,(1,-1)),parc_matrix)
    parc_means_at_vertices = parc_means @ parc_matrix
    return (data - parc_means_at_vertices)


def get_smoothing_kernel(subject,surface,fwhm_for_gdist,smooth_noise_fwhm,MSMAll=False,mesh_template='fsLR32k'):
        from get_gdistances import get_gdistances
        from Connectome_Spatial_Smoothing import CSS as css
        gdistances = get_gdistances(subject,surface,fwhm_for_gdist,epsilon=0.01,load_from_cache=True,save_to_cache=True,MSMAll=MSMAll,mesh_template=mesh_template)
        skernel = css._local_distances_to_smoothing_coefficients(gdistances, css._fwhm2sigma(smooth_noise_fwhm))
        return skernel
def smooth(x,skernel):
    return skernel.dot(x.T).T

def faces2connectivity(faces):
    """
    Given a list of faces, return a sparse connectivity matrix. First convert the list of faces to a list of edges, then convert this list of edges to a sparse connectivity matrix
    Parameters:
    ----------
    faces: np.array (nfaces,3)
        List of faces
    Returns:
    -------
    result: sparse matrix (nvertices,nvertices)
    """
    edges = np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]],faces[:,[1,0]],faces[:,[2,1]],faces[:,[0,2]]],axis=0)
    edges = np.sort(edges,axis=1)
    edges = np.unique(edges,axis=0)
    from scipy import sparse
    coo_matrix = sparse.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),shape=(edges.max()+1,edges.max()+1))
    result = coo_matrix.tocsr()
    return result,edges

def find_edges_left(edges, ngrayl):
    edges_left_bool = np.zeros(edges.shape[0]).astype(bool)
    for i in range(edges.shape[0]):
        first_vert = edges[i,0]
        second_vert = edges[i,1]
        if (first_vert < ngrayl) and (second_vert < ngrayl):
            edges_left_bool[i] = True
    edges_left = edges[edges_left_bool,:]
    return edges_left

def get_border_vertices(ngrayl,edges_left,labels):
        
    #Get border points on left hemisphere (points straddling a parcel boundary)
    n_repeats = labels.shape[1]
    border = np.zeros((ngrayl,n_repeats))
    for j in range(n_repeats):
        for i in range(edges_left.shape[0]):
            first_vert = edges_left[i,0]
            second_vert = edges_left[i,1]
            if (labels[first_vert,j]!=labels[second_vert,j]):
                border[first_vert,j]=1
                border[second_vert,j]=1
    """
    #Make a map of 'distance' from the border, for each vertex. Initialize by setting vertices adjacent to the border (n=1) as the reference points. In each iteration, for any vertices which are adjacent to any reference points and still at value 0, set their value to (n+1). Iterate for n = 1,2,3,4
    n=0
    n_max = 0
    while np.any(border==0) and n<n_max:
        n+=1
        print(f"n={n}, {np.sum(border==0)}/{len(border)} vertices left")
        for i in range(edges_left.shape[0]): #iterate through each vertex
            first_vert = edges_left[i,0]
            second_vert = edges_left[i,1]
            if border[first_vert] == n and border[second_vert] == 0:
                border[second_vert] = n+1
            elif border[second_vert] == n and border[first_vert] == 0:
                border[first_vert] = n+1
    border_bool = border>0 
    border[border==0] = n_max+2
    """
    return border

def get_cohen_d(x,y):
    #correct if the population S.D. is expected to be equal for the two groups
    from numpy import std, mean, sqrt
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)


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

def sum_each_quantile(x,y,n):
    """
    x and y are vectors of the same length. Divide x into n quantiles. For each quantile, find the mean x value, and sum the corresponding y values. Return the mean x values and summed y values for each quantile
    """
    x_sorted = np.sort(x)
    y_sorted = y[np.argsort(x)]
    x_quantiles = np.array_split(x_sorted,n)
    y_quantiles = np.array_split(y_sorted,n)
    x_means = [np.mean(q) for q in x_quantiles]
    y_sums = [np.sum(q) for q in y_quantiles]
    return x_means,y_sums

def do_spin_test(x,mask,n_perm):
    from neuromaps import stats, nulls
    print(f"Do spin test")
    x2=fillnongray(x,mask)
    x2_nulls = nulls.alexander_bloch(x2,atlas='fsLR',density='32k',n_perm=n_perm,seed=0)
    return x2_nulls[mask]

def corr_with_nulls(x,y,mask,method='spin_test',n_perm=100):
    """
    Put zeros in the non-gray vertices of brain maps x and y, then correlate them using a null method
    Parameters:
    ----------
    x: np.array, shape (59412,)
        Brain map in fsLR 32k space without medial wall vertices (i.e. 59,412 vertices)
    y: np.array, shape (59412,)
    method: str, 'spin_test' or 'eigenstrapping'
    n_perm: int, number of permutations for the null method
    Returns:
    ----------
    corr: float, correlation between x and y
    pval: float, p-value of the correlation
    """

    if method=='no_null':
        from scipy.stats import pearsonr
        corr,pval = pearsonr(x,y)
    else:
        from neuromaps import stats, nulls 
        if method=='spin_test':
            x_nulls = do_spin_test(x,mask,n_perm)
        corr,pval = stats.compare_images(x,y,nulls=x_nulls,metric='pearsonr')
    return corr,pval

def get_interp_dists(distance_range):
    #Return distance values at which to linear-spline interpolate, 0mm, 1mm, 2mm, 3mm, etc
    #e.g. if distance range is (2,6), return np.array([4,5,6])
    minval = int(np.ceil(distance_range[0]))
    minval = max(minval,4) #only interpolate for distances 4mm and above
    maxval = int(np.floor(distance_range[1]+1))
    interp_dists = np.array(range(minval,maxval)) 
    return interp_dists

def corr_dist_fit_linear_spline(c,interp_dists,all_neighbour_distances_sub,ims_adjcorr_full_sub):
    """
    For each vertex, fit a linear spline to the correlation-distance curve. Interpolate the correlation at integer distances. Return the interpolated correlations for each vertex, for each distance value
    """
    from scipy.interpolate import interp1d
    nvertices=len(all_neighbour_distances_sub)
    interp_corrs = np.zeros((nvertices,len(interp_dists))) #save interpolated correlations
    #Get interpolated correlations for each vertex, for each distance value
    for nvertex in range(nvertices):
        if nvertex%10000==0: print(f'{c.time()}: Vertex {nvertex}/{nvertices}')
        distsx = all_neighbour_distances_sub[nvertex]
        corrsx = ims_adjcorr_full_sub[nvertex]
        f = interp1d(distsx, corrsx, kind='linear', fill_value='extrapolate') #fit linear spline
        interp_corrs[nvertex,:] = f(interp_dists)
    interp_corrs[interp_corrs<0] = 0 #set min and max interpolated values to 0 and 1
    interp_corrs[interp_corrs>1] = 1
    return interp_corrs

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def corr_dist_fit_exp(c,distance_range,all_neighbour_distances_sub,ims_adjcorr_full_sub,n_jobs=-1,prefer='processes'):
    """
    For each vertex, fit an exponential curve to the correlation-distance curve. Return expfit parameters.
    Ignore OptimizeWarning
    """
    from scipy.optimize import curve_fit
    from joblib import Parallel,delayed
    def func(x,y):
        popt, pcov = curve_fit(exp_func, x, y,p0=[1.5,0.5,0.1],bounds=((0.5,0.001,-0.1),(2.5,1.0,0.5)))
        return popt
    nvertices=len(all_neighbour_distances_sub)
    temp = Parallel(n_jobs=n_jobs,prefer=prefer)(delayed(func)(all_neighbour_distances_sub[i],ims_adjcorr_full_sub[i]) for i in range(nvertices))
    expfit_params = np.vstack(temp)
    return expfit_params

def interpolate_from_expfit_params(expfit_params,interp_dists):
    """
    Given exponential fit parameters for each vertex, interpolate correlation values at specific x-values given in "interp_dists"
    """
    nvertices=expfit_params.shape[0]
    interp_corrs = np.zeros((nvertices,len(interp_dists))) #interpolated correlations at integer distances
    for nvertex in range(nvertices):
        for ndist in range(len(interp_dists)):
            dist = interp_dists[ndist]
            interp_corrs[nvertex,ndist] = exp_func(dist, expfit_params[nvertex,0], expfit_params[nvertex,1], expfit_params[nvertex,2])
    return interp_corrs

def corr_dist_plot_linear_spline(c,interp_dists,interp_corrs,sulcs_subject,mask,null_method,n_perm):
    #plot interpolated correlations (at particular distance value) as a function of sulcal depth
    from matplotlib import pyplot as plt
    ndists = len(interp_dists)
    nrows = int(np.ceil(np.sqrt(ndists)))
    fig,axs = plt.subplots(nrows,nrows,figsize=(10,7))
    for i in range(ndists): #iterate through distances, 0mm, 1mm, etc
        corrs = interp_corrs[:,i]
        ax=axs.flatten()[i]
        valid = ((corrs!=0) & (corrs!=1))
        ax.scatter(sulcs_subject[valid],corrs[valid],1,alpha=0.05,color='k')
        ax.set_xlabel('Sulcal depth')
        ax.set_ylabel('Interpolated correlation')
        corr,pval = corr_with_nulls(sulcs_subject,corrs,mask,null_method,n_perm)
        ax.set_title(f'Distance {interp_dists[i]}mm\nR={corr:.3f}, p={pval:.3f}')
    fig.tight_layout()
    return fig,axs

def corr_dist_plot_exp(c,expfit_params,sulcs_subject,mask,null_method,n_perm):
    #plot exponential fit parameters as a function of sulcal depth
    from matplotlib import pyplot as plt
    expfit_param_names = ['Amplitude','Decay rate','Bias']
    fig,axs = plt.subplots(3,figsize=(4,9))
    for i in range(3):
        values = expfit_params[:,i]
        ax=axs.flatten()[i]
        ax.scatter(sulcs_subject,values,1,alpha=0.05,color='k')
        ax.set_xlabel('Sulcal depth')
        ax.set_ylabel(f'{expfit_param_names[i]}')
        corr,pval = corr_with_nulls(sulcs_subject,values,mask,null_method,n_perm)
        ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
    fig.tight_layout()
    return fig,axs

def corr_dist_plot_samples(all_neighbour_distances_sub,ims_adjcorr_full_sub,distance_range,interp_dists,interp_corrs):
    # Plot correlation vs. distance for some example vertices, and the linear spline fit
    from matplotlib import pyplot as plt
    nvertices=len(all_neighbour_distances_sub)
    samplevertices = np.linspace(0,nvertices-1,16).astype(int)
    nrows = int(np.ceil(np.sqrt(len(samplevertices))))
    fig,axs = plt.subplots(nrows,nrows,figsize=(10,7))
    for i,nvertex in enumerate(samplevertices):
        ax=axs.flatten()[i]
        distsx = all_neighbour_distances_sub[nvertex]
        corrsx = ims_adjcorr_full_sub[nvertex]
        interp_corrs_vertex = interp_corrs[nvertex,:]
        ax.scatter(distsx,corrsx,20,alpha=1,color='k')
        ax.plot(interp_dists, interp_corrs_vertex, '-', color='r')
        ax.set_xlim(distance_range[0],distance_range[1])
        ax.set_ylim(0,1)
        ax.set_xlabel('Distance from source vertex (mm)')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Source vertex is {nvertex}')
    fig.tight_layout()
    return fig,axs


def ttest_ind_with_nulldata_given(groups,observed_data,null_data):
    """
    Vector "group" which contains group memberships of each individual, and another vector "data" which contains the values for each individual. I wish to compare the mean value in "data" across the two different groups. I wish to do the test non-parametrically. To this end, I have generated 100 randomized versions of "data" where the values are randomized across all individuals. This null dataset is given in variable "data_null", and it is a 2D matrix (number of individuals, number of surrogates). I wish do conduct a t-test for group differences in "data", and compare the t-statistic to the distribution of t-statistics when I do the same thing with "data_null", and hence derive a p-value for the deviation of the observed statistic from the expected distribution from the null
    """
    from scipy.stats import ttest_ind
    group_names = np.unique(groups)
    def calculate_t_statistic(data, groups):
        group_A = data[groups == group_names[0]]
        group_B = data[groups == group_names[1]]
        tstat, p_val = ttest_ind(group_A, group_B, equal_var=True)
        return tstat
    observed_t_stat = calculate_t_statistic(observed_data,groups)
    n_perm = null_data.shape[1]
    null_t_stats = np.array([calculate_t_statistic(null_data[:,i],groups) for i in range(n_perm)])
    percentile = (np.sum(null_t_stats <= observed_t_stat) / len(null_t_stats))
    p_value = 2 * min(percentile, 1 - percentile)
    if p_value==0: 
        p_value = 1/n_perm
    cohen_d = get_cohen_d(observed_data[groups==group_names[0]],observed_data[groups==group_names[1]])
    return cohen_d, observed_t_stat, p_value

"""
def addfit(x,y,ax,linewidth=1,color='black'):
    #add a line of best fit
    m,b = np.polyfit(x,y,1)
    ax.plot(x,m*x+b,color=color,linewidth=linewidth)
"""

def get_geodesic_distances_within_masked_mesh(mesh,mask):
    """
    Compute all pairwise geodesic distances within a masked region of a mesh
    """
    import gdist
    verts,faces=mesh[0],mesh[1]
    verts_singleparc = verts[mask] #vertices in the single parcel
    faces_singleparc = triangles_removenongray(faces,mask)
    r_sparse=gdist.local_gdist_matrix(verts_singleparc.astype(np.float64),faces_singleparc)
    r = r_sparse.astype(np.float32).toarray()
    return r