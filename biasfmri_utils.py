
import numpy as np

project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"

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

def fillnongray(arr,mask):
    """
    Takes a 1D array of fMRI grayordinates and returns the values on the vertices of the left cortex mesh which is neccessary for surface visualization. 
    The unused vertices are filled with a constant (zero by default).
    E.g. stepping up from 18k to 20k vertices 
    Like hcp_utils.cortex_data()
    """
    out = np.zeros(len(mask))
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

def get_nearest_vertices(mesh,mask):
    """
    Given a mesh, return the list of nearest vertices for each vertex, and their distances. Initiate both lists as numpy arrays first. Initialize the distances array with Inf values. Then loop through the triangles. For each triangle, consider the 3 vertex pairs. For each vertex, find distance to the two other vertices. If the closest of these two distances is less than the current distance, replace the current distance and nearest vertex. Finally, return values for the vertices that are gray matter only
    Parameters:
    ----------
    mesh: tuple (vertices,triangles)
        vertices: array (nvertices,3)
        triangles: array (ntriangles,3)
    mask: boolean array (nvertices,)
    """
    vertices=mesh[0]
    faces=mesh[1]
    nearest_vertices=np.zeros(len(vertices),dtype=int)
    nearest_distances=np.zeros(len(vertices),dtype=np.float32)
    nearest_distances[:]=np.inf
    for face in faces:
        for i in range(3):
            source_vertex=face[i]
            vertex1=face[(i+1)%3]
            vertex2=face[(i+2)%3]
            distance1=np.linalg.norm(vertices[source_vertex]-vertices[vertex1])
            distance2=np.linalg.norm(vertices[source_vertex]-vertices[vertex2])
            if distance1 < nearest_distances[source_vertex]:
                nearest_distances[source_vertex] = distance1
                nearest_vertices[source_vertex] = vertex1
            if distance2 < nearest_distances[source_vertex]:
                nearest_distances[source_vertex] = distance2
                nearest_vertices[source_vertex] = vertex2
    nearest_distances = nearest_distances[mask]
    nearest_vertices = (removenongray(mask)[nearest_vertices])[mask]
    return nearest_vertices,nearest_distances