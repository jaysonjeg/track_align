"""
Utility functions for working with brain meshes
"""

import numpy as np
import nibabel as nib #needed for surf_file_to_mesh
import gdist #required by get_geodesic_distances_within_masked_mesh
import generic_utils as gutils


### GETTING VERTICES AND TRIANGLES FROM SURFACE FILES ###

def surf_file_to_mesh(surface_gifti_filepath):
    """
    Given a filepath for a .gii surface file, return the vertices and triangles as numpy arrays
    Old name: get_surface
    Parameters:
    ---------
    surface_gifti_filepath: str

    Returns:
    ---------
    vertices: np.ndarray
        (nvertices,3)
    """
    surface=nib.load(surface_gifti_filepath)
    vertices=surface.darrays[0].data #array (nvertices,3)
    triangles=surface.darrays[1].data #array (ntriangles,3)
    return vertices,triangles


def surf_files_bihemispheric_to_mesh(surface_gifti_filepath_left,surface_gifti_filepath_right):
    """
    Given filepaths for left and right hemisphere .gii surface files, return the vertices and triangles as numpy arrays
    Old name get_verts_and_triangles_from_surf_file
    Parameters:
    ---------
    surface_gifti_filepath_left: str
    surface_gifti_filepath_right: str

    Returns:
    ---------
    all_vertices: np.ndarray
        (nvertices,3)
    all_triangles: np.ndarray
        (ntriangles,3)
    """
    Lvertices,Ltriangles=surf_file_to_mesh(surface_gifti_filepath_left)
    Rvertices,Rtriangles=surf_file_to_mesh(surface_gifti_filepath_right)
    Rtriangles_new=Rtriangles+Lvertices.shape[0] #R triangles now numbered 32492 to 64983
    all_vertices=np.vstack((Lvertices,Rvertices)) #array (64984,3)
    all_triangles=np.vstack((Ltriangles,Rtriangles_new))
    return all_vertices,all_triangles

def hcp_get_surf_file_hemi(sub,hemi,surface_type,MSMAll=False,folder='MNINonLinear',version='fsaverage_LR32k',hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200"):
    """
    Get filepath for a subject-specific single hemisphere surface file from HCP directory structure
    Old name: get_surface_filepath_hemi
    Parameters:
    ---------
    sub: str
    hemi: 'L' or 'R'
    surface_type: 'white', 'pial', 'midthickness' or 'inflated'
    MSMAll: bool
    folder: str
        'MNINonLinear' or 'T1w'
    version: str
        'native', 'fsaverage_LR32k' or '164k'
    hcp_folder: str

    Returns:
    ----------
    surf_file: str
    """
    if MSMAll:
        MSMstring = '_MSMAll'
    else:
        MSMstring = ''

    if folder=='MNINonLinear':
        if version=='fsaverage_LR32k':
            surf_file=gutils.ospath(f"{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.{hemi}.{surface_type}{MSMstring}.32k_fs_LR.surf.gii")
        elif version=='native':
            surf_file=gutils.ospath(f"{hcp_folder}/{sub}/MNINonLinear/Native/{sub}.{hemi}.{surface_type}.native.surf.gii")
        elif version=='164k':
            surf_file=gutils.ospath(f"{hcp_folder}/{sub}/MNINonLinear/{sub}.{hemi}.{surface_type}{MSMstring}.164k_fs_LR.surf.gii")
        else: 
            assert(0)
    elif folder=='T1w':
        if version=='native':
            surf_file=gutils.ospath(f"{hcp_folder}/{sub}/T1w/Native/{sub}.{hemi}.{surface_type}.native.surf.gii")
        else:
            assert(0)
    return surf_file

def hcp_get_mesh_hemi(sub,hemi,surface_type,MSMAll=False,folder='MNINonLinear',version='fsaverage_LR32k',hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200"):
    """
    Get vertices and triangles for a subject-specific single hemisphere surface from HCP directory structure
    Parameters:
    ---------
    sub: str
    hemi: 'L' or 'R'
    surface_type: 'white', 'pial', 'midthickness' or 'inflated'
    MSMAll: bool
    folder: str
        'MNINonLinear' or 'T1w'
    version: str
        'native', 'fsaverage_LR32k' or '164k'
    hcp_folder: str

    Returns:
    ---------
    vertices: np.ndarray
        (nvertices,3)
    triangles: np.ndarray
        (ntriangles,3)
    """
    surface_filepath = hcp_get_surf_file_hemi(sub,hemi,surface_type,MSMAll,folder,version,hcp_folder)
    vertices,triangles=surf_file_to_mesh(surface_filepath)
    return vertices,triangles

def hcp_get_mesh(sub,surface_type,MSMAll=False,folder='MNINonLinear',version='fsaverage_LR32k',hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200"):
    """
    Get bi-hemispheric vertices and triangles for a subject-specific surface from HCP directory structure
    Old name: get_verts_and_triangles
    Parameters:
    ---------
    sub: str
    surface_type: str
        'white', 'pial', 'midthickness' or 'inflated'
    MSMAll: bool
    folder: str
        'MNINonLinear' or 'T1w'
    version: str
        'native', 'fsaverage_LR32k' or '164k'
    hcp_folder: str

    Returns:
    ---------
    all_vertices: np.ndarray
        (nvertices,3)
    all_triangles: np.ndarray
        (ntriangles,3)
    """
    surface_filepath_left = hcp_get_surf_file_hemi(sub,'L',surface_type,MSMAll,folder,version,hcp_folder)
    surface_filepath_right = hcp_get_surf_file_hemi(sub,'R',surface_type,MSMAll,folder,version,hcp_folder)
    all_vertices,all_triangles=surf_files_bihemispheric_to_mesh(surface_filepath_left,surface_filepath_right)
    return all_vertices,all_triangles

### MANIPULATING BRAIN SURFACE MESHES ###

def fillnongray(array,mask,fillvalue=0):
    """
    Given a mask boolean array, fill the True values with values given in the input array. Fill the False values with a constant (zero by default). If a gray matter mask and a gray matter data array are passed, this function will fill the non-gray matter vertices with zeros. 
    Parameters:
    ---------
    array: np.ndarray
        1D array of data values
    mask: np.ndarray
        boolean array of vertices. Same number of True values as elements in the array
    fillvalue: float
        Value to fill the mask==False vertices with
    """
    out = np.zeros(len(mask))
    out[:]=fillvalue
    out[mask] = array
    return out

def renumbering(mask):
    """
    Returns a numbering for all True indices in the mask. For example, let's say there were 7 vertices originally, of which 4 survive masking. The mask is np.array([0,0,1,1,1,0,1]). Then this function returns np.array([0,0,0,1,2,0,3]). Let's say we had a list of original vertex indices x=[3,6]. Then these vertice new indices are [1,3]. This is obtained using removenongray(mask)[x]. This is useful for renumbering lists of vertex indices based on a gray matter mask.
    Old name: removenongray
    Parameters:
    ---------
    mask: np.ndarray
        boolean array of vertices

    Returns:
    ---------
    out: np.ndarray
        array of same length as mask. False values in the mask remain as zeros. True values in the mask are numbered (0,1,2,3..). 
    """   
    indices_where_mask_is_true = np.where(mask==1)[0] #list (18k gray matter vertices) of indices (in full mesh)
    out = np.zeros(len(mask),dtype=int)
    for index,value in enumerate(indices_where_mask_is_true):
        out[value]=index
    return out

def triangles_removenongray(triangles,mask):
    """
    Given a list of triangles in a mesh, return a new list of triangles that only contain vertices in a vertex mask. First remove the rows containing non-gray vertices, then renumber the vertices so they are 0 to nGrayVertices
    Parameters:
    ---------
    triangles: np.ndarray
        array (n,3) list of triangles in a mesh
    mask: np.ndarray
        boolean array indicating which vertices are gray matter
    """
    gray = np.where(mask)[0]
    temp=np.isin(triangles,gray)
    temp2=np.all(temp,axis=1)
    triangles_v2 = triangles[temp2,:]
    triangles_v3 = renumbering(mask)[triangles_v2] #renumbering
    return triangles_v3

def get_triangle_areas(mesh):
    """
    Given a mesh, return the list of areas corresponding to each triangle 
    Parameters:
    ----------
    mesh: tuple (verts,triangles)
        verts: np.array, shape (nvertices,3)
        triangles: np.array, shape (triangles,3)
    Returns:
    ----------
    areas: np.array, shape (triangles,)
    """
    from scipy.spatial import distance
    vertices=mesh[0]
    triangles=mesh[1]
    vertices_of_triangles = [vertices[triangles[i],:] for i in range(len(triangles))] #list(triangles) of arrays (3,3). In the array, each row is a face, each column is a vertex coordinate
    def get_triangle_area(coords):
        a=distance.euclidean(coords[0],coords[1])
        b=distance.euclidean(coords[1],coords[2])
        c=distance.euclidean(coords[2],coords[0])
        s=(a+b+c)/2
        return np.sqrt(s*(s-a)*(s-b)*(s-c))
    areas = np.array([get_triangle_area(coords) for coords in vertices_of_triangles])
    return areas

def get_vertex_areas(mesh):
    """
    Given a mesh, calculate the area corresponding to each vertex. For each triangle, give 1/3 of its area to each of its 3 vertices
    Parameters:
    ----------
    mesh: tuple (verts,triangles)
        verts: np.array, shape (nvertices,3)
        triangles: np.array, shape (triangles,3)
    Returns:
    ----------
    vertex_areas: np.array, shape (nvertices,)
    """
    vertices=mesh[0]
    triangles=mesh[1]
    triangle_areas = get_triangle_areas(mesh)
    vertex_areas = np.zeros(len(vertices))
    for i, triangle in enumerate(triangles):
        for vertex in triangle:
            vertex_areas[vertex] += triangle_areas[i]/3
    return vertex_areas

def get_parcel_areas(vertex_areas,parc_labels):
    """
    Calculate surface area corresponding to each parcel
    Parameters:
    ----------
    vertex_areas: np.array, shape (nvertices,)
    parc_labels: np.array, shape (nvertices,)
        Parcel labels
    Returns:
    ----------
    parcel_areas: np.array, shape (nparcels,)
    """
    unique_labels = np.unique(parc_labels)
    nparcs = len(unique_labels)
    parcel_areas = np.zeros(nparcs)
    for i,parcel_index in enumerate(unique_labels):
        mask = (parc_labels==parcel_index)
        parcel_areas[i] = np.sum(vertex_areas[mask])
    return parcel_areas


def triangles2edges(triangles):
    """
    Given a list of mesh triangles (faces), return a sparse connectivity matrix of mesh edges. First convert the list of faces to a list of edges, then convert this list of edges to a sparse connectivity matrix
    Parameters:
    ----------
    triangles: np.array (nfaces,3)
        List of triangles
    Returns:
    -------
    result: sparse matrix (nvertices,nvertices)
    edges: np.ndarray
        Array of shape (nedges,2). Each row contains indices of two vertices that form an edge
    """
    from scipy import sparse
    edges = np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]],triangles[:,[1,0]],triangles[:,[2,1]],triangles[:,[0,2]]],axis=0)
    edges = np.sort(edges,axis=1)
    edges = np.unique(edges,axis=0)
    coo_matrix = sparse.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),shape=(edges.max()+1,edges.max()+1))
    result = coo_matrix.tocsr()
    return result,edges

def get_border_vertices(edges,labels):
    """
    Given a list of mesh edges and a list of vertex labels, return a boolean array indicating which vertices are on the border of a parcel. A vertex is on the border if it is connected to a vertex with a different label
    Parameters:
    ----------
    edges: np.ndarray
        Array of shape (nedges,2). Each row contains indices of two vertices that form an edge. Obtain this using triangles2edges(triangles)
    labels: np.ndarray
        Array of shape (nvertices,) containing the label of each vertex
    """
    border = np.zeros(len(labels))
    for i in range(edges.shape[0]):
        first_vert = edges[i,0]
        second_vert = edges[i,1]
        if (labels[first_vert]!=labels[second_vert]):
            border[first_vert]=1
            border[second_vert]=1
    return border

def reduce_mesh(mesh,mask):
    """
    Given a mesh surface, remove vertices not in mask
    Parameters:
    ----------
    mesh: tuple (verts,faces)
        verts: np.array, shape (nvertices,3)
        faces: np.array, shape (nfaces,3)
    mask: np.array, shape (nvertices,)
        Boolean mask
    Returns:
    ----------
    verts_masked: np.array, shape (nvertices,3)
        Vertices in the single parcel
    faces_masked: np.array, shape (nfaces,3)
        Faces in the single parcel
    """
    verts,faces=mesh[0],mesh[1]
    verts_masked = verts[mask] #vertices in the single parcel
    faces_masked = triangles_removenongray(faces,mask)
    return verts_masked, faces_masked


def get_gdists(vertices,faces):
    """
    Get geodesic distances between all vertex pairs in a mesh
    Parameters:
    ----------
    vertices: np.array, shape (nvertices,3)
        Vertex coordinates
    faces: np.array, shape (nfaces,3)
        Faces of the mesh
    Returns:
    ----------
    r: np.array, shape (nvertices,nvertices)
        Geodesic distance matrix
    """
    r_sparse=gdist.local_gdist_matrix(vertices.astype(np.float64),faces)
    r = r_sparse.astype(np.float32).toarray()
    return r


def get_gdists_within_masked_mesh(mesh,mask):
    """
    Compute all pairwise geodesic distances between vertices within a masked region of a mesh
    Parameters:
    ----------
    mesh: tuple (verts,faces)
        verts: np.array, shape (nvertices,3)
        faces: np.array, shape (nfaces,3)
    mask: np.array, shape (nvertices,)
        Boolean mask
    Returns:
    ----------
    r: np.array, shape (nvertices,nvertices)
        Geodesic distance matrix
    """
    verts_singleparc, faces_singleparc = reduce_mesh(mesh,mask)
    return get_gdists(verts_singleparc,faces_singleparc)

def get_gdists_singleparc(mesh,parc_labels,parc_index):
    """
    Compute all pairwise geodesic distances between vertices within a single parcel
    Parameters:
    ----------
    mesh: tuple (verts,faces)
        verts: np.array, shape (nvertices,3)
        faces: np.array, shape (nfaces,3)
    parc_labels: np.array, shape (nvertices,)
        Parcel labels
    parc_index: int
        Target parcel index
    """
    mask = (parc_labels==parc_index)
    gdists = get_gdists_within_masked_mesh(mesh,mask)
    return gdists