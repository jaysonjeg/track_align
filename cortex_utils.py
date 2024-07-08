"""
Utility functions for dealing with brain data
Requires nilearn, hcp_utils
"""

import numpy as np
import os, pickle
import hcp_utils as hcp
import nibabel as nib
from generic_utils import ospath

hcp_folder='/mnt/d/FORSTORAGE/Data/HCP_S1200'
intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/intermediates'
results_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/results'

class surfplot():
    """
    Plot surface functional activations. Data is array(59412,).
    p=surfplot('/mnt/d/Users/Jayson/Figures')
    p.plot(data,'Figure1')
    """
    def __init__(self, figpath,mesh=None,vmin=None,vmax=None,cmap='inferno',symmetric_cmap=True,plot_type='open_in_browser'):
        from pathlib import Path
        if mesh is None:
            import hcp_utils as hcp
            mesh = hcp.mesh.midthickness
        self.mesh=mesh
        self.figpath=figpath
        if plot_type=='save_as_html':
            filepath=Path(ospath(figpath))
            if not(filepath.exists()):
                os.mkdir(filepath)
        self.vmin=vmin
        self.vmax=vmax
        self.cmap=cmap
        self.symmetric_cmap=symmetric_cmap
        self.plot_type=plot_type
    def plot(self,data,savename=None,vmin=None,vmax=None,cmap=None,symmetric_cmap=None):
        from nilearn import plotting
        import hcp_utils as hcp
        """
        if data.shape[0]<59412: #fill missing data
            ones=np.ones((59412))*(min(data)-0.5*(max(data)-min(data)))
            ones[0:data.shape[0]]=data
            data=ones
        """
        data = np.squeeze(data) 
        if np.min(data)<0: 
            self.symmetric_cmap=True
        else: 
            self.vmin=np.min(data)
            self.symmetric_cmap=False

        if symmetric_cmap is not None: self.symmetric_cmap=symmetric_cmap
        if self.symmetric_cmap==True: self.cmap='bwr'
        elif self.symmetric_cmap==False: self.cmap='inferno'
        if cmap is not None: self.cmap=cmap
        if vmin is not None: self.vmin=vmin
        if vmax is not None: self.vmax=vmax


        #if self.mesh[0].shape[0] > 59412: #if using full 64,983-vertex mesh
        if len(data) < self.mesh[0].shape[0]: #if data is shorter than full mesh
            new_data = hcp.cortex_data(data)
        else:
            new_data = data

        view=plotting.view_surf(self.mesh,new_data,cmap=self.cmap,vmin=self.vmin,vmax=self.vmax,symmetric_cmap=self.symmetric_cmap) 
        if self.plot_type=='save_as_html':
            view.save_as_html(ospath('{}/{}.html'.format(self.figpath,savename)))
        elif self.plot_type=='open_in_browser':
            view.open_in_browser()
        self.vmin=None
        self.vmax=None


def get(filename,vertices=slice(0,None)):
    """
    filename corresponds to cifti-2 image
    Returns surface data as numpy array
    Default all vertices
    """
    return nib.load(filename).get_fdata()[:,vertices]


def vertexmap_59kto64k(hemi='both'):
    """
    List of 59k cortical vertices in fsLR32k, with their mapping onto 64k cortex mesh
    hemi='both','L','R'
    """
    import hcp_utils as hcp
    grayl=hcp.vertex_info.grayl
    grayr=hcp.vertex_info.grayr
    grayr_for_appending=hcp.vertex_info.grayr+hcp.vertex_info.num_meshl
    grayboth=np.hstack((grayl,grayr_for_appending))
    if hemi=='both': return grayboth
    elif hemi=='L': return grayl
    elif hemi=='R': return grayr

def get_fsLR32k_mask(hemi='both'):
    """
    Returns a boolean array indicating, for each vertex in fsaverage5 surface, whether it is gray matter (1) or medial wall (0)
    hemi='both','L','R'
    """
    gray = vertexmap_59kto64k(hemi=hemi)
    num_mesh_64k = gray.max()+1 #number of vertices in 64k cortex mesh
    """
    import hcp_utils as hcp
    if hemi=='both':
        num_mesh_64k = hcp.vertex_info.num_meshl+hcp.vertex_info.num_meshr
    elif hemi=='L':
        num_mesh_64k = hcp.vertex_info.num_meshl
    elif hemi=='R':
        num_mesh_64k = hcp.vertex_info.num_meshr
    """
    temp=np.zeros(num_mesh_64k,dtype=bool)
    for index,value in enumerate(gray):
        temp[value]=True
    return temp

def parc_char_matrix(parc):
    """
    Similar to connectome-spatial-smoothing.parcellation_characteristic_matrix
    parc is parcellation e.g. Schaefer(300)
    """
    from scipy import sparse
    parcellation_matrix=np.zeros((max(parc)+1,59412))
    for i in range(len(parc)):
        value=parc[i]
        parcellation_matrix[value,i]=1
    return list(set(parc)),sparse.csr_matrix(parcellation_matrix).astype(np.float32)         

def Schaefer_original(nparcels):
    #get Schaefer Kong surface parcellation
    filename=ospath('/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/SchaeferParcellations/HCP/fslr32k/cifti/Schaefer2018_{}Parcels_Kong2022_17Networks_order.dlabel.nii'.format(nparcels))
    mask = get_fsLR32k_mask()
    return get(filename).squeeze()[mask].astype(int)
    #return cortex_64kto59k(get(filename).squeeze()).astype(int)
def Schaefer(nparcels):
    save_folder=f'{intermediates_path}\schaeferparcellation'
    save=ospath(f'{save_folder}/schaefer_{nparcels}parcs.p')
    return pickle.load( open( ospath(save), "rb" ) )    
def Schaefer_matrix(nparcels):
    save_folder=f'{intermediates_path}\schaeferparcellation'
    save=ospath(f'{save_folder}/schaefer_{nparcels}parcs_matrix.p')
    return pickle.load( open( ospath(save), "rb" ) )  
def kmeans(nparcels):
    #get my random kmeans surface parcellation
    save_folder=f'{intermediates_path}\kmeansparcellation'
    #save=ospath(f'{save_folder}/funckmeansparc_3subs_4movies_pca100_{nparcels}.p')
    save=ospath(f'{save_folder}/kmeansparc_sub100610_sphere_{nparcels}parcs.p')
    return pickle.load( open( ospath(save), "rb" ) ) 
def kmeans_matrix(nparcs):
    #get parc_matrix for kmeans parcellation
    save_folder= f'{intermediates_path}\kmeansparcellation'
    save=ospath(f'{save_folder}/kmeansparc_sub100610_sphere_{nparcs}parcs_matrix.p')
    return pickle.load( open( ospath(save), "rb" ) )


def parcellation_string_to_parcellation(parcellation_string,subjects=None):
    #Inputs: parcellation_string: 'S300' for Schaefer 300, 'K400' for kmeans 400, 'R10' for searchlight radius 10mm, 'M' for HCP multimodal parcellation. 'I300' for individualized from Kong(2022)
    #Returns an array of size (59412,) with parcel labels for each vertex in fs32k cortex
    import hcp_utils as hcp
    if len(parcellation_string) > 1:
        nparcs = int(parcellation_string[1:])
    if parcellation_string[0]=='S':      
        parcellation = Schaefer(nparcs)
    elif parcellation_string[0]=='K':
        parcellation = kmeans(nparcs)
    elif parcellation_string[0]=='M':
        parcellation = hcp.mmp.map_all[hcp.struct.cortex]
    elif parcellation_string[0]=='R':
        from get_gdistances import get_searchlights
        parcellation = get_searchlights(sub='102311',surface='midthickness',radius_mm=15)
    elif parcellation_string[0]=='I':
        parcellation = get_individualized_parcellation(nparcs,subjects)
    return parcellation

def parcellation_string_to_parcmatrix(parcellation_string):
    #Inputs: parcellation_string: 'S300' for Schaefer 300, 'K400' for kmeans 400, 'M' for HCP multimodal parcellation
    #Returns parcellation matrix (nparcs,nvertices)
    import hcp_utils as hcp
    if len(parcellation_string) > 1:
        nparcs = int(parcellation_string[1:])
    if parcellation_string[0]=='S':      
        matrix = Schaefer_matrix(nparcs).astype(bool)
    elif parcellation_string[0]=='K':
        matrix = kmeans_matrix(nparcs).astype(bool)
    elif parcellation_string[0]=='M':
        matrix = parc_char_matrix(hcp.mmp.map_all[hcp.struct.cortex])[1].astype(bool)
    elif parcellation_string[0]=='R':
        matrix = np.eye(59412,dtype=bool)
    nonempty_parcels = np.array((matrix.sum(axis=1)!=0)).squeeze()
    assert(len(nonempty_parcels)==matrix.shape[0]) #no empty parcels
    return matrix

def get_individualized_parcellation(nparcs,subjects):
    """
    Get individualized parcellations, saved from https://github.com/ThomasYeoLab/Kong2022_ArealMSHBM/
    Get the list of subjects from HCP_subject_list.txt. Then find indices of desired subjects from that list. Load the parcellation file (nvertices,1029 subjects). Remove gray matter vertices, and select the desired subjects. Finaly, check that each subject has the correct number of parcels.
    Parameters:
    ----------
    nparcs: int
        number of parcels
    subjects: list of strings
        list of subject IDs
    Returns:
    ----------
    labels: np.array
        array containing individualized parcellations. Array of shape (nsubjects,nvertices) containing parcel labels for each vertex
    """
    folder =  f"{intermediates_path}/Kong2022_ArealMSHBM"
    subject_list_file = ospath(f"{folder}/HCP_subject_list.txt")
    subject_list=list(np.loadtxt(subject_list_file,dtype='str')) #or dtype 'str'
    subject_indices = [subject_list.index(sub) for sub in subjects]
    parcellation_file = ospath(f"{folder}/Parcellations/HCP_1029sub_{nparcs}Parcels_Kong2022_gMSHBM.mat")
    import h5py
    f = h5py.File(parcellation_file,'r')
    colors = np.array(f.get('colors')).astype(int)
    lh_labels_all = np.array(f.get('lh_labels_all')).astype(int)
    rh_labels_all = np.array(f.get('rh_labels_all')).astype(int)
    labels_all = np.hstack([lh_labels_all,rh_labels_all])
    gray_mask=get_fsLR32k_mask()
    labels = labels_all[:,gray_mask]
    labels = labels[subject_indices,:]
    num_unique_labels = [len(np.unique(labels[i,:])) for i in range(len(subjects))]
    assert(all([num_unique_labels[i]==nparcs for i in range(len(subjects))])) 
    return labels #,colors


