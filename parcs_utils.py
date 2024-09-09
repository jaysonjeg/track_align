"""
Utility functions for parcs.py
"""

import numpy as np

def get_parcellation_pan(parcs_dir,parcellation_name):
    """
    Get parcellation of fsLR 32k surface from pan
    Separate all except: Mars82, Brainnetom210, Glasser360, Schaefer300, SchaeferHomotopic300, ??Craddock300??
    Parameters:
    --------
    parcellation_name: str
        Name of parcellation, e.g. 'Brodmann78'
    Returns:
    --------
    array: np.ndarray
        Parcellation of fsLR 32k surface with 59,412 vertices
    """
    filename_left = f"{parcs_dir}/pan/fsLR_32k_{parcellation_name}-lh.txt"
    array_left = np.loadtxt(filename_left).astype(int)
    filename_right = f"{parcs_dir}/pan/fsLR_32k_{parcellation_name}-rh.txt"
    array_right = np.loadtxt(filename_right).astype(int)
    array = np.hstack([array_left,array_right])
    print(f"{parcellation_name}: n {len(np.unique(array))}, L {len(np.unique(array_left))} unique, max {array_left.max()}, R {len(np.unique(array_right))} unique, max {array_right.max()}")
    return array

def get_parcellation_geom(parcs_dir,nparcels):
    """
    Get parcellation of geometric fsLR 32k surface from pan
    Parameters:
    --------
    nparcels: int
    Returns:
    --------
    array: np.ndarray
        Parcellation of fsLR 32k surface
    """
    filename_left = f"{parcs_dir}/pan/fsLR_32k_midthickness-lh_geometric_num_parcels={nparcels}.txt"
    array_left = np.loadtxt(filename_left).astype(int)
    #print(f"Geom {nparcels}: L {len(np.unique(array_left))} unique, max {array_left.max()}")
    filename_right = f"{parcs_dir}/pan/fsLR_32k_midthickness-rh_geometric_num_parcels={nparcels}.txt"
    array_right = np.loadtxt(filename_right).astype(int)
    #print(f"Geom {nparcels}: R {len(np.unique(array_right))} unique, max {array_right.max()}")
          
    array_right = array_right + array_left.max() + 1
    print(f"Geom {nparcels}: R {len(np.unique(array_right))} unique, max {array_right.max()}")

    array = np.hstack([array_left,array_right])
    #print(f"Geom {nparcels}: L {len(np.unique(array_left))} unique, max {array_left.max()}, R {len(np.unique(array_right))} unique, max {array_right.max()}")
    return array

def get_parcellation_EA(parcs_dir,name):
    """
    Get equal area parcellation
    Parameters:
    --------
    name: str
        Name of parcellation, e.g. 'EqualAreaPCA_150Parcels_2-5-5-3'
    Returns:
    --------
    array: np.ndarray
        Parcellation of fsLR 32k surface
    """
    filename_left = f"{parcs_dir}/equal_area/{name}_lh.txt"
    array_left = np.loadtxt(filename_left).astype(int)
    #print(f"{name}: L {len(np.unique(array_left))} unique, max {array_left.max()}")

    filename_right = f"{parcs_dir}/equal_area/{name}_rh.txt"
    array_right = np.loadtxt(filename_right).astype(int)
    #print(f"{name}: R {len(np.unique(array_right))} unique, max {array_right.max()}")

    array_right = array_right + array_left.max() + 1
    print(f"{name}: R {len(np.unique(array_right))} unique, max {array_right.max()}")

    array = np.hstack([array_left,array_right])
    #print(f"{name}: L {len(np.unique(array_left))} unique, max {array_left.max()}, R {len(np.unique(array_right))} unique, max {array_right.max()}")
    return array

def homogeneity_meanFC(data):
    """
    Given fMRI data for vertices in a single parcel, compute parcel homogeneity using mean FC among all vertex pairs
    Parameters:
    ----------
    data: np.ndarray
        shape (timepoints,vertices)
    Returns:
    ----------
    mean_correlation: float
        Mean correlation among all vertex pairs in the parcel
    """
    if data.shape[1]==1: #only 1 vertex in the parcel
        return 1
    else:
        correlations = np.corrcoef(data.T)
        np.fill_diagonal(correlations,0)
        return correlations.mean()

def homogeneity_meanFC_min_distance(data, min_distance=0, gdists=None):
    """
    Compute parcel homogeneity using mean FC among vertex pairs, for all vertex pairs separated by a minimum geodesic distance
    Parameters:
    ----------
    data: np.ndarray
        shape (timepoints,vertices)
    min_distance: float
        Distance cutoff value (mm)
    gdists: np.ndarray
        shape (vertices,vertices), containing geodesic distances
    Returns:
    ----------
    mean_correlation: float
        Mean correlation among vertex pairs
    """

    if data.shape[1]==1: #only 1 vertex in the parcel
        return 1
    else:
        correlations = np.corrcoef(data.T)
        np.fill_diagonal(correlations,0)
        valid = (gdists > min_distance)
        return correlations[valid].mean()

def homogeneity_meanFC_interp(data,gdists=None):
    """
    TO DO
    """
    correlations = np.corrcoef(data.T)
    np.fill_diagonal(correlations,0)
    from matplotlib import pyplot as plt
    dists = gdists.ravel()
    corrs = correlations.ravel()
    valid = (dists>0)
    dists = dists[valid]
    corrs = corrs[valid]
    fig,ax=plt.subplots()
    ax.scatter(dists,corrs,1,alpha=0.05,color='k')
    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('Correlation')
    fig.tight_layout()
    plt.show(block=False)
    assert(0)

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def expfit(x,y):
    """
    Fit an exponential curve. Return expfit parameters.
    Ignore OptimizeWarning
    """
    from scipy.optimize import curve_fit
    expfit_params, _ = curve_fit(exp_func, x, y,p0=[1.5,0.5,0.1],bounds=((0.5,0.001,-0.1),(2.5,1.0,0.5)))
    return expfit_params

def homogeneity_expfit_decay(data,gdists=None):
    """
    Compute parcel homogeneity by fitting exponential to the plot of correlation against distance, and obtaining the decay rate
    """
    correlations = np.corrcoef(data.T)
    np.fill_diagonal(correlations,0)
    expfit_params = expfit(gdists.ravel(),correlations.ravel())
    return expfit_params[1] #decay rate

def allparcs(data, parc_labels, parcel_function, *args, **kwargs):
    """
    Given some brain data, compute a function that returns a single value for each parcel, and return the parcel-specific values. For example, use this to find parcel homogeneity of each parcel with allparcs(my_data, my_parc_labels, homo_meanFC)
    Parameters:
    ----------
    data: np.ndarray
        shape (timepoints,vertices)
    parc_labels: np.ndarray
        shape (vertices), containing parcel labels: 0,1,2...
    parcel_function: any function handle that maps from brain data to a single value, e.g. homo_meanFC, homoe_meanFC_min_distance, etc
    *args: extra arguments passed to homogeneity_function
    **kwargs: keywords arguments passed to parcel_function
        kwargs can optionally include an item for inter-vertex geodesic distances for each parcel. In this case, the key is 'gdists', and the value is a list (nparcels) containing geodesic distance matrices (nvertices,nvertices)
    Returns:
    ----------
    parcel_values: np.ndarray
        Parcel values for each parcel
    """
    unique_labels = np.unique(parc_labels)
    nparcs = len(unique_labels)

    parcel_values = np.zeros(nparcs)
    for nparc in range(nparcs):
        kwargs_parcel = {**kwargs} #copy the dictionary of keyword arguments
        if 'gdists' in kwargs.keys():
            kwargs_parcel['gdists'] = kwargs['gdists'][nparc] #replace the value for 'gdists' with the parcel-specific geodesic distance matrix
        data_singleparc = data[:, parc_labels == unique_labels[nparc]]
        parcel_values[nparc] = parcel_function(data_singleparc, *args, **kwargs_parcel)
    return parcel_values

def dlabel_filepath_to_array(filepath,mask):
    """
    Given a filepath to cifti dlabel.nii, return a numpy array of the data contained within
    """
    import nibabel as nib
    """
    if type(filepath)==nib.nifti1.NiftiImage:
        x=filepath
    else:
        x=nib.load(filepath)
    """
    x=nib.load(filepath)
    return np.array(x.get_fdata()).squeeze()[mask].astype(int)


def atlas_vol2surf(vol_atlas_path,mask):
    from neuromaps import transforms
    vol_atlas_surf_gifti = transforms.mni152_to_fslr(vol_atlas_path, '32k',method='nearest')
    vol_atlas_surf_gifti_left = vol_atlas_surf_gifti[0].darrays[0].data
    vol_atlas_surf_gifti_right = vol_atlas_surf_gifti[1].darrays[0].data
    temp=np.hstack([vol_atlas_surf_gifti_left,vol_atlas_surf_gifti_right])
    vol_atlas_surf=temp[mask].astype(int)
    return vol_atlas_surf