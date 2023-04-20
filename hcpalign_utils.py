"""
Contains all the utility functions
"""

import numpy as np, pandas as pd
import os, pickle
import hcp_utils as hcp
import nibabel as nib
nib.imageglobals.logger.level = 40  #suppress pixdim error msg
from nilearn import signal
from pathlib import Path
from datetime import datetime
from scipy import sparse, stats
import warnings
from concurrent.futures import ThreadPoolExecutor as TPE 
from joblib import Parallel,delayed
from Connectome_Spatial_Smoothing import CSS as css

###

import socket
hostname=socket.gethostname()
if hostname=='DESKTOP-EGSQF3A':
    #home pc
    hcp_folder='/mnt/d/FORSTORAGE/Data/HCP_S1200'
    intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/intermediates'
    results_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/results'
else:
    #service workbench
    hcp_folder='/home/ec2-user/hcp/HCP_1200'
    intermediates_path='/home/ec2-user/studies/files0/intermediates'
    results_path='/home/ec2-user/studies/files0/results'

all_top_block_labels=0 #for hcpalign.py using top most connected blocks for DA

#global parameters for HCP dataset

movies=['MOVIE1_7T_AP','MOVIE2_7T_PA','MOVIE3_7T_PA','MOVIE4_7T_AP']
rests=['REST1_7T_PA','REST2_7T_AP','REST3_7T_PA','REST4_7T_AP']
tasks=['WM','GAMBLING','RELATIONAL','MOTOR','EMOTION','LANGUAGE','SOCIAL']
subs=['100610','102311','102816','104416','105923','108323','109123','111312','111514','114823','115017','115825','116726','118225','125525']

subs=list(np.loadtxt('included_subs_minus3.csv',dtype='str')) #made from findpts.py

logical2str={True:'T',False:'F'}
MSMlogical2str={True:'_MSMAll',False:''}

#start/end times(s)/volumes for each video and each rest period
movieVidTimes=[
[[20,264],[285,505],[526,714],[734,798],[818,901]],
[[20,247],[267,526],[545,795],[815,898]],
[[20,200],[221,405],[425,629],[650,792],[812,895]],
[[20,253],[272,502],[522,778],[798,881]]]

#which contrasts to get from each task
allowed_labels_dict_all={'MOTOR':range(1,5),'WM':range(4),'EMOTION':range(2),'GAMBLING':range(2),'LANGUAGE':range(2),'RELATIONAL':range(2),'SOCIAL':range(2),} 
allowed_labels_dict_visual={'MOTOR':range(0),'WM':range(4,8),'EMOTION':range(2),'GAMBLING':range(0),'LANGUAGE':range(0),'RELATIONAL':range(0),'SOCIAL':range(0),}
allowed_labels_dict_motor={'MOTOR':range(1,5),'WM':range(0),'EMOTION':range(0),'GAMBLING':range(0),'LANGUAGE':range(0),'RELATIONAL':range(0),'SOCIAL':range(0),}
allowed_labels_dict = allowed_labels_dict_all

def ospath(x):
    """
    If file path doesn't match operating system, change it e.g.
    "D:\\FORSTORAGE\\Data\\HCP_S1200"
    to Ubuntu path
    '/mnt/d/FORSTORAGE/Data/HCP_S1200'
    """
    if hostname=='DESKTOP-EGSQF3A':
        if os.name=='nt' and x[0]=='/':
            return '{}:\\{}'.format(x[5].upper(),x[7:].replace('/','\\'))
        elif os.name=='posix' and x[0]!='/':
            return '/mnt/{}/{}'.format(x[0].lower(),x[3:].replace('\\','/'))
        else:
            return x
    else: #Service Workbench, posix
        return x.replace('\\','/')

def mkdir(folderpath):
    #make the folder if it doesn't exist
    folderpath=ospath(folderpath)
    if not(os.path.isdir(folderpath)):
        os.mkdir(folderpath)

class cprint():
    """
    Class to write 'print' outputs to console and to a given textfile
    """
    def __init__(self,resultsfile):
        self.resultsfile = resultsfile
    def print(self,*args,**kwargs):    
        temp=sys.stdout 
        print(*args,**kwargs)
        sys.stdout=self.resultsfile #assign console output to a text file
        print(*args,**kwargs)
        sys.stdout=temp #set stdout back to console output

def get_filenames(func_type,func_nruns):
    if func_type=='movie':
        filenames = movies[0:func_nruns]
    elif func_type=='rest':
        filenames = rests[0:func_nruns]
    return filenames

def getfilepath(filename,ts_type,sub,MSMAll=False,cleaned=True):
    MSMString=MSMlogical2str[MSMAll]
    if cleaned: cleanString='_hp2000_clean'
    else: cleanString=''     
    if ts_type=='movie':
        return ospath(f'{hcp_folder}/{sub}/MNINonLinear/Results/tfMRI_{filename}/tfMRI_{filename}_Atlas{MSMString}{cleanString}.dtseries.nii')
    elif ts_type=='rest':
        return ospath(f'{hcp_folder}/{sub}/MNINonLinear/Results/rfMRI_{filename}/rfMRI_{filename}_Atlas{MSMString}{cleanString}.dtseries.nii')

def get_timeseries(sub,ts_type,filename,MSMAll,dtype,vertices=slice(0,59412)):
    filepath=getfilepath(filename,ts_type,sub,MSMAll)
    return get(filepath)[:,vertices].astype(dtype)

def get_timeseries_cachepath(sub,ts_type,filename,MSMAll,dtype):
    MSMString=MSMlogical2str[MSMAll]
    return f'{intermediates_path}/hcp_timeseries/{sub}_{filename}{MSMString}'

def get_all_timeseries_sub(sub,ts_type,filenames,MSMAll,ts_fwhm,ts_clean):
    """
    ts_type: 'movie' or 'rest'
    filenames e.g. ['MOVIE1_7T_AP','MOVIE2_7T_PA','MOVIE3_7T_PA','MOVIE4_7T_AP'] 
    """    
    dtype=np.float16 #np.float32 or np.float32
    nalign_sub=[from_cache(get_timeseries_cachepath,get_timeseries,sub,ts_type,filename,MSMAll,dtype,load=True,save=True) for filename in filenames]   
    if ts_type=='movie':   
        movieVidVols = [movieVolumeSelect(v,10,10) for v in movieVidTimes] #get list of all movie volumes to be included  
        nalign_sub = [nalign_sub[i][movieVidVols[i],:] for i in range(len(nalign_sub))]
    #Following only relevant if X_clean=True
    clean_each_movie_separately=True
    standardize,detrend,low_pass,high_pass,t_r=True,True,None,None,1.0
    ts_preproc=make_preproc(ts_fwhm,ts_clean,standardize,detrend,low_pass,high_pass,t_r)  
    if clean_each_movie_separately:
        temp=np.vstack([ts_preproc(i) for i in nalign_sub])
    else:
        temp=ts_preproc(np.vstack(nalign_sub))
    return temp.astype(dtype)


def get_tasks_cachepath(tasks,sub,MSMAll=False):
    MSMtypestring = MSMlogical2str[MSMAll]  
    task_string=''.join([i[0] for i in tasks])
    return f'{intermediates_path}/hcp_taskcontrasts/{sub}{MSMtypestring}_{task_string}'

def gettasks(tasks,sub,vertices=slice(0,59412),MSMAll=False):
    #Get 3T task analysis contrast map data
    MSMString=MSMlogical2str[MSMAll]
    task_files=[ospath(f'{hcp_folder}/{sub}/MNINonLinear/Results/tfMRI_{task}/tfMRI_{task}_hp200_s2_level2{MSMString}.feat/{sub}_tfMRI_{task}_level2_hp200_s2{MSMString}.dscalar.nii') for task in tasks]
    task_data=[get(task_files[i])[allowed_labels_dict[tasks[i]],vertices] for i in range(len(tasks))]
    return np.vstack(task_data).astype(np.float32)

def gettaskcontrastfiles(tasks,subs):
    #Get filenames for 3T task analysis list of contrasts. shape(subs,tasks)
    ftaskcontrasts=[[ospath('{}/{}/MNINonLinear/Results/tfMRI_{}/tfMRI_{}_hp200_s2_level2.feat/Contrasts.txt'.format(hcp_folder,sub,task,task)) for task in tasks] for sub in subs]
    ftaskcontrasts=pd.DataFrame(data=ftaskcontrasts,index=subs,columns=tasks)
    return ftaskcontrasts

def get_tasklabels_cachepath(tasks,sub):
    task_string=''.join([i[0] for i in tasks])
    return f'{intermediates_path}/hcp_tasklabels/labels_{sub}_{task_string}'

def gettasklabels(tasks,sub):
    #Get 3T task analysis contrast labels
    contrast_files=[ospath(f'{hcp_folder}/{sub}/MNINonLinear/Results/tfMRI_{task}/tfMRI_{task}_hp200_s2_level2.feat/Contrasts.txt') for task in tasks]
    labels = [pd.read_csv(contrast_files[i],header=None).iloc[allowed_labels_dict[tasks[i]]] for i in range(len(tasks))]
    return np.vstack(labels).squeeze()

def get_thickness(sub):
    #Get cortical thickness. Returns an array (59412,)
    filename=f'{sub}.thickness.32k_fs_LR.dscalar.nii'
    filepath=ospath(f'{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{filename}')
    return nib.load(filepath).get_fdata().squeeze()


def get (filename,vertices=slice(0,None)):
    """
    filename corresponds to cifti-2 image
    Returns surface data as numpy array
    Default all vertices
    """
    return nib.load(filename).get_fdata()[:,vertices]
    
def pairwise_correlation_for_all_tasks(t):
    """
    t is a list(nsub) containing task maps (ncontrasts x nvertices)
    Outputs correlation coefficients for each task contrast, for each subject-pair
    """
    from itertools import combinations
    n_subs=len(t)
    allsubjectpairs=[i for i in combinations(range(n_subs),2)]
    nsubjectpairs=len(allsubjectpairs)
    ncontrasts=t[0].shape[0]
    corr_scores = np.zeros((nsubjectpairs,ncontrasts),dtype=np.float32) 
    for i in range(nsubjectpairs):
        subjectpair=allsubjectpairs[i]
        for ncontrast in range(ncontrasts):
            corr_scores[i,ncontrast] = np.corrcoef(t[subjectpair[0]][ncontrast,:] , t[subjectpair[1]][ncontrast,:])[0,1]
    return corr_scores

def func_acf(x, length=20): 
    #autocorrelation function of a vector
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])

def acf_array(array, length=20): 
    #autocorrelation for each column in square array separately
    nvertices=array.shape[1]
    acfs=np.zeros((length,nvertices),dtype=float) #store autocorrelation for each lag
    acds=np.zeros((nvertices),float) #store autocorrelation distance (first lag with autocorr<0
    for i in range(nvertices):
        acf=func_acf(array[:,i],length)
        acfs[:,i]=acf
        temp=np.where(acf<0)[0]
        if temp.size>0:
            acds[i]=temp[0]
        else:
            acds[i]=length
    return acfs,acds

def union(x,y):
    return (x+y).getnnz()
def intersect(x,y):
    return (x.multiply(y)).getnnz()
def jaccard(X,Y):
    assert(X.dtype==bool)
    """
    val=union(X,Y)
    if val: return intersect(X,Y)/union(X,Y)
    else: return np.nan
    """
    val=union(X,Y)
    if val: return (X.getnnz()+Y.getnnz()-val)/val
    else: return np.nan
def jaccard_binomtest(x,y):
    #returns p-value for overlap of x and y
    binom_n=x.shape[0]*x.shape[1]
    binom_p=(x.getnnz()*y.getnnz())/binom_n**2
    binom_k=intersect(x,y) #value being tested
    result_binom=stats.binom_test(binom_k,binom_n,binom_p,alternative='greater')
    return result_binom


def rowcorr(sp1, sp2):
    '''
    From Sina Mansour email
    Calculates the correlation of each row between two sparse matrices.
    Note: both input matrices (sp1, sp2) need to be scipy.sparse.csr_matrix
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.divide(
            (sp1.multiply(sp2).mean(axis=0) - np.multiply(sp1.mean(axis=0), sp2.mean(axis=0))),
            np.sqrt(
                np.multiply(
                    (sp1.power(2).mean(axis=0)) - np.power(sp1.mean(axis=0), 2),
                    (sp2.power(2).mean(axis=0)) - np.power(sp2.mean(axis=0), 2)
                )
            )
        )

def rowcorr_nonsparse(sp1,sp2):
    """Non-sparse array version of above"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.divide(
            (np.mean(np.multiply(sp1,sp2),0) - np.multiply(np.mean(sp1,0),np.mean(sp2,0))),
            np.sqrt(
                np.multiply(
                    np.mean(np.power(sp1,2),0) - np.power(np.mean(sp1,0),2),
                    np.mean(np.power(sp2,2),0) - np.power(np.mean(sp2,0),2)
                    )
                )
            )

def density(sparsearray):
    return sparsearray.getnnz()/(sparsearray.shape[0]*sparsearray.shape[1])

def from_cache(func_filepath,func,*args,load=True,save=True,**kwargs):
    """
    Generate filepath using func_filepath(*args,**kwargs). Check if filepath already exists. If it doesn't exist, generate required value or array using func(*args,**kwargs) and save this in filepath. 
    Optional arguments load and save can be provided after **kwargs
    """
    filepath=func_filepath(*args,**kwargs)   
    if load and os.path.exists(ospath(filepath)):
        values = pickle.load(open(ospath(filepath), "rb" ))
    else:
        values = func(*args,**kwargs)
        if save:
            pickle.dump(values,open(ospath(filepath),"wb"))
    return values

def get_func_type(string):
    if 'movie' in string: return 'movie'
    elif 'rest' in string: return 'rest'
    else: assert(0)

def get_FC_filepath(
    sub,
    align_with,
    vertices,
    MSMAll,
    align_clean,
    align_fwhm,
    targets_parcellation,
    targets_nparcs,
    filenames,
    FC_type): 

    align_clean_string = logical2str[align_clean]    
    MSMtypestring = MSMlogical2str[MSMAll]    
    return f'{intermediates_path}/functional_connectivity/{get_func_type(align_with)}{MSMtypestring}_{len(filenames)}runs_{align_clean_string}_fwhm{align_fwhm}_{targets_parcellation}{targets_nparcs}_{FC_type}_sub{sub}.p'

def get_FC(
    sub,
    align_with,
    vertices,
    MSMAll,
    align_clean,
    align_fwhm,
    targets_parcellation,
    targets_nparcs,
    filenames,
    FC_type):   
    """
    FC_type is 'pxn' or pxp'
    """
    _,parc_matrix=get_parcellation(targets_parcellation,targets_nparcs)
    na = get_all_timeseries_sub(sub,get_func_type(align_with),filenames,MSMAll,align_fwhm,align_clean)
    nap=(na@parc_matrix.T) #ntimepoints * nparcs
    if FC_type=='pxn':
        return corr4(nap,na,b_blocksize=parc_matrix.shape[0])
    elif FC_type=='pxp':
        return corr4(nap,nap,b_blocksize=parc_matrix.shape[0])

def get_all_FC(subs,args):
    return Parallel(n_jobs=-1,prefer='threads')(delayed(from_cache)(get_FC_filepath, get_FC, *(sub, *args), load=True, save=True) for sub in subs)


def get_hrc_filepath(sub,tckfile,hcp_path,tract_path,sift2,threshold,MSMAll):
    tractsub_path=f'{tract_path}/{sub}' #can put 'y' or 'z' before {sub}
    tract_file=f'{tractsub_path}/{tckfile}'              
    MSMtypestring={True:'_MSMAll',False:''}[MSMAll]
    if sift2:
        cache_file=tract_file[:-4]+'_sift2act.p'
    else:
        cache_file=tract_file[:-4]+'.p'
    cache_file=cache_file[:-2] + MSMtypestring + '.p'
    return cache_file
    
def get_hrc(sub,tckfile,hcp_path,tract_path,sift2,threshold,MSMAll):
    print(f'Calculating hrc (not found in cache): {sub} {tckfile}')
    hcpsub_path=f'{hcp_path}/{sub}'
    tractsub_path=f'{tract_path}/{sub}' #can put 'y' or 'z' before {sub}
    tract_file=f'{tractsub_path}/{tckfile}'               
    MSMtypestring={True:'_MSMAll',False:''}[MSMAll]
    if sift2:
        if tckfile[0]=='v':
            weights_file=ospath(f'{tractsub_path}/volumetric_probabilistic_sift_weights_5M.txt')
            weights=np.loadtxt(weights_file)
        elif tckfile[-7:-4]=='end':
            weights_file=ospath(f'{tract_file[:-8]}_sift2act_weights.txt')
            weights=np.loadtxt(weights_file,skiprows=1)  
        else:
            weights_file=ospath(f'{tract_file[:-4]}_sift2act_weights.txt')
            weights=np.loadtxt(weights_file,skiprows=1)  
    else:
        weights=None
    left_MNI_surface_file=f'{hcpsub_path}/MNINonLinear/fsaverage_LR32k/{sub}.L.white{MSMtypestring}.32k_fs_LR.surf.gii'
    right_MNI_surface_file=f'{hcpsub_path}/MNINonLinear/fsaverage_LR32k/{sub}.R.white{MSMtypestring}.32k_fs_LR.surf.gii'
    warp_file=f'{hcpsub_path}/MNINonLinear/xfms/standard2acpc_dc.nii.gz'   
    hrs = css.map_high_resolution_structural_connectivity(ospath(tract_file), ospath(left_MNI_surface_file), ospath(right_MNI_surface_file), warp_file=ospath(warp_file),weights=weights,threshold=threshold) 
    return hrs
     
def get_highres_connectomes(
    c,
    subs,
    tckfile,
    tract_path="/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/HCP_tractography",
    threshold=2,
    sift2=False,
    cache_loadhrc=True,
    cache_savehrc=True,
    MSMAll=False,
    n_jobs=-1,
    prefer='processes'):
    """
    Args:
        c: clock from hcpalign_utils.clock()
        tckfile: 'tracks_5M_sift1M.tck' 'tracks_5M_50k.tck', 'tracks_5M_sift1M_200k.tck'
        threshold: threshold for css.map_high_resolution_structural_connectivity. Default 2
        cache_loadhrc: load cached high-res connectomes if they are available
        cache_savehrc: to save high-res connectomes if they're not available
        hrc_type: 'mrtrix' or 'fsl'
        prefer: threads better when cached files already exist, processes better when they don't

    Returns:
        hr: list of high-res connectomes as sparse arrays
    """    
    if tckfile[0]=='v': #sina's connectomes
        tract_path=f'{intermediates_path}/diff'
        tckfile='volumetric_probabilistic_track_endpoints_5M.tck'   
    elif tckfile[-7:-4]=='end': #my ec2 connectomes
        tract_path=f'{intermediates_path}/diff2'
    func = lambda sub: from_cache(get_hrc_filepath,get_hrc,sub,tckfile,hcp_folder,tract_path,sift2,threshold,MSMAll,load=cache_loadhrc,save=cache_savehrc)
    if n_jobs==1:
        return [func(sub) for sub in subs]
    else:
        return Parallel(n_jobs=n_jobs,prefer=prefer)(delayed(func)(sub) for sub in subs)
        
def smooth_highres_connectomes(hr,smoother):
    """
    hr is list of sparse connectomes from get_highres_connectomes
    smoother is a smoothing kernel as a sparse array
    """   
    from Connectome_Spatial_Smoothing import CSS as css   
    def func(hrs): return css.smooth_high_resolution_connectome(hrs,smoother)
    hr=Parallel(n_jobs=-1,prefer='threads')(delayed(func)(i) for i in hr)   
    return hr 

def _make_diagonals_zero(sparse_array):
    #for a single sparse array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   
        sparse_array.setdiag(0)
        sparse_array.eliminate_zeros()

def make_diagonals_zero(list_of_sparse_arrays,concurrent=False):    
    if concurrent:
        with TPE() as executor:
            list(executor.map(_make_diagonals_zero,list_of_sparse_arrays))
    else:
        for sparse_array in list_of_sparse_arrays:
            _make_diagonals_zero(sparse_array)


def indices_of_largest_values_in_array(array,n):
    if type(array)==sparse.csr_matrix:
        h=sparse.triu(array.tocoo(),0) #only take upper triangular elements (assuming symmetric)
        if n > h.getnnz():
            row_inds=h.row
            col_inds=h.col
        else:
            inds=np.argpartition(h.data,-n)[-n:]
            inds2=np.argsort(h.data[inds])[::-1] #indices of included data from largest to smallest
            inds3=inds[inds2] #order the partition from largest to smallest
            row_inds=h.row[inds3]
            col_inds=h.col[inds3]
        return np.vstack([row_inds,col_inds])
    elif type(array)==np.ndarray:
        assert(len(array) >= n)
        #return np.argpartition(array,-n)[-n:]
        
        inds=np.argpartition(array,-n)[-n:]
        inds2=np.argsort(array[inds])[::-1]
        inds3=inds[inds2]
        return inds3

def vertexmap_59kto64k(hemi='both'):
    """
    List of 59k cortical vertices in fsLR32k, with their mapping onto 64k cortex mesh
    hemi='both','L','R'
    """
    grayl=hcp.vertex_info.grayl
    grayr=hcp.vertex_info.grayr
    grayr_for_appending=hcp.vertex_info.grayr+hcp.vertex_info.num_meshl
    grayboth=np.hstack((grayl,grayr_for_appending))
    
    if hemi=='both': return grayboth
    elif hemi=='L': return grayl
    elif hemi=='R': return grayr

def vertexmap_64kto59k(hemi='both'):
    """
    List of 64k cortex mesh vertices, with their mapping onto 59k vertices in fsLR32k. Vertices not present in 59k version are given value 0
    hemi='both','L','R'
    """
    gray=vertexmap_59kto64k(hemi=hemi)
    if hemi=='both':
        num_mesh_64k = hcp.vertex_info.num_meshl+hcp.vertex_info.num_meshr
    elif hemi=='L':
        num_mesh_64k = hcp.vertex_info.num_meshl
    elif hemi=='R':
        num_mesh_64k = hcp.vertex_info.num_meshr
    temp=np.zeros(num_mesh_64k,dtype=int)
    for index,value in enumerate(gray):
        temp[value]=index
    return temp


def cortex_64kto59k(arr):
    """
    Opposite of hcp_utils.cortex_data
    Map functional/scalar/parcellation data on 64984-vertex cortex mesh onto 59412-vertex cortex mesh 
    """
    gray=vertexmap_59kto64k()
    return arr[gray]

def cortex_64kto59k_for_triangles(triangles,hemi='both'):
    """
    triangles: array (n,3) list of triangles in a 64984-vertex mesh
    Returns an abridged version of 'triangles' within 59412-vertex mesh
    """
    #first remove unnecessary vertices
    gray=vertexmap_59kto64k(hemi=hemi)
    temp=np.isin(triangles,gray)
    temp2=np.all(temp,axis=1)
    triangles_v2 = triangles[temp2,:]

    #then renumber vertices so they are 0 to 59411
    triangles_v3 = vertexmap_64kto59k(hemi=hemi)[triangles_v2]
    
    return triangles_v3

def vertex_59kto64k(vertices):
    """Given a list of vertices in 59k, convert each index to the corresponding index in 64k cortex mesh
    """
    gray=vertexmap_59kto64k()
    return gray[vertices]

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def surfgeodistances(source_vertices_59k, surf=hcp.mesh.midthickness):
    """
    Given geodesic surface distances
    Inputs: source_vertices_59k - array(n,) of source vertices in 59k cortex space
            surf - surface vertices/triangles tuple(2,) in 59k cortex space
    Output: array(59k,) of distances from source vertices
    """
    import gdist
    source_vertices_64k=vertex_59kto64k(source_vertices_59k).astype('int32')
    distances_64k=gdist.compute_gdist(surf[0].astype('float64'),surf[1],source_vertices_64k)
    distances_59k=cortex_64kto59k(distances_64k)
    return distances_59k

def surfgeoroi(source_vertices_59k,limit=0,surf=hcp.mesh.midthickness):
    """
    Like wb_command -surface-geodesic-rois
    Output list of all vertices within limit mm of source vertices
    If limit==0, then just output source vertices
    If limit==0, return entire cortex
    """
    if limit==0: 
        return makesurfmap(source_vertices_59k)
    elif limit==np.inf:
        return np.ones([59412])
    else:
        return (surfgeodistances(np.array(source_vertices_59k),surf) < limit).astype('float')

def load_geodesic_distances(surface='midthickness',fwhm=2,epsilon=0.01,sub='100610'):
    return sparse.load_npz(ospath(f'/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/intermediates/geodesic_distances/gdist_{surface}_fwhm{fwhm}_eps{epsilon}_sub{sub}.npz'))

def _xy_distances_to_2Dgaussian(xdists,ydists,fwhm_x,fwhm_y):
    from Connectome_Spatial_Smoothing import CSS as css
    sigma_x=css._fwhm2sigma(fwhm_x)
    sigma_y=css._fwhm2sigma(fwhm_y)
    xgauss = -(xdists.power(2) / (2 * (sigma_x ** 2)))
    ygauss = -(ydists.power(2) / (2 * (sigma_y ** 2)))
    gaussian = xgauss + ygauss
    np.exp(gaussian.data, out=gaussian.data)
    gaussian += sparse.eye(gaussian.shape[0], dtype=gaussian.dtype).tocsr()
    from sklearn.preprocessing import normalize
    gaussian=normalize(gaussian,norm='l1',axis=0)
    return gaussian.astype(np.float32)

def _local_distances_to_gaussian(local_distance, sigma):
    """like the function css._local_distances_to_smoothing_coefficients, but returns the gaussian sparse array"""
    gaussian = -(local_distance.power(2) / (2 * (sigma ** 2)))
    np.exp(gaussian.data, out=gaussian.data)
    gaussian += sparse.eye(gaussian.shape[0], dtype=gaussian.dtype).tocsr()
    return gaussian

def make_smoother_100610(fwhm):
    """
    Given fwhm value, returns a surface smoothing function which operates on arrays(n,ncorticalvertices=59412). Based on subject 100610's mesh
    """
    if fwhm==0:
        return lambda x: x
    else:
        skernel=sparse.load_npz(ospath(f'{intermediates_path}/smoothers/100610_{fwhm}_0.01.npz'))
        return lambda x: skernel.dot(x.T).T

def standardize(array):
    #standardize to 0 mean unit variance (of entire array)
    array -= np.mean(array)
    array /= np.std(array)   
    return array

def make_preproc(fwhm,ToClean,standardize,detrend,low_pass,high_pass,t_r):
    if ToClean:
        cleaner=lambda x: signal.clean(x,standardize=standardize,detrend=detrend,low_pass=low_pass, high_pass=high_pass, t_r=t_r, ensure_finite=True).astype(np.float32)
    else:
        cleaner=lambda x: x
    smoother=make_smoother_100610(fwhm)     
    return lambda x: cleaner(smoother(x))

def surfsmooth(x,left_surf_file,right_surf_file,fwhm=3,epsilon=0.01):
    """
    surf_files are surf.gii
    x is data array(ncorticalvertices=59412,n) for any n
    """
    from Connectome_Spatial_Smoothing import CSS as css
    skernel = css.compute_smoothing_kernel(left_surf_file, right_surf_file, fwhm, epsilon)
    return skernel.dot(x)

def makesurfmap(voxels,totalvoxels=59412):
    """
    Make cifti2 style surface array with 1s at the designated voxels and 0 elsewhere
    Operates on hcp.struct.cortex surface alone
    """
    output=np.zeros([totalvoxels])
    if type(voxels)==int:
        output[voxels]=1
    else: #if voxels is list or 1D array
        for voxel in voxels:
            output[voxel]=1
    return output

def Schaefer_original(nparcels):
    #get Schaefer Kong surface parcellation
    filename=ospath('/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/SchaeferParcellations/HCP/fslr32k/cifti/Schaefer2018_{}Parcels_Kong2022_17Networks_order.dlabel.nii'.format(nparcels))
    return cortex_64kto59k(get(filename).squeeze()).astype(int)
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

def get_parcellation(parcellation,nparcs,return_nonempty=True):
    """
    Returns kmeans or Schaefer parcellations with 'nparcs' parcels. 
    return_nonempty means to return parc_matrix with only rows corresponding to nonempty parcels
    """
    if parcellation=='kmeans': 
        f_matrix = kmeans_matrix
        f = kmeans
    elif parcellation=='Schaefer': 
        f = Schaefer
        f_matrix = Schaefer_matrix
    labels = f(nparcs)
    parc_matrix= f_matrix(nparcs).astype(bool)

    nonempty_parcels = np.array((parc_matrix.sum(axis=1)!=0)).squeeze()
    assert(len(nonempty_parcels)==parc_matrix.shape[0]) #no empty parcels

    return labels, parc_matrix

def parc_char_matrix(parc):
    """
    Similar to connectome-spatial-smoothing.parcellation_characteristic_matrix
    parc is parcellation e.g. Schaefer(300)
    """
    
    parcellation_matrix=np.zeros((max(parc)+1,59412))
    for i in range(len(parc)):
        value=parc[i]
        parcellation_matrix[value,i]=1
    return list(set(parc)),sparse.csr_matrix(parcellation_matrix).astype(np.float32)            

def reverse_parc_char_matrix(matrix):
    result = np.zeros(matrix.shape[1],dtype=int)
    matrix=matrix.astype(bool).toarray()
    for i in range(matrix.shape[1]):
        result[i]=np.argmax(matrix[:,i])
    return result

def replace_with_parcelmean(X,parcellation):
    #Replace each value in a functional activation map, with the mean of its parcel
    X2=np.copy(X)   
    ids=np.unique(parcellation)
    for parcel in ids:
        indices=parcellation==parcel
        
        if X.ndim==1:
            #if X is array(nvertices)
            parcelmean=np.mean(X[indices])
            X2[indices]=parcelmean
        elif X.ndim==2:
            #if X is array(nsamples,nvertices)
            parcelmeans=np.mean(X[:,indices],axis=1)
            X2[:,indices]=np.tile(parcelmeans,(indices.sum(),1)).T
        
    return X2

import fmralign
def aligner_downsample(estimator,dtype='float32'):
    """
    estimator is instance of mySurfacePairwiseAlignment
    dtype can be 'float32' or 'float16'
    If using aligner from templateAlignment, do --> aligner.estimators=[hcpalign_utils.aligner_downsample(i) for i in aligner.estimators]
    """
    
    #Hungarian, OptimalTransportAlignment, ScaledOrthogonalAlignment, RidgeAlignment
    
    fits=estimator.fit_
    pairwise_method=type(fits[0])
    for j in range(len(fits)):
        if pairwise_method==fmralign.alignment_methods.ScaledOrthogonalAlignment:
            x=fits[j].R
            y=x.astype(dtype)
            estimator.fit_[j].R=y
        elif pairwise_method==fmralign.alignment_methods.RidgeAlignment:
            pass
            # use fits[j].R.coef_, fits[j].R.intercept_ ??
        
    return estimator

        
 
class surfplot():
    """
    Plot surface functional activations. Data is array(59412,).
    p=surfplot('/mnt/d/Users/Jayson/Figures')
    p.plot(data,'Figure1')
    """
    import hcp_utils as hcp
    from pathlib import Path
    def __init__(self, figpath,mesh=hcp.mesh.midthickness,vmin=None,vmax=None,cmap='inferno',symmetric_cmap=True,plot_type='open_in_browser'):
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
        """
        if data.shape[0]<59412: #fill missing data
            ones=np.ones((59412))*(min(data)-0.5*(max(data)-min(data)))
            ones[0:data.shape[0]]=data
            data=ones
        """
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


        if self.mesh[0].shape[0] > 59412: #if using full 64,983-vertex mesh
            new_data = hcp.cortex_data(data)
        else:
            new_data = data
        
        view=plotting.view_surf(self.mesh,new_data,cmap=self.cmap,vmin=self.vmin,vmax=self.vmax,symmetric_cmap=self.symmetric_cmap)  
        if self.plot_type=='save_as_html':
            view.save_as_html(ospath('{}/{}.html'.format(self.figpath,savename)))
        elif self.plot_type=='open_in_browser':
            view.open_in_browser()
        self.vmin=None


def plot_parc(p,align_parc_matrix,data,savename=None):
    """
    Given colour data for each parcel, plot this on hcp_utils viewer
    p is an instance of class surfplot
    align_parc_matrix can be derived from function Schaefermatrix
    data is a (59412) length list or array
    """
    data=np.array(data)
    p.plot(data @ align_parc_matrix,savename=savename)
    
def plot_parc_multi(p,align_parc_matrix,strings,values):
    for string,value in zip(strings,values):
        plot_parc(p,align_parc_matrix,value,string)


def do_plot_impulse_responses(p,plot_prefix,aligner,method,lowdim_vertices):
    """
    aligner can be my_surf_pairwise_alignment or my_template_alignment
    p is surfplot instance
    plot_prefix is a string
    method can be 'pairwise' or 'template'
    lowdim_vertices can be 'true' or 'false'
    """
    verticesx=[1,2,3,4,5,6,7,8,29696+1,29696+2,29696+3,29696+4,29696+5,29696+6,29696+7,29696+8]
    for radius in [2]: #default [0,2]
        s=surfgeoroi(verticesx,radius)
        t=aligner.transform(s[None,:])[0]  
        if method=='template' and lowdim_vertices==True:
            t=aligner.pcas[1].inverse_transform(aligner.transform(s[None,:],0))      
        p.plot(s, f"{plot_prefix}_roi_source{radius}",symmetric_cmap=True)
        p.plot(np.squeeze(t),f"{plot_prefix}_roi_target{radius}",symmetric_cmap=True)



def timer(start_time):
    end_time=datetime.now()
    runtime=end_time-start_time
    return runtime.total_seconds()

class clock():
    """
    How to use
    c=hcpalign_utils.clock()
    print(c.time())
    """
    def __init__(self):
        self.start_time=datetime.now()       
    def time(self):
        end_time=datetime.now()
        runtime=end_time-self.start_time
        value='{:.1f}s'.format(runtime.total_seconds())
        return value

def now():
    now=datetime.now()
    return now.strftime("%H:%M:%S")

def datetime_for_filename():
    now=datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def sizeof_fmt(num, suffix='B'):
    #called by sizeof
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

import sys
import gc

def _sizeof(input_obj):
    #called by sizeof
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

def sizeof(input_obj):
    if type(input_obj)==list:
        return sizeof_fmt(sum([_sizeof(i) for i in input_obj]))
    else:
        return sizeof_fmt(_sizeof(input_obj))

def getloadavg():
    import psutil
    print([x / psutil.cpu_count() for x in psutil.getloadavg()])

def memused(): #By Python
    import os, psutil
    process = psutil.Process(os.getpid())
    return f'Python mem: {process.memory_info().rss/1e9:.1f} GB, PC mem: {psutil.virtual_memory()[3]/1e9:.1f}/{psutil.virtual_memory()[0]/1e9:.0f} GB'


from scipy.stats import spearmanr as sp
def corr2(a,b,corr_type='pearson'):
    #Required for corr4
    #max size is (a.shape[1]+b.shape[1])**2 * 8 / 10**9 gigabytes
    #Takes 15sec for (a.shape[1]+b.shape[1])==20300, and uses max 3 GiB
    if corr_type=='pearson':
        temp=np.corrcoef(a,b,rowvar=False)
    elif corr_type=='spearman':
        temp,_=sp(a,b)  
    return temp[0:a.shape[1],a.shape[1]:].astype(np.float32) 



def maxcorrfit(source_array,target_array,dists,max_dist,nverts=59412,typeof=1):
    """
    source_array[:,result] will have columns (vertices) which are more correlated with columns of target_array
    typeof:
    1. dists is not 'full', so it already contains info about max_dist
    2. use gdist_full, and np.where
    3. use gdist_full. sparse dist precalculated

    """
    result=np.zeros((nverts),int)
    precalc=True #better when max_dist >= 7
    if precalc:
        cutoff=len(hcp.vertex_info.grayl)
        corrL=corr3(source_array[:,0:cutoff],target_array[:,0:cutoff])
        corrR=corr3(source_array[:,cutoff:],target_array[:,cutoff:])       
    for vert_target in range(nverts):
        verts_source = dists.indices[dists.indptr[vert_target]:dists.indptr[vert_target+1]]   #sources are close to target
        if typeof==1:
            verts_source = list(verts_source) + [vert_target] # add the 'identity' vertex           
        if precalc:
            if vert_target < cutoff:
                corrs=corrL[verts_source,vert_target]
            else:
                if type(verts_source)==list:
                    verts_source_corrected = [i-cutoff for i in verts_source]
                elif type(verts_source)==np.ndarray:
                    verts_source_corrected = verts_source - cutoff
                corrs=corrR[verts_source_corrected , vert_target-cutoff]
        else:           
            target=target_array[:,vert_target]
            target=target.reshape((len(target),1))
            source=source_array[:,verts_source]
            corrs=corr3(source,target)          
        result[vert_target]=verts_source[np.argmax(corrs)]
    return result

class maxcorr():
    def __init__(self,dists,max_dist=10,typeof=1):
        self.dists=dists
        self.max_dist=max_dist
        self.typeof=typeof
    def fit(self,X,Y):
        self.fit_ = maxcorrfit(X,Y,self.dists,self.max_dist,typeof=self.typeof)
    def transform(self,X):
        return X[:,self.fit_]




def divide(n,blocksize):
    """Required for corr4. Python code to divide integers from 0 to n into blocks of length blocksize. The last block might be smaller. Returns a list of slices of integers from 0 to n"""
    if blocksize>=n: return [slice(0,n)]
    else:
        return[slice(i, min(i+blocksize, n)) for i in range(0,n,blocksize)]

def corr4(a,b,a_blocksize=np.inf,b_blocksize=np.inf,dtype=np.float32):
    """
    correlations between columns of a and columns of b. Do it in subsets of columns of a and subsets of columns of b at a time, to reduce maximum RAM usage. a_blocksize and b_blocksize determine the size of these subsets. Setting these to np.inf will use a lot of RAM but be very fast. Setting a_blocksize < a.shape[1] will be slower.
    """
    out = np.zeros((a.shape[1],b.shape[1]),dtype=dtype)   
    for a_slice in divide(a.shape[1],a_blocksize):
        for b_slice in divide(b.shape[1],b_blocksize):
            a_subblock = a[:,a_slice]
            b_subblock = b[:,b_slice]
            out[a_slice,b_slice] = corr2(a_subblock,b_subblock)               
    return out

def corr3(X,Y): #the best
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    X_centered = X - mean_X
    Y_centered = Y - mean_Y
    std_X = np.std(X, axis=0)
    std_Y = np.std(Y, axis=0)
    corr = (X_centered.T @ Y_centered) / (std_X[:,None] * std_Y[None,:])
    return corr/X.shape[0]

def corrmatch(X,Y,func):
    #Return average correlation between rows/or cols? of X and corresponding  rows/cols of Y
    temp=func(X,Y)
    return np.diag(temp)

def movieVolumeSelect(time_list,start=0, end=0):
    """
    Inputs:
    time_list: video start/end timestamps. e.g. [[2,4],[10,13]]
    Exclude first 'start' sec of video
    Include 'end' sec of volumes after each video ends 
    
    Output: list containing all movie volumes to be included
    e.g. [2,3,10,11,12] if start=end=0
    """
    temp=[list(range(x+start,y+end)) for x,y in time_list]
    return [item for sublist in temp for item in sublist]