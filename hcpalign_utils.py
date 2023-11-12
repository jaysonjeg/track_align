"""
Contains all the utility functions
"""


import numpy as np, pandas as pd
import os, pickle
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
#rests=['REST1_7T_PA','REST2_7T_AP','REST3_7T_PA','REST4_7T_AP']
rests=['REST1_LR','REST1_RL','REST2_LR','REST2_RL']
tasks=['WM','GAMBLING','RELATIONAL','MOTOR','EMOTION','LANGUAGE','SOCIAL']
all_subs=['100610','102311','102816','104416','105923','108323','109123','111312','111514','114823','115017','115825','116726','118225','125525']
all_subs=list(np.loadtxt('included_subs_minus3.csv',dtype='str')) #made from findpts.py

logical2str={True:'t',False:'f'}
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

def getvalue(df,subject,column):
    """
    Get value from dataframe df, for subject and column
    Parameters:
        df: dataframe
        subject: subject number or string
        column: column name
    Returns:
        value
    """
    if type(subject)==str:
        subject = int(subject)
    return df.loc[df['Subject']==subject,column].values[0]

def shuffle_colormap(cmap_string,upsample=500):
    """
    Given a named matplotlib colormap, upsample the indices then shuffle the indices to return a new colormap where the colors are shuffled. This has the effect that adjacent indices are likely to be mapped to different colors
    Parameters
    ----------
    cmap_string: string, name of a matplotlib colormap, e.g. 'tab20'
    upsample: int, number of indices in the new colormap 
    """

    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    cmap=mpl.colormaps[cmap_string].resampled(upsample)
    inds = np.arange(cmap.N)
    np.random.shuffle(inds)
    shuf_cols = cmap(inds)
    shuf_cmap = ListedColormap(shuf_cols)
    return shuf_cmap 

def get_hcp_behavioral_data():
    df=pd.read_csv(ospath(f'{intermediates_path}/BehavioralData.csv'))
    df.loc[df['Subject']==179548,'3T_Full_Task_fMRI']  = False #does not have MSMAll data for WM task
    return df

def get_rows_in_behavioral_data():
    """
    In the HCP dataset (behavioral data dataframe), get the row indices of subjects who have completed different MRI tasks
    """
    df = get_hcp_behavioral_data()
    cognitive_measures = ['Flanker_AgeAdj', 'CardSort_AgeAdj', 'PicSeq_AgeAdj', 'ListSort_AgeAdj', 'ProcSpeed_AgeAdj','PicVocab_AgeAdj', 'ReadEng_AgeAdj','PMAT24_A_CR','IWRD_TOT','VSPLOT_TC'] 
    rows_with_cognitive = ~df[cognitive_measures].isna().any(axis=1)
    rows_with_3T_taskfMRI = (df['3T_Full_Task_fMRI']==True)
    rows_with_3T_rsfmri = (df['3T_RS-fMRI_Count']==4)
    rows_with_7T_rsfmri = (df['7T_RS-fMRI_Count']==4)
    rows_with_7T_movie = (df['fMRI_Movie_Compl']==True)
    return cognitive_measures, rows_with_cognitive, rows_with_3T_taskfMRI, rows_with_3T_rsfmri, rows_with_7T_rsfmri, rows_with_7T_movie



def get_filenames(func_type,func_nruns):
    if func_type=='movie':
        filenames = [movies[i] for i in func_nruns]
    elif func_type=='rest':
        filenames = [rests[i] for i in func_nruns]
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


def get_all_timeseries_sub(sub,ts_type,filenames,MSMAll,ts_preproc):
    """
    Returns an array containing fMRI time series
    sub: subject ID 
    ts_type: 'movie' or 'rest'
    filenames e.g. ['MOVIE1_7T_AP','MOVIE2_7T_PA','MOVIE3_7T_PA','MOVIE4_7T_AP'] 
    MSMAll: True/False
    ts_fwhm: smoothing kernel mm
    ts_clean: True/False
    """    
    dtype=np.float16 #np.float32 or np.float32
    mkdir(f'{intermediates_path}/hcp_timeseries')
    imgs_align_sub=[from_cache(get_timeseries_cachepath,get_timeseries,sub,ts_type,filename,MSMAll,dtype,load=True,save=True) for filename in filenames]   
    if ts_type=='movie':   
        movieVidVols = [movieVolumeSelect(v,10,10) for v in movieVidTimes] #get list of all movie volumes to be included  
        movie_index = lambda string: np.where([string==i for i in movies])[0][0] 
        imgs_align_sub = [imgs_align_sub[i][movieVidVols[movie_index(filenames[i])],:] for i in range(len(imgs_align_sub))]
    #Following only relevant if X_clean=True
    clean_each_movie_separately=True
    if clean_each_movie_separately:
        temp=np.vstack([ts_preproc(i) for i in imgs_align_sub])
    else:
        temp=ts_preproc(np.vstack(imgs_align_sub))
    return temp.astype(dtype)

def get_movie_or_rest_string(align_with,runs,fwhm,clean,MSMAll,FC_parcellation_string,FC_normalize):
    runs_string = ''.join([str(i) for i in runs])
    dict1 = {'movie':'mov','rest':'res','movie_FC':'movfc','rest_FC':'resfc','diffusion':'diff'}
    string = f'{dict1[align_with]}{logical2str[MSMAll]}{runs_string}{logical2str[clean]}{fwhm}'
    if 'FC' in align_with:
        if FC_normalize: FC_normalize_string=''
        else: FC_normalize_string='f'
        string=f'{string}{FC_parcellation_string}{logical2str[FC_normalize]}'
    return string

def get_movie_or_rest_data(subs,align_with,prefer='threads',runs=None,fwhm=0,clean=True,MSMAll=False,FC_parcellation_string=None, FC_normalize=None, string_only=False):
    """
    Returns movie viewing or resting state fMRI data, and a string describing the movie or rest data
    align_with: 'movie', 'rest', 'movie_FC', 'rest_FC', 'diffusion'
    prefer: 'threads' (default) or 'processes' for parallelization
    runs: list of runs to use, e.g. [0], [0,2]
    fwhm: spatial smoothing kernel mm
    clean: True for standardization and detrending
    MSMAll: True/False
    FC_parcellation_string: e.g. 'S300', 'K1000'
    FC_normalize: bool
    string_only: bool
    """
    align_string = get_movie_or_rest_string(align_with,runs,fwhm,clean,MSMAll,FC_parcellation_string,FC_normalize)
    if string_only:
        return  [[] for sub in subs],align_string
    else:
        if align_with in ['movie','rest']:
            align_preproc = make_preproc(fwhm,clean,'zscore_sample',True,None,None,1.0)
            filenames = get_filenames(align_with,runs)
            func = lambda sub: get_all_timeseries_sub(sub,align_with,filenames,MSMAll,align_preproc)
            imgs_align=Parallel(n_jobs=-1,prefer="threads")(delayed(func)(sub) for sub in subs)
        elif 'FC' in align_with:     
            filenames = get_filenames(align_with[:-3],runs)
            imgs_align=get_all_FC(subs,[align_with,MSMAll,clean,fwhm,FC_parcellation_string,filenames,'pxn'],FC_normalize)

        if False: #circular shift imgs_align to scramble
            print('circular shift imgs_align to scramble')
            imgs_align.append(imgs_align.pop(0)) 
            imgs_align.append(imgs_align.pop(0))
        if False: #reduce dimensionality of alignment data in ntimepoints/nsamples axis using PCA
            imgs_align, string = reduce_dimensionality_samples(c,imgs_align,ncomponents=300,method='pca')
            align_string = f'{align_string}{string}'

        return imgs_align,align_string

def get_tasks_cachepath(tasks,sub,MSMAll=False):
    mkdir(f'{intermediates_path}/hcp_taskcontrasts')
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
    mkdir(f'{intermediates_path}/hcp_tasklabels')
    task_string=''.join([i[0] for i in tasks])
    return f'{intermediates_path}/hcp_tasklabels/labels_{sub}_{task_string}'

def gettasklabels(tasks,sub):
    #Get 3T task analysis contrast labels
    contrast_files=[ospath(f'{hcp_folder}/{sub}/MNINonLinear/Results/tfMRI_{task}/tfMRI_{task}_hp200_s2_level2.feat/Contrasts.txt') for task in tasks]
    labels = [pd.read_csv(contrast_files[i],header=None).iloc[allowed_labels_dict[tasks[i]]] for i in range(len(tasks))]
    return np.vstack(labels).squeeze()

def get_task_data(subs,tasks,MSMAll=False):
    #Given a list of subjects and some tasks, return a list (nsubjects) of task data arrays (ncontrasts,nvertices), and a description string
    decode_string = f'{len(tasks)}tasks{logical2str[MSMAll]}'
    func = lambda sub: from_cache(get_tasks_cachepath,gettasks,tasks,sub,MSMAll=MSMAll)
    imgs_decode=Parallel(n_jobs=-1,prefer="threads")(delayed(func)(sub) for sub in subs)
    #imgs_decode=[from_cache(get_tasks_cachepath,gettasks,tasks,sub,MSMAll=MSMAll) for sub in subs]  
    #labels=[from_cache(get_tasklabels_cachepath,gettasklabels,tasks,sub) for sub in subs] #list (nsubjects) of labels (ncontrasts,)
    labels = [np.array(range(i.shape[0])) for i in imgs_decode] #since the exact label names are not important, just use the contrast number as the label     
    return imgs_decode, decode_string  

def get_pre_aligned_X_data(foldername,subs):
    """
    Given a foldername and list of subject IDs, return a list (nsubjects) of task data arrays (ncontrasts,nvertices)
    Parameters:
    ----------
    foldername: str
        foldername containing data. The folder contains a .npy file for each subject in format {subname}.npy
    subs: list of str
        list of subject IDs
    """
    return Parallel(n_jobs=-1,prefer='threads')(delayed(np.load)(ospath(f'{intermediates_path}/alignpickles2/{foldername}/{sub}.npy')) for sub in subs)

"""
def get_subjects(sub_slice,subjects):
    subs=subjects[sub_slice]
    sub_slice_string = f'sub{sub_slice.start}to{sub_slice.stop}'
    return subs,sub_slice_string
"""

"""
def get_subjects(sub_slice,subs_template_slice):
    subs=all_subs[sub_slice]
    sub_slice_string = f'sub{sub_slice.start}to{sub_slice.stop}'
    subs_template = all_subs[subs_template_slice] 
    subs_template_slice_string = f'sub{subs_template_slice.start}to{subs_template_slice.stop}'
    return subs,sub_slice_string,subs_template,subs_template_slice_string
"""

def get_decode_data(c,subs,decode_with,align_fwhm,align_clean,MSMAll,decode_ncomponents=None,standardize=None,demean=True,unit_variance=False,parcellation_string=None,use_parcelmeanstds=False):
    """
    Get decoding data. List (nsubjects) of decode data (ncontrasts, nvertices)

    Parameters:
    ----------
    subs: list of subject IDs
    decode_with: string, 'task' or 'movie'
    align_fwhm: float, spatial smoothing kernel mm
    align_clean: bool, True for standardization and detrending
    MSMAll: bool, True/False
    decode_ncomponents: int
        if int is provided, do PCA to reduce nsamples
    standardize: None, 'wholebrain', 'parcel'
        if 'wholebrain', standardize each sample across all vertices
        if 'parcel', standardize each sample within each parcel
    demean: bool, True/False
        relevant if standardize!=None
    unit_variance: bool, True/False
        relevant if standardize!=None
    parcellation_string: string , e.g. 'S300'
        relevant if standardize!=None
    use_parcelmeanstds: bool, True/False
        if True, return parcel mean and std for each sample for usage in classification later
    """
    print(f"{c.time()} Get decoding data start")
    if decode_with=='tasks': #task fMRI contrasts
        imgs_decode,decode_string = get_task_data(subs,tasks[0:7],MSMAll=MSMAll)
    elif decode_with=='movie': #### Decode movie viewing data instead
        print("Decode data is movie viewing runs 2 and 3")
        imgs_decode,decode_string = get_movie_or_rest_data(subs,'movie',runs=[2,3],fwhm=align_fwhm,clean=align_clean,MSMAll=MSMAll)
    if decode_ncomponents is not None:
        print(f"{c.time()} Decode data, PCA start")
        imgs_decode, string = reduce_dimensionality_samples(c,imgs_decode,ncomponents=decode_ncomponents,method='pca')
        print(f"Keep last 20 out of {decode_ncomponents} components")
        imgs_decode=[i[-20:] for i in imgs_decode] #only keep last 20 out of decode_ncomponents PCA components, to make classification task harder
        string+=f"to20"
        print(f"{c.time()} Decode data, PCA, end")
        decode_string = f'{decode_string}{string}'

    if use_parcelmeanstds: 
        clustering = parcellation_string_to_parcellation(parcellation_string)
        imgs_decode_meanstds = [get_parcelwise_mean_and_std(img,clustering) for img in imgs_decode]
        decode_string = f'{decode_string}&ms'
    else: 
        imgs_decode_meanstds = None

    if standardize == 'wholebrain':
        if unit_variance:
            from scipy.stats import zscore
            imgs_decode = [zscore(i,axis=1) for i in imgs_decode]
        else:
            imgs_decode = [i - np.mean(i, axis=1, keepdims=True) for i in imgs_decode]
    elif standardize == 'parcel':
        clustering = parcellation_string_to_parcellation(parcellation_string)
        parc_matrix = parcellation_string_to_parcmatrix(parcellation_string)
        imgs_decode = [standardize_image_parcelwise(img,clustering,parc_matrix,demean=demean,unit_variance=unit_variance) for img in imgs_decode]
    if standardize is not None:
        decode_string = f'{decode_string}{standardize[0].capitalize()}{logical2str[demean]}{logical2str[unit_variance]}'

    return imgs_decode,decode_string, imgs_decode_meanstds

def get_alignment_data(c,subs,method,align_with,runs,align_fwhm,align_clean,MSMAll,load_pickle,FC_parcellation_string=None,FC_normalize=None):
    """
    Get alignment data. List (nsubjects) of alignment data (nsamples,nvertices)
    """
    print(f"{c.time()} Get alignment data start")
    if method=='anat': 
        align_string='anat'
        imgs_align = [[] for sub in subs] #irrelevant anyway
    else:
        imgs_align, align_string = get_movie_or_rest_data(subs,align_with,runs=runs,fwhm=align_fwhm,clean=align_clean,MSMAll=MSMAll,string_only=load_pickle,FC_parcellation_string=FC_parcellation_string,FC_normalize=FC_normalize) #load_pickle=True means return string only
    #imgs_align,align_string = get_aligndata_highres_connectomes(c,subs,MSMAll,{'sift2':False , 'tckfile':'tracks_5M_sift1M.tck' , 'targets_nparcs':False , 'targets_nvertices':16000 , 'fwhm_circ':3 })  

    return imgs_align,align_string


def get_template_making_alignment_data(c,method,subs_template,subs_template_slice_string,align_with,runs,align_fwhm,align_clean,MSMAll,load_pickle,lowdim_template,args_template,n_bags_template,gamma_template,FC_parcellation_string,FC_normalize):
    #### Get template-making alignment data. List (nsubjects) of data (nsamples,nvertices)
    print(f"{c.time()} Get template-making data start")  
    if method=='template':
        imgs_template, template_imgtype_string = get_movie_or_rest_data(subs_template,align_with,runs=runs,fwhm=align_fwhm,clean=align_clean,MSMAll=MSMAll,string_only=load_pickle,FC_parcellation_string=FC_parcellation_string,FC_normalize=FC_normalize) #load_pickle=True means return string only
        template_string = f'_T{template_imgtype_string}{subs_template_slice_string}_{get_template_making_string(lowdim_template,args_template,n_bags_template,gamma_template)}'
    else:
        imgs_template, template_string = None, ''
    return imgs_template,template_string

def get_template_making_string(lowdim_template,args,n_bags_template,gamma_template):
    """
    Return short string describing how template was made
    """
    dict2 = {'rescale':'r','zscore':'z',None:'n'}
    string=""
    if lowdim_template:
        string = 'L'
    else:
        if args['do_level_1']==True:
            template_type_string = 'H'
        else:
            template_type_string = 'G'
        string += f"{template_type_string}{args['n_iter']}{logical2str[args['remove_self']]}{logical2str[args['level1_equal_weight']]}{dict2[args['normalize_imgs']]}{dict2[args['normalize_template']]}"
        if gamma_template!=0:
            string+=f"gam{gamma_template}"
    if n_bags_template!=1:
        string+=f"b{n_bags_template}"

    return string

def alignment_method_string(method,alignment_method,alignment_kwargs,per_parcel_kwargs,n_bags,gamma):
    """
    Returns a string describing keyword arguments passed to SurfacePairwiseAlignment
    """
    string=f"{method[0:4].capitalize()}{alignment_method[0:4].capitalize()}_"
    if n_bags!=1:
        string+=f"b{n_bags}"
    if type(gamma) in [list,np.ndarray]:
        string+=f"gamcustom"
    else:
        if gamma!=0:
            string+=f"gam{gamma}"
    if 'scaling' in alignment_kwargs:
        string += f"sc{logical2str[alignment_kwargs['scaling']]}"
    if 'scca_alpha' in alignment_kwargs:
        string += f"scca{alignment_kwargs['scca_alpha']}"
    if 'promises_k' in alignment_kwargs:
        string += f"ProM{alignment_kwargs['promises_k']}"
    if 'alphas' in alignment_kwargs:
        string += f"alphas{alignment_kwargs['alphas']}"
    if 'reg' in alignment_kwargs:
        string += f"reg{alignment_kwargs['reg']}"
    if 'max_iter' in alignment_kwargs:
        string += f"maxiter{alignment_kwargs['max_iter']}"
    if 'tol' in alignment_kwargs:
        string += f"tol{alignment_kwargs['tol']}"
    return string

def get_all_pairwise_aligners(subs,imgs_align,alignment_method,clustering,n_bags,n_jobs,alignment_kwargs,per_parcel_kwargs,gamma,absValueOfAligner):
    """
    Calculate alignment transformations between all pairs of subjects. First find all aligners for all pairs of subjects and put them in a list 'temp'. Then assign each aligner to a dictionary 'aligners' with keys '100610-102310', '100610-102816', etc.
    INPUTS:
    subs: list
        subject IDs
    Other parameters are same as function func() in hcpalign.py
    RETURNS:
    aligners: dict
        Values are SurfacePairwiseAlignment objects
        Key "100610-102310" points to the transformation from subject 100610 to subject 102310 
    """
    from fmralign.surf_pairwise_alignment import SurfacePairwiseAlignment
    import itertools
    subs_indices = np.arange(len(subs))
    def init_and_fit_aligner(source_align, target_align, absValueOfAligner):
        aligner=SurfacePairwiseAlignment(alignment_method=alignment_method, clustering=clustering,n_bags=n_bags,n_jobs=n_jobs,alignment_kwargs=alignment_kwargs,per_parcel_kwargs=per_parcel_kwargs,gamma=gamma)
        print(memused())
        aligner.fit(source_align, target_align)
        print(memused())
        if absValueOfAligner: aligner_absvalue(aligner)
        return aligner
    temp=Parallel(n_jobs=-1,prefer='processes')(delayed(init_and_fit_aligner)(imgs_align[source],imgs_align[target], absValueOfAligner) for source,target in itertools.permutations(subs_indices,2))

    aligners={}
    index=0
    for source,target in itertools.permutations(subs_indices,2):
        aligners[f'{subs[source]}-{subs[target]}'] = temp[index]
        index +=1
    del temp
    return aligners


from sklearn.base import clone
def classify_pairwise(target,subs_indices,labels,target_decode,aligned_sources_decode,classifier):
    """
    target: index of subject to be decoded
    subs_indices: indices of all subjects
    labels: list (nsubjects) of labels (ncontrasts,)
    target_decode: decode data for target subject (ncontrasts,nvertices)
    aligned_sources_decode: list (n_nontargetsubjects) of arrays (ncontrasts,nvertices)
        Decode data for non-target subjects, aligned to target subject's functional space
    """
    sources=[i for i in subs_indices if i!=target]
    sources_labels=[labels[i] for i in sources] 
    target_labels = labels[target] #labels for target subject (ncontrasts,)
    clf=clone(classifier)
    clf.fit(aligned_sources_decode, np.hstack(sources_labels))
    return clf.score(target_decode, target_labels)


def transform_all_decode_data(subs,imgs_decode,aligners,post_decode_smooth,imgs_decode_meanstds=None,stack=True):
    """
    Transform decode data for all non-target subjects to the functional space of a target subject. Repeat this, using a different target subject each time.

    Parameters
    ----------
    subs: list
        subject IDs
    imgs_decode: list
        list (nsubjects) of decode data arrays (ncontrasts,nvertices)
    aligners: dict
        Values are SurfacePairwiseAlignment objects
        Key "100610-102310" points to the transformation from subject 100610 to subject 102310    
    post_decode_smooth: function
        To spatially smooth data after transformation
    imgs_decode_meanstds: list
        list (nsubjects) of parcel mean and std for each sample for usage in classification later
    stack: bool
        if True, stack the decode data of all non-target subjects

    Returns
    ----------
    all_aligned_sources_decode: list
        list (nsubjects) of all others subjects' transformed decode data concatenated (ncontrasts*nleftoutsubjects,nvertices)
    """
    subs_indices = np.arange(len(subs)) #list from 0 to n_subs-1
    all_aligned_sources_decode=[]
    for target in subs_indices:
        sources=[i for i in subs_indices if i!=target]  
        aligned_sources_decode=[]
        for source in sources:
            source_decode = imgs_decode[source]
            aligner=aligners[f'{subs[source]}-{subs[target]}'] 
            source_decode_aligned = post_decode_smooth(aligner.transform(source_decode))
            if imgs_decode_meanstds is not None:
                source_decode_meanstds = imgs_decode_meanstds[source]
                source_decode_aligned = np.hstack([source_decode_aligned,source_decode_meanstds])
            aligned_sources_decode.append(source_decode_aligned)
        if stack:
            aligned_sources_decode = np.vstack(aligned_sources_decode) #(ncontrasts*nleftoutsubjects,nvertices)
        all_aligned_sources_decode.append(aligned_sources_decode)
    return all_aligned_sources_decode


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
    MSMAll,
    align_clean,
    align_fwhm,
    parcellation_string,
    filenames,
    FC_type): 
    if parcellation_string[0]=='S': targets_parcellation = 'Schaefer'
    elif parcellation_string[0]=='K': targets_parcellation = 'kmeans'
    else: assert(0)
    align_clean_string = logical2str[align_clean]    
    MSMtypestring = MSMlogical2str[MSMAll]    
    return f'{intermediates_path}/functional_connectivity/{get_func_type(align_with)}3T_{MSMtypestring}_{len(filenames)}runs_{align_clean_string}_fwhm{align_fwhm}_{targets_parcellation}{parcellation_string[1:]}_{FC_type}_sub{sub}.p'

def get_FC(
    sub,
    align_with,
    MSMAll,
    align_clean,
    align_fwhm,
    parcellation_string,
    filenames,
    FC_type):   
    """
    FC_type is 'pxn' or pxp'
    """
    #_,parc_matrix=get_parcellation(targets_parcellation,targets_nparcs)
    parc_matrix = parcellation_string_to_parcmatrix(parcellation_string)

    standardize,detrend,low_pass,high_pass,t_r='zscore_sample',True,None,None,1.0 #These parameters only apply to 'movie' and 'decode' data depending on whether movie_clean=True or decode_clean=True
    align_preproc = make_preproc(align_fwhm,align_clean,standardize,detrend,low_pass,high_pass,t_r)
    na = get_all_timeseries_sub(sub,get_func_type(align_with),filenames,MSMAll,align_preproc)
    nap=(na@parc_matrix.T) #ntimepoints * nparcs
    if FC_type=='pxn':
        return corr4(nap,na,b_blocksize=parc_matrix.shape[0])
    elif FC_type=='pxp':
        return corr4(nap,nap,b_blocksize=parc_matrix.shape[0])

def get_all_FC(subs,args,normalize):
    """
    Given a list of subjects, get functional connectivity data for each subject. Returns a list (nsubjects) of FC arrays (nparcels,nparcels) or (ntargets,nparcels)
    Parameters:
    ----------
    subs: list of subject IDs (str)
    args: list of arguments to pass to get_FC
    normalize: bool
        if True, normalize columns of FC arrays, so that each vertex's distribution of connectivities (to targets) is 0-centred 
    """
    imgs_align = Parallel(n_jobs=-1,prefer='threads')(delayed(from_cache)(get_FC_filepath, get_FC, *(sub, *args), load=True, save=True) for sub in subs)

    if normalize:
        from sklearn.preprocessing import StandardScaler
        print('normalizing FC arrays start')
        imgs_align = Parallel(n_jobs=-1,prefer='threads')(delayed(StandardScaler().fit_transform)(i) for i in imgs_align)
        print('normalizing FC arrays done')

    return [i.astype(np.float16) for i in imgs_align] 


def reduce_dimensionality_samples(c,imgs_align,ncomponents,method):
    """
    Reduce data dimensionality in axis 0 (nsamples,ntimepoints)
    """
    from sklearn.decomposition import PCA,FastICA
    if method=='pca':
        decomp=PCA(n_components=ncomponents,whiten=False,random_state=0)
    elif method=='ica':
        decomp=FastICA(n_components=ncomponents,max_iter=100000,random_state=0)
    temp=np.dstack(imgs_align).mean(axis=2) #mean of all subjects
    decomp.fit(temp.T)
    imgs_align=[decomp.transform(i.T).T for i in imgs_align]     
    return imgs_align, f'{method}{ncomponents}'

def dataframe_get_subs(df,subjects):
    """
    Extract rows from dataframe corresponding to subjects. Make sure that rows of df are sorted in same order as subjects
    Parameters:
    ----------
    df: pandas dataframe with a column 'Subject' with dtype int
    subjects: list of subject IDs (str)

    Returns:
    --------
    df2: subset of original pandas dataframe with rows corresponding to subjects
    """
    subjects = [int(i) for i in subjects]
    df2 = df[df['Subject'].isin(subjects)].copy()
    # Specify a custom sort order for the 'Subject' column using pandas.Categorical
    df2['Subject'] = pd.Categorical(df2['Subject'], categories=subjects, ordered=True)
    # Now sort by 'Subject' using the custom order
    df2 = df2.sort_values(by='Subject').reset_index(drop=True)
    return df2

def do_quantile_transform(y):
    """
    Transform list of values to quantiles on a uniform distribution
    """
    from sklearn.preprocessing import QuantileTransformer
    import warnings
    quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_quantiles .* is set to n_samples.")
        y = quantile_transformer.fit_transform(y.reshape(-1, 1)).ravel()
    return y

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

def aligner_absvalue(aligner):
    #Given a SurfacePairwiseAlignment object, change the elements in alignment matrices to absolute values
    for i in range(len(aligner.fit_)):
        try:
            aligner.fit_[i].R = np.abs(aligner.fit_[i].R)
        except:
            pass #usually when fit_ is identity matrix due to empty parcel

def aligner_descale(aligner):
    #Given a SurfacePairwiseAlignment object, remove the scale factor from Procrustes alignment
    for i in range(len(aligner.fit_)):
        try:
            scale=aligner.fit_[i].scale
            aligner.fit_[i].scale=1
            aligner.fit_[i].R /= scale                          
        except:
            pass

def aligner_get_scale_map(aligner):
    #Given a MySurfacePairwiseAlignment object, return the 'scale' value at each vertex
    scales=np.ones((len(aligner.clustering)))
    parcels=np.unique(aligner.clustering)
    for i in range(len(parcels)):
        try:
            parcel=parcels[i]
            scale=aligner.fit_[i].scale
            indices=np.where(aligner.clustering==parcel)[0]
            scales[indices]=scale
        except: 
            pass
    return scales

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

def corr_rows_parcel(imgs_decode_parcel):
    return np.dstack(Parallel(n_jobs=-1,prefer='processes')(delayed(corr_rows)(i,prefer='noparallel') for i in imgs_decode_parcel)) #array (nsamples,nsubjectpairs,nparcels)

import itertools

def corr_rows(x,prefer='noparallel'):
    """
    Given a list of 2D arrays (nsamples,nfeatures), one array for each subject, for each row index, for each pair of subjects, calculate the correlation coefficient between their corresponding rows

    Parameters:
    ----------
    x: list (nsubjects) of arrays (nsamples,nfeatures)
    prefer: 'processes', 'threads', or 'noparallel'

    Returns:
    ----------
    correlations: array (nsamples,nsubjectpairs)
        correlations between rows of x
    """
    nsubjects=len(x)
    if prefer=='noparallel':
        temp = [rowcorr_nonsparse(x[i].T,x[j].T) for i,j in itertools.combinations(range(nsubjects),2)] #list (nsubjectpairs) of arrays (nfeatures). Each array containing correlations between rows of x[i] and x[j]
    else:
        temp = Parallel(n_jobs=-1,prefer=prefer)(delayed(rowcorr_nonsparse)(x[i].T,x[j].T) for i,j in itertools.combinations(range(nsubjects),2))
    result = np.vstack(temp).T
    del temp
    return result

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
    """Non-sparse array version of above. Row here means axis 0 so it is actually columns"""
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


def get_aligndata_highres_connectomes(c,subs,MSMAll,tckfile='tracks_5M_sift1M.tck',sift2=False,fwhm=3,targets_nparcs=False,targets_nvertices=16000):
    #Get high-res-connectomes as alignment data, and returns a short description string
    imgs_align = get_highres_connectomes(c,subs,tckfile,MSMAll=MSMAll,sift2=sift2)
    if fwhm:
        imgs_align = smooth_highres_connectomes_mm(imgs_align,fwhm)
        imgs_align = [i.astype(np.float32) for i in imgs_align]
    if targets_nparcs: #connectivity from each vertex, to each targetparcel
        align_parc_matrix=Schaefer_matrix(targets_nparcs) 
        imgs_align=[align_parc_matrix.dot(i) for i in imgs_align]
    else:
        these_vertices=np.linspace(0,imgs_align[0].shape[0]-1,targets_nvertices).astype(int) #default 16000   
        imgs_align=[i[these_vertices,:] for i in imgs_align]         
    imgs_align=[i.toarray().astype('float32') for i in imgs_align]  
    string = f'diff{logical2str[MSMAll]}{logical2str["sift2"]}{fwhm}{targets_nvertices}{tckfile[:-4]}'
    return imgs_align,string

def smooth_highres_connectomes(hr,smoother):
    """
    Parallelizes connectome-spatial-smoothing.smooth_high_resolution_connectome
    hr is list of sparse connectomes from get_highres_connectomes
    smoother is a smoothing kernel as a sparse array
    """   
    from Connectome_Spatial_Smoothing import CSS as css   
    def func(hrs): return css.smooth_high_resolution_connectome(hrs,smoother)
    hr=Parallel(n_jobs=-1,prefer='threads')(delayed(func)(i) for i in hr)   
    return hr 

def smooth_highres_connectomes_mm(hr,fwhm):
    #fwhm: float
    smoother=sparse.load_npz(ospath(f"{intermediates_path}/smoothers/100610_{fwhm}_0.01.npz"))
    return smooth_highres_connectomes(hr,smoother)

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
    import hcpalign_utils as hcp
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
    import hcpalign_utils as hcp
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

def surfgeodistances(source_vertices_59k, surf=None):
    """
    Given geodesic surface distances
    Inputs: source_vertices_59k - array(n,) of source vertices in 59k cortex space
            surf - surface vertices/triangles tuple(2,) in 59k cortex space
    Output: array(59k,) of distances from source vertices
    """
    import gdist
    if surf is None: 
        import hcp_utils as hcp
        surf = hcp.mesh.midthickness
    source_vertices_64k=vertex_59kto64k(source_vertices_59k).astype('int32')
    distances_64k=gdist.compute_gdist(surf[0].astype('float64'),surf[1],source_vertices_64k)
    distances_59k=cortex_64kto59k(distances_64k)
    return distances_59k

def surfgeoroi(source_vertices_59k,limit=0,surf=None):
    """
    Like wb_command -surface-geodesic-rois
    Output list of all vertices within limit mm of source vertices
    If limit==0, then just output source vertices
    If limit==0, return entire cortex
    """
    if surf is None: 
        import hcp_utils as hcp
        surf = hcp.mesh.midthickness
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


def get_parcelwise_mean_and_std(img,clustering):
        img_parcel = [img[:,clustering==i] for i in np.unique(clustering)] #divided into parcels
        img_parcelsamplemeans = np.hstack([np.mean(array,axis=1,keepdims=True) for array in img_parcel]) #array (nsamples,nparcels) containing mean value for each parcel in each sample
        img_parcelsamplestds = np.hstack([np.std(array,axis=1,keepdims=True) for array in img_parcel]) 
        return np.hstack([img_parcelsamplemeans,img_parcelsamplestds]) #array (nsamples,2*nparcels) containing mean and std for each parcel in each sample

def get_parcelwise_mean(img,clustering):
        img_parcel = [img[:,clustering==i] for i in np.unique(clustering)] #divided into parcels
        img_parcelsamplemeans = np.hstack([np.mean(array,axis=1,keepdims=True) for array in img_parcel]) #array (nsamples,nparcels) containing mean value for each parcel in each sample
        return img_parcelsamplemeans 

def standardize_image_parcelwise(img,clustering,parc_matrix,demean=True,unit_variance=True):
    """
    Given some samples of brain images, standardize the data corresponding to each sample and each parcel. 
    Parameters:
    -----------
    img: array of shape (nsamples,nfeatures)
    clustering: array of shape (nfeatures,) 
        Containing the parcel number for each feature
    parc_matrix: array of shape (nfeatures,nvertices)
        Each row contains a 1 in the columns corresponding to the vertices in the parcel
    demean: bool
        If True, subtract the mean of each parcel from each sample
    unit_variance: bool 
        If True, divide each sample by the standard deviation of each parcel

    Returns:
    ----------
    img: array of shape (nsamples,nfeatures)
        Standardized image data
    img_parcelsamplemeans: array of shape (nsamples,nparcels)
        Mean value for each parcel in each sample (return None if demean==False)
    img_parcelsamplestds: array of shape (nsamples,nparcels)
        Standard deviation for each parcel in each sample (return None if unit_variance==False)
    """

    img_parcel = [img[:,clustering==i] for i in np.unique(clustering)] #divided into parcels

    if demean:
        img_parcelsamplemeans = np.hstack([np.mean(array,axis=1,keepdims=True) for array in img_parcel]) #array (nsamples,nparcels) containing mean value for each parcel in each sample
        img_parcelsamplemeans_vertices = img_parcelsamplemeans @ parc_matrix #array of same size as 'img', except now containing the means
        img = img - img_parcelsamplemeans_vertices #means for each parcel and sample have been subtracted 
    if unit_variance:
        img_parcelsamplestds = np.hstack([np.std(array,axis=1,keepdims=True) for array in img_parcel]) 
        img_parcelsamplestds_vertices = img_parcelsamplestds @ parc_matrix 
        img = img / img_parcelsamplestds_vertices #divide by standard deviation to produce z-scores
    return img

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

def parcellation_string_to_parcellation(parcellation_string):
    #Inputs: parcellation_string: 'S300' for Schaefer 300, 'K400' for kmeans 400, 'M' for HCP multimodal parcellation
    #Returns an array of size (59412,) with parcel labels for each vertex in fs32k cortex
    import hcp_utils as hcp
    nparcs = int(parcellation_string[1:])
    if parcellation_string[0]=='S':      
        parcellation = Schaefer(nparcs)
    elif parcellation_string[0]=='K':
        parcellation = kmeans(nparcs)
    elif parcellation_string[0]=='M':
        parcellation = hcp.mmp.map_all[hcp.struct.cortex]
    return parcellation

def parcellation_string_to_parcmatrix(parcellation_string):
    #Inputs: parcellation_string: 'S300' for Schaefer 300, 'K400' for kmeans 400, 'M' for HCP multimodal parcellation
    #Returns parcellation matrix (nparcs,nvertices)
    import hcp_utils as hcp
    nparcs = int(parcellation_string[1:])
    if parcellation_string[0]=='S':      
        matrix = Schaefer_matrix(nparcs).astype(bool)
    elif parcellation_string[0]=='K':
        matrix = kmeans_matrix(nparcs).astype(bool)
    elif parcellation_string[0]=='M':
        matrix = parc_char_matrix(hcp.mmp.map_all[hcp.struct.cortex])[1].astype(bool)
    nonempty_parcels = np.array((matrix.sum(axis=1)!=0)).squeeze()
    assert(len(nonempty_parcels)==matrix.shape[0]) #no empty parcels
    return matrix

'''
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
'''
    
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
    from pathlib import Path
    def __init__(self, figpath,mesh=None,vmin=None,vmax=None,cmap='inferno',symmetric_cmap=True,plot_type='open_in_browser'):
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


def do_plot_impulse_responses(p,plot_prefix,aligner,radius=1,vertices=None):
    """
    Show the impulse response of a functional alignment matrix. That is, how a small circular ROI activation is transformed by alignment. Also return the ratio of the vector norm of the map that is contained within the original ROI.

    Parameters
    ----------
    p: hcpalign_utils.surfplot instance
    plot_prefix: string
    aligner: fmralign.SurfacePairwiseAlignment object
    radius: float
        radius of the ROI in mm
    """
    if vertices is None:
        vertices_L = [1,2,5,6,8,9,17500] #[1,2,3,4,5,6,7,8,29696+1,29696+2,29696+3,29696+4,29696+5,29696+6,29696+7,29696+8]
        vertices_R = [29696 + i for i in vertices_L]
        vertices = vertices_L + vertices_R
    s=surfgeoroi(vertices,radius)
    t=aligner.transform(s[None,:])[0]     
    p.plot(s, f"x{plot_prefix}_roi_source{radius}",symmetric_cmap=True)
    p.plot(np.squeeze(t),f"x{plot_prefix}_roi_target{radius}",symmetric_cmap=True)
    ratio_within_roi = np.linalg.norm(t[s!=0]) / np.linalg.norm(t) #what proportion of the spatial map's 'norm' is within the original ROI? High value means the aligner is highly spatially regularized
    return ratio_within_roi



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
    return f'Python mem: {process.memory_info().rss/1e9:.2f} GB, PC mem: {psutil.virtual_memory()[3]/1e9:.2f}/{psutil.virtual_memory()[0]/1e9:.0f} GB'

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
    import hcp_utils as hcp
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

