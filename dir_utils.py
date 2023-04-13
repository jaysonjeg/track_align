"""
Utility functions for tkdir.py
"""
import hcpalign_utils as hutils
import tkalign_utils as tutils
import numpy as np
import pickle


align_nparcs=300
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)
nparcs=301
nblocks=5
n=4 #decimal places for answers

def pr(value, string=None):
    if string is None:
        print(f'{value:.{n}f}')
    else:
        print(f'{string} {value:.{n}f}')
def get_a(z):
    b,f,a=tutils.load_f_and_a(z)
    return a
def reshape(a):
    if a.ndim==1:
        return np.transpose(np.reshape(a,(nblocks,nparcs)))
    elif a.ndim==2:
        return np.transpose(np.reshape(a,(a.shape[0],nblocks,nparcs)),(0,2,1))
    elif a.ndim==3:
        return np.transpose( np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs)) , (0,1,3,2)) #Now a is n_subs_test * n_subs_test * nparcs * nblocksperparc  
    else:
        assert(0)  
def removenans(arr):
    arr = arr[~np.isnan(arr)]
    return np.array(arr)
def count(a):
    if a.ndim==4:
        out= [tutils.count_negs(a[:,:,i,:]) for i in range(a.shape[2])]    
    elif a.ndim==3:
        out= [tutils.count_negs(a[:,:,i]) for i in range(a.shape[-1])]
    elif a.ndim==2:
        out= [tutils.count_negs(a[:,i]) for i in range(a.shape[-1])]
    return np.array(out)
def minus(a,b):
    #subtract corresponding elements in two lists
    return [i-j for i,j in zip(a,b)]
def get_vals(a):
    array=[]
    for i in range(a.shape[2]):
        if a.ndim==4:
            values = removenans(a[:,:,i,:])
        elif a.ndim==3:
            values = removenans(a[:,:,i])
        array.append(values)
    output = np.array(array).T
    assert(output.ndim==2)
    return output
def corr(x):
    #Correlations between columns in x, averaged across all pairs of columns (ignoring corr between a col and itself)
    values = np.corrcoef(x)
    ones = np.ones(values.shape)
    non_ones = ~(np.isclose(values,ones))
    #non_ones = values!=1
    return np.mean(values[non_ones])
def OLS(Y,X):
    #Y is array(n,1). X is array of regressors (n,nregressor). Outputs R
    #e.g. X = np.column_stack((dce, tce, np.ones(len(dce))))
    import statsmodels.api as sm
    model = sm.OLS(Y, X).fit()
    print("R-squared: {:.3f}".format(model.rsquared))
    print("R: {:.3f}".format(np.sqrt(model.rsquared)))
def trinarize(mylist,cutoff):
    """
    Given a list li, convert values above 'cutoff' to 1, values below -cutoff to -1, and all other values to 0
    """
    li=np.copy(mylist)
    greater=li>cutoff
    lesser=li<-cutoff
    neither= np.logical_not(np.logical_or(greater,lesser))
    li[greater]=1
    li[lesser]=-1
    li[neither]=0
    return li

### Get confounders ###

def get_nverts_parc():
    #number of vertices in each parcel
    nverts_parc=np.array(align_parc_matrix.sum(axis=1)).squeeze()
    nverts_parc[0]=nverts_parc.mean()
    return nverts_parc

def get_aligner_scale_factor():
    alignfile = 'hcpalign_movie_temp_scaled_orthogonal_10-4-7_TF_0_0_0_FFF_S300_False'
    aligner_file = f'{hutils.intermediates_path}/alignpickles/{alignfile}.p'
    all_aligners = pickle.load( open( hutils.ospath(aligner_file), "rb" ))
    scales = np.vstack( [[all_aligners.estimators[i].fit_[nparc].scale for nparc in range(nparcs)] for i in range(len(all_aligners.estimators))] )   
    scales_mean=scales.mean(axis=0)    
    log_scales_mean=np.log10(scales_mean)
    log_scales_mean_adj = log_scales_mean - log_scales_mean.min()
    return scales_mean,log_scales_mean_adj

def get_vertex_areas():
    path='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\old_intermediates\\vertex_areas\\vert_area_100610_white.npy'
    vertex_areas=np.load(path)
    total_areas_parc = align_parc_matrix @ vertex_areas 
    mean_vert_areas_parc = total_areas_parc / get_nverts_parc()
    return total_areas_parc , mean_vert_areas_parc 
 