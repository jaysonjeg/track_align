"""
Utility functions for tkdir.py
"""
import hcpalign_utils as hutils
import tkalign_utils as tutils
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt


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
    X =  np.column_stack((X[0], X[1], np.ones(len(X[0]))))
    model = sm.OLS(Y, X).fit()
    return model
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

def get_aligner_variability(alignfile):
    aligner_file = f'{hutils.intermediates_path}/alignpickles/{alignfile}.p'
    all_aligners = pickle.load( open( hutils.ospath(aligner_file), "rb" ))
    nparcs = len(all_aligners.estimators[0].fit_)
    nsubjectpairs = len([i for i in itertools.combinations(range(len(all_aligners.estimators)),2)])
    values = np.zeros((nparcs,nsubjectpairs))
    #normalize all the arrays inside
    for nparc in range(nparcs):
        for i in range(len(all_aligners.estimators)):
            norm = np.linalg.norm(all_aligners.estimators[i].fit_[nparc].R)
            array = all_aligners.estimators[i].fit_[nparc].R
            all_aligners.estimators[i].fit_[nparc].R = array / norm
    #get norms of differences for all subject pairs
    for nparc in range(nparcs):
        count=-1
        for i,j in itertools.combinations(range(len(all_aligners.estimators)),2):
            count +=1
            array_i = all_aligners.estimators[i].fit_[nparc].R
            array_j = all_aligners.estimators[j].fit_[nparc].R
            values[nparc,count] = np.linalg.norm(array_i - array_j)
    Y =  values.mean(axis=1)

    #regress out impact of parcel area, on these topographic variability
    total_areas_parc , _ =  get_vertex_areas() 
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=1)
    regressors = poly.fit_transform(np.sqrt(total_areas_parc).reshape(-1,1))
    model = LinearRegression().fit(regressors, Y)
    Y_pred = model.predict(regressors)
    residuals = Y - Y_pred
    return residuals,Y



def get_aligner_scale_factor(alignfile):
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
 
def get_mean_strengths():
    path='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\AWS_studies\\files0\\intermediates\\mean_strengths_S300_50subs.npy'
    with open(path,'rb') as f:
        mean_strengths_50subs = np.load(f)
    return mean_strengths_50subs

def make_null(list_of_values,samplesize,n_nulls):
    return [np.mean(np.random.choice(list_of_values,size=samplesize,replace=False)) for i in range(n_nulls)]
def get_all_nulls(list_of_values,max_samplesize=40,n_nulls=1000):
    allnulls={}
    for samplesize in range(1,max_samplesize):
        allnulls[samplesize]=make_null(list_of_values,samplesize,n_nulls)
    return allnulls
def get_percentiles(cluster_means,cluster_sizes,nulls):
    output=[]
    for i in range(len(cluster_sizes)):
        cluster_mean=cluster_means[i]
        cluster_size=cluster_sizes[i]
        null = nulls[cluster_size]
        percentile = np.sum(cluster_mean > null)*100/len(null)
        output.append(percentile)
    return output



def bar_plot(data,labels,title,xlim,cmap=None,leftmargin=None,vertlines=False):
    data=np.array(data)
    data[data<1]=1

    if cmap is None:
        cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots()
    y_pos = np.arange(len(data))
    ax.barh(y_pos, data, color=cmap(np.arange(len(data))), alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Original > Scrambled (%)')
    ax.set_ylabel(title)
    #ax.set_title(title)
    ax.set_xlim(xlim)
    if vertlines:
        ax.axvline(x=2.5, color='r')
        ax.axvline(x=97.5, color='r')

    if leftmargin is not None:
        fig.subplots_adjust(left=leftmargin)
    # Create legend
    #handles = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(len(data))]
    #ax.legend(handles, labels)



