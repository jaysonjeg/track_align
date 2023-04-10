import re
import numpy as np
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse, stats
import hcpalign_utils as hutils
from hcpalign_utils import memused, sizeof, ospath
import matplotlib.pyplot as plt
import warnings
import itertools

def extract_nparcs(string):
    pattern = r'_FFF_S(.*?)_False'
    match = re.search(pattern, string)
    if match:
        return int(match.group(1))
    else:
        return None

def makesorter(labels):
    """
    Given list of unordered labels (59412,), return sorting indices, unsorting indices, and labelslice. labelslice is list containing slice for each label in the sorted list
    """
    #Makes labelslice for unique labels only
    unique_labels=np.unique(labels)
    sorter=np.argsort(labels,kind='stable')
    unsorter=np.argsort(sorter)
    labelsort=labels[sorter]
    labelslice=np.zeros(len(unique_labels),dtype=object)
    for i in range(len(unique_labels)):
        where=np.where(labelsort==unique_labels[i])[0]
        labelslice[i]=slice(where[0],where[-1]+1)
    return sorter,unsorter,labelslice

def load_f_and_a(save_path):
    print(f"Loading from {save_path}")
    temp=np.load(save_path,allow_pickle=True)
    blocks=temp[()]['blocks']
    f=temp[()]['f']
    a=temp[()]['a']
    return blocks,f,a

def get_hps(hp):       
    hps=sum(hp)
    hpsx=hps.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        hpsx.setdiag(0)
        hpsx.eliminate_zeros() 
    hpsxa=hpsx.toarray()
    return hps,hpsx,hpsxa
def get_blocks_largest(nblocks,parcel_pair_type,hps,hpsx):
    if parcel_pair_type=='i':
        temp=hutils.indices_of_largest_values_in_array(sparse.csr_matrix.diagonal(hps),nblocks)  
        blocks = np.vstack([temp,temp])
    elif parcel_pair_type=='o': 
        blocks = hutils.indices_of_largest_values_in_array(hpsx,nblocks)
    return blocks
def get_blocks_source(align_labels,hpsxa,nblocks,nparcs,type_rowcol):
    ###To get all blocks linking to a target parcel (containing a target vertex) ###
    #e.g. L calcarine 23178, R calcarine 52994, L lateral occipital 21696, R lateral occipital 51512
    source_vertex=23178
    target_parc=align_labels[source_vertex]
    temp=hpsxa[:,target_parc]
    nblocks = min(nblocks , nparcs) #can't be bigger than the total number of blocks
    temp2=hutils.indices_of_largest_values_in_array(temp,len(temp)) 
    temp1=np.ones((nparcs),dtype=int) * target_parc
    if type_rowcol=='row':
        blocks= np.vstack([temp1,temp2])
    if type_rowcol=='col':
        blocks= np.vstack([temp2,temp1])
    return blocks
def get_blocks_all():
    nparcs=5
    inds=np.triu_indices(nparcs)
    blocks=np.vstack(inds)
    return blocks
def get_blocks_maxhpmult(nblocks,hpsxa,nparcs):
    nblocks_per_parcel=nblocks
    inds=np.argpartition(hpsxa,-nblocks_per_parcel,axis=0)[-nblocks_per_parcel:,:]
    temp2=inds.ravel()
    temp0a=np.array([i for i in range(nparcs)])
    temp0b=np.tile(temp0a,(nblocks_per_parcel,1))
    temp1=temp0b.ravel()   
    blocks=np.vstack([temp1,temp2])
    return blocks

import fmralign
def aligner2sparse(a,slices,fa_sparse=False,aligned_descale=False,aligned_negs='abs'):
    """
    Convert pairwise aligner a into a sparse csc matrix
    aligned_descale=True will descale so that output is purely a rotation matrix
    aligned_negs: options 'abs' (default), 'zero', 'leave'. What to do with negative values in R matrix. 'abs' to use absolute value. 'zero' to make it zero. 'leave' to leave as is.
    For RidgeAlignment, R.coef_ is (ntargetverts,nsourceverts). For all others, R is (nsourceverts,ntargetverts). That is, given X(nsamples,nsourceverts), X is transformed with XR by the transform method.
    """  
    if fa_sparse:   
        from scipy.sparse import lil_matrix
        mat=lil_matrix((59412,59412),dtype=np.float32)
    for i in range(len(a.fit_)):
        pairwise_method=type(a.fit_[0])
        if pairwise_method in [fmralign.alignment_methods.ScaledOrthogonalAlignment , fmralign.alignment_methods.OptimalTransportAlignment,fmralign.alignment_methods.Hungarian]:
            R_i=a.fit_[i].R.T
            if aligned_descale:
                scale=a.fit_[i].scale
                R_i /= scale
        elif pairwise_method==fmralign.alignment_methods.RidgeAlignment:
            R_i=a.fit_[i].R.coef_  
        if aligned_negs=='abs': 
            R_i=np.abs(R_i)
        elif aligned_negs=='zero':
            R_i[R_i<0]=0                 
        if fa_sparse: mat[slices[i],slices[i]]=R_i
        else: a.fit_[i].R=R_i        
    
    if fa_sparse:
        matr=mat.tocsc()
        return matr
    else:
        return a

def all_subject_pairs(aligned_method,n_subs):
    #Get an iterator for all possible subject pairs
    if aligned_method=='template': return itertools.combinations(range(n_subs),2)
    elif aligned_method=='pairwise': return itertools.permutations(range(n_subs),2) #consider sub X aligned to Y and sub Y separately

def get_key(aligned_method,subs,nD,nR):
    """
    Subfunction for get_aligned_block
    """
    if aligned_method=='template':
        key=nR #here, the key means the element number for accessing a2, a list of sparse arrays
    elif aligned_method=='pairwise':
        key=f"{subs[nD]}-{subs[nR]}" #here, the key means a string key e.g. '102311-102816', for accessing dictionary a2
    return key

def get_nyR_nonscrambled(nxD,nyD,aligned_method):
    if aligned_method=='template':
        return nyD #nonscrambled is when nyR==nyD (each sub aligned with their own aligner)
    elif aligned_method=='pairwise':
        return nxD #nonscrambled is when nyR==nxD 

def get_nyR_unaligned(nxD,nyD,aligned_method):
    if aligned_method=='template': 
        return nxD #nyR==nxD is unaligned (both subs aligned with same subs' aligner)
    elif aligned_method=='pairwise': 
        return nyD #nyR==nyD is unaligned (target same as source)  

def fix(datarow,nyR_nonscrambled):
    nonscrambled_value=datarow[nyR_nonscrambled]
    datarow[nyR_nonscrambled]=np.nan #set unscrambled elements to nan
    newdatarow=[i-nonscrambled_value for i in datarow]
    return newdatarow  

def subtract_nonscrambled_from_a(a_original):
    #Haven't adjusted this for aligned_method=='pairwise'
    """
    """
    n_subs=a_original.shape[0]
    a=np.copy(a_original)
    nparcs=a.shape[-1]
    for nD in range(n_subs):
        if a.ndim==3:
            for nblock in range(nparcs):
                a[nD,:,nblock]=fix(a[nD,:,nblock],nD)
        elif a.ndim==4: #from ident_grouped_type=='perparcel'
            for i,j in itertools.product(range(nparcs),range(nparcs)):
                if not(np.isnan(a[0,1,i,j])):
                    a[nD,:,i,j] = fix(a[nD,:,i,j],nD)
    return a
        
def subtract_nonscrambled_from_z(z_original):
    """
    Subtract nonscrambled correlations from scrambled correlation, and set nonscrambled elements to nan
    Template: For each subject X diffusion map aligned with X's func, subject Y diffusion map aligned with subject Z's func, and parcelxparcel block... find difference from the correlation using Y diff map aligned with Y's func..
    Pairwise: Difference between (correlation between sub X's diffusion map and sub Y diff map aligned to sub X) and (correlation between sub X's diffusion map and sub Y diff map aligned to sub Z)
    """  
    n_subs_template=z_original.shape[0]
    n_subs_test=z_original.shape[1]  
    z=np.copy(z_original)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nxD,nyD in itertools.product(range(n_subs_template),range(n_subs_test)):
            if z[nxD,nyD,0,0] != np.nan:
                nyR_nonscrambled = nyD          
                if z_original.ndim==4:          
                    for nblock in range(z.shape[3]):
                        z[nxD,nyD,:,nblock]=fix(z[nxD,nyD,:,nblock],nyR_nonscrambled)                   
                elif z_original.ndim==3:
                    z[nxD,nyD,:]=fix(z[nxD,nyD,:],nyR_nonscrambled)
    return z       

def index(list_of_items,value):
    #return the index of 'value' in a given list, otherwise return None
    try:
        ind = list_of_items.index(value)
    except ValueError:
        ind=None
    return ind

def unaligned_to_nan(z_original,subs_test,subs_template):
    #Template: set elements with nyR==nxR (aligning both subs with same aligner, which can be thought of as not aligning, but having same smoothing effects as aligning) to become nan
    #Pairwise: set elements with nyR==nyD (neither subs' diffusion maps are actually aligned at all) to become nan
    n_subs_template=z_original.shape[0]
    n_subs_test=z_original.shape[1]  
    z=np.copy(z_original)
    
    for nxD,nyD in itertools.product(range(n_subs_template),range(n_subs_test)):
        if z[nxD,nyD,0,0] != np.nan:
            #nyR_unaligned = get_nyR_unaligned(nxD,nyD,aligned_method)
            nyR_unaligned = index(subs_test,subs_template[nxD]) 
            #assumes aligned_method=='template'
            #Takes nxD from subs_template based indexing to subs_test based indexing
            
            if nyR_unaligned is not None:
                if z.ndim==5:
                    z[nxD,nyD,nyR_unaligned,:,:]=np.nan
                elif z.ndim==4:
                    z[nxD,nyD,nyR_unaligned,:]=np.nan
                elif z.ndim==3:
                    z[nxD,nyD,nyR_unaligned]=np.nan
    return z   

def get_unscram_ranks(ranks,aligned_method):
    """
    get rank of unscrambled compared to scrambled. Large numbers mean high rank (more correlation). But the largest rank will almost always go to unaligned, which is nyR==nxR for 'template', or nyR=nyD for 'pairwise'
    ranks is array(nxD,nyD,nyR)
    ranks is a list of length the rank of each unscrambled among the scrambleds
    """
    n_subs_template=ranks.shape[0]
    n_subs_test=ranks.shape[1]  
    ranks_nonscrambled=[]
    for nxD,nyD in itertools.product(range(n_subs_template),range(n_subs_test)):
        if ranks[nxD,nyD,0] != np.nan:
            datarow=ranks[nxD,nyD,:]
            nyR_nonscrambled=get_nyR_nonscrambled(nxD,nyD,aligned_method)
            ranks_nonscrambled.append(datarow[nyR_nonscrambled])
    return ranks_nonscrambled

def array_to_flat(array,get_upper_triangular_elems=False, get_lower_triangular_elems=False):
    indices=[np.array([],dtype=int) , np.array([],dtype=int)]
    if get_upper_triangular_elems:
        indices=[np.concatenate((i,j)) for i,j in zip(indices,np.triu_indices_from(array,1))]
    if get_lower_triangular_elems:
        indices=[np.concatenate((i,j)) for i,j in zip(indices,np.tril_indices_from(array,-1))]
    return array[tuple(indices)]

def collapse(array,aligned_method):
    if aligned_method=='template':
        return np.squeeze(np.dstack([array_to_flat(array[:,:,i],get_upper_triangular_elems=True) for i in range(array.shape[2])])) #collapse n x n x k array into (n*(n-1)/2) x k array (gets upper triangular elems only)
    elif aligned_method=='pairwise':
        return np.squeeze(np.dstack([array_to_flat(array[:,:,i],get_upper_triangular_elems=True,get_lower_triangular_elems=True) for i in range(array.shape[2])])) #collapse n x n x k array into (n*(n-1)) x k array (gets elems not on diagonal)

def values2ranks(x,axis=2,kind='quicksort'):
    """
    Return same-sized array with ranks along a particular axis. Large values have higher ranks
    """
    x2=x.argsort(axis=axis,kind=kind).argsort(axis=axis,kind=kind).astype(np.float32)
    x2[np.isnan(x)]=np.nan #restore any nans in the original
    return x2+1 #increment rank 0 to 1

def prep(y,include_axis_2=True):

    x=y.ravel()
    s0,s1,s2=y.shape[0],y.shape[1],y.shape[2]
    if include_axis_2:
        X=np.zeros((y.size,s0+s1+s2+1))
    else:
        X=np.zeros((y.size,s0+s1+1))
    X[:,-1]=1
    num=0
    for i in range(s0):
        for j in range(s1):
            for k in range(s2):
                X[num,i]=1
                X[num,s0+j]=1
                if include_axis_2:
                    X[num,s0+s1+k]=1
                num += 1
    return X,x
def regress(y,include_axis_2=True):
    #Linear regression where each dimension of y is a different categorical variable, and the elements in y are the values of the single dependent variable. Returns beta weights and residuals
    
    if y.ndim==2:
        return np.squeeze(regress(np.expand_dims(y,axis=2)))
    else:
        X,x = prep(y,include_axis_2=include_axis_2)

        resids_full=np.zeros(x.shape,dtype=x.dtype)
        nans=np.isnan(x) #this section ignores tha nans
        x=x[~nans]
        X=X[~nans,:]      

        beta = np.linalg.lstsq(X,x,rcond=None)[0]
        x_pred = X @ beta
        resids = x - x_pred
        
        resids_full[~nans]=resids
        resids_full[nans]=np.nan #put the nans back
        
        y_resids = resids_full.reshape(y.shape)
        return y_resids

def full(x_ranks):
    """
    Input is 3-dim or 4-dim array, upper triangular in axes 0 and 1 with nans elsewhere. Make the matrix 'full' by transposing the upper triangular part
    """
    w2=np.copy(x_ranks)
    w2[np.isnan(w2)]=0 #change nans to zeros       
    if x_ranks.ndim==3:
        w3 = w2 + np.transpose(w2,(1,0,2)) #make it symmetric
    elif x_ranks.ndim==4:
        w3 = w2 + np.transpose(w2,(1,0,2,3))  
    elif x_ranks.ndim==5:
        w3 = w2 + np.transpose(w2,(1,0,2,3,4))
    assert((x_ranks==0).sum()==0) #check that original array had no zeros
    w3[w3==0]=np.nan #convert any new zeros (formerly nans on diagonal) back to nans   
    return w3    

def identifiability(mat):
    """
    If input is 2-dim array, returns percentage of dim0 for which the diagonal element (dim1==dim0) was the largest. If input is 3-dim, returns the percentage of dim0*dim2 for which the diagonal element (dim1==dim0) was the largest.
    
    Input is array(nyD,nyR) or array(nyD,nyR,nblocks) of ranks. For pairwise, do full(minter_ranks).mean(axis=0) to get appropriate input
    Returns percentage of Diffusion (nyD) for whom the matching Functional (nyR) was identifiable
    For a given nyD, we know how each nyR ranks in improving correlations with nxD*nxR (averaged across nxDs). The 'identified' nyR has the highest mean rank.              
    So, the 'chosen' Functional nyR is the one that tends to have the highest correlation with other subjects when paired with diffusion array nyD
    """
    ndim=mat.ndim
    
    matsort=np.argsort(mat,axis=1) #sort along axis 1, smallest to biggest
    if ndim==2:
        identified_F = matsort[:,-1] #For each given dim0/nyD, list the best dim1/nyR
    elif ndim==3:
        identified_F = matsort[:,-1,:]
    n_subs = identified_F.shape[0]        
    if ndim==2:
        func = lambda d: 100*np.sum([d[i]==i for i in range(n_subs)])/n_subs
        identifiability_F = func(identified_F)
    elif ndim==3:
        ideal_case = np.tile(np.arange(0,n_subs).reshape(n_subs,1),(1,identified_F.shape[1]))
        correct_id = ideal_case==identified_F
        identifiability_F = 100*np.sum(correct_id,axis=0) / correct_id.shape[0]
    return identifiability_F

def nonnans(array):
    #count non-nan elements in given array
    return (~np.isnan(array)).sum()
def count_nonnans(array,func):
    return 100* (func(array)).sum() / (nonnans(array))
def count_negs(array):
    return count_nonnans(array, lambda x: x<0)


def plotter(axis,x,aligned_method,show_same_aligner,title,subs_test,subs_template,drawhorzline=False): 
    n_subs_template = x.shape[0]
    n_subs_test = x.shape[1]
    row=0    
    for nxD,nyD in itertools.product(range(n_subs_template),range(n_subs_test)):
        if subs_template[nxD] != subs_test[nyD]:
            row+=1
            nxR = nxD
            for nyR in range(n_subs_test):
                nyR_unaligned = index(subs_test,subs_template[nxD]) #assumes 'template' method
                if (nyR != nyR_unaligned) or show_same_aligner:
                    if nyR==get_nyR_nonscrambled(nxD,nyD,aligned_method): 
                        #nyD==nyR: #D and R belong to same pt
                        marker='r.'
                    elif nyR==nyR_unaligned:
                        #nyR==nxR: #both Rs belong to same pt 
                        marker='g.'
                    else: #D and R belong to diff pts
                        marker='b.'
                    axis.plot(row,x[nxD,nyD,nyR],marker)                     
    axis.set_title(title)          
    axis.get_xaxis().set_ticks([])
    axis.set_xlabel('Subject pairs')
    axis.set_ylabel('Correlation')
    if drawhorzline:
        axis.axhline(0) #draw horizontal line


def plot_id(axis,x,title='title'):
    colors=['tab:blue','tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','k','m','b']
    n_subs = x.shape[0]
    for nyD in range(n_subs):
        for nyR in range(n_subs):
            if nyD==nyR: marker='r'
            else:
                #marker='b'
                marker=colors[nyR%len(colors)]
            axis.scatter(nyD+1, x[nyD, nyR], s=20, c=marker)
    axis.set_title(title)          
    axis.get_xaxis().set_ticks([])
    axis.set_xlabel('Subjects (connectome)')
    axis.set_ylabel('Score (functional data)')

