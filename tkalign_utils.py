import re
import numpy as np
from scipy import sparse
import hcpalign_utils as hutils
from hcpalign_utils import memused, sizeof, ospath
import matplotlib.pyplot as plt
import warnings
import itertools

def correlate_block(X,Y,corr_args=['pearson','ravel']):
    """
    Given two 2D arrays, return the correlation between their vectorized versions?
    Why is np.nanmean there?
    Parameters:
    -----------
    X: np.array
        2-dim array
    Y: np.array
        2-dim array
    corr_args: list
        ['pearson','ravel'] or ['spearman','ravel']    
    """
    if corr_args==['pearson','ravel']: #4.5s
        ccs=[np.corrcoef(X.ravel(),Y.ravel())[0,1]]   
    elif corr_args==['spearman','ravel']: #21s
        from scipy.stats import spearmanr as sp
        ccs,_=sp(X.ravel(),Y.ravel())
    return np.nanmean(ccs)

def extract_nsubs_alignpickles1(string):
    """
    Given a filename from intermediates/alignpickles, extract the number of subjects
    """
    pattern=r'hcpalign\w*_(\d+)-\d+-\d+_TF_0_0_0_\w*'
    match=re.search(pattern,string)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_sub_range_alignpickles1(string):
    nsubs = extract_nsubs_alignpickles1(string)
    return range(0,nsubs)

alignpickles3_pattern = r'A(mov|res|movfc|resfc|diff)(t|f)(\d*)(t|f)(\d*)(.*?)_([A-Z]\d*)_T(.*?)sub(\d*)to(\d*)_(.*?)_(.*)'
alignpickles3_dictionary = {'align_with':1,'MSMAll':2,'runs_string':3,'clean':4,'fwhm':5,'FC_args':6,'parcellation_string':7,'template_data':8,'template_sub_start':9,'template_sub_end':10,'template_making_string':11,'details':12}

def replace_S300_alignpickles3(string):
    #if input is 'Amovf0t0_R5_Tmovf0t0sub0to2_RG0ffrr_TempScal_gam0.3', return 'Amovf0t0_S300_Tmovf0t0sub0to2_RG0ffrr_TempScal_gam0.3'

    match = re.search(alignpickles3_pattern, string)
    result = f"A{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}{match.group(5)}{match.group(6)}_S300_T{match.group(8)}sub{match.group(9)}to{match.group(10)}_{match.group(11)}_{match.group(12)}"
    return result   

def extract_alignpickles3(string,what):
    """
    Given a filename from intermediates/alignpickles3, extract some content
    Parameters:
    -----------
    string: filename
    what: string
    """

    match = re.search(alignpickles3_pattern, string)
    if match:
        value = match.group(alignpickles3_dictionary[what])
        if what in ['MSMAll','clean']:
            str2logical={'t':True,'f':False}
            value = str2logical[value]
        elif what in ['fwhm','template_sub_start','template_sub_end']:
            value = int(value)
        return value
    else:
        return None

def extract_tkalign_corrs2(string):
    """
    Given a filename from intermediates/tkalign_corrs2, extract some content
    Parameters:
    -----------
    string: filename
        e.g. 'Amovf0123t0_S300_Tmovf0123t0sub30to40_G0ffrr_TempScal_gam0.2_B100la40-90_Dtracks_5M_1M_end_3mm_3mm_RDRT_S40_40-50'
    """

    tkalign_corrs2_pattern = r'(.*)_B(\d+)(la|fr|al|fe)(\d+)-(\d+)_D(.*)_(\d*)mm_(\d*)mm_(.*)_S(\d*)_(\d*)-(\d*)'
    tkalign_corrs2_dictionary = {'alignfile':1,'nblocks':2,'block_choice_str':3,'subs_blocks_range_start':4,'subs_blocks_range_end':5,'tckfile_prefix':6,'pre_hrc_fwhm':7,'post_hrc_fwhm':8,'howtoalign':9,'nsubs_for_template_connectome':10,'subs_test_range_start':11,'subs_test_range_end':12}

    match = re.search(tkalign_corrs2_pattern, string)
    newdict = {what: match.group(tkalign_corrs2_dictionary[what]) for what in tkalign_corrs2_dictionary.keys()}
    for key in ['nblocks','subs_blocks_range_start','subs_blocks_range_end','pre_hrc_fwhm','post_hrc_fwhm','nsubs_for_template_connectome','subs_test_range_start','subs_test_range_end']:
        newdict[key] = int(newdict[key])
    nblocks = newdict['nblocks']
    block_choice_dict = {'la':'largest', 'fr': 'fromsourcevertex', 'al':'all','fe':'few_from_each_vertex'}
    block_choice = block_choice_dict[newdict['block_choice_str']]
    howtoalign = newdict['howtoalign']
    pre_hrc_fwhm = newdict['pre_hrc_fwhm']
    post_hrc_fwhm = newdict['post_hrc_fwhm']
    alignfiles = [newdict['alignfile']]
    tckfile = newdict['tckfile_prefix']+ '.tck'
    subs_test_range_start = newdict['subs_test_range_start']
    subs_test_range_end = newdict['subs_test_range_end']
    return nblocks,block_choice,howtoalign,pre_hrc_fwhm,post_hrc_fwhm,alignfiles,tckfile, subs_test_range_start, subs_test_range_end

def get_extremal_indices(a,n,kind='largest'):
    #Get indices of the top n elements of a
    flattened = a.flatten()
    valid_indices = np.argwhere(~np.isnan(flattened))
    sorted_indices = np.argsort(flattened[valid_indices.flatten()])
    # Get indices of the top n elements
    if kind=='largest':
        top_indices = valid_indices.flatten()[sorted_indices][-n:]
    elif kind=='smallest':
        top_indices = valid_indices.flatten()[sorted_indices][:n]
    indices = np.stack(np.unravel_index(top_indices, a.shape)) # Convert flat indices to 3D indices
    return indices

def new_func(c,subs_inds, alignfile, aligner2sparsearray, aligner_descale, aligner_negatives, sorter, slices, smoother_post, groups):
    from joblib import Parallel, delayed
    import pickle
    print_times = True
    fa={} #fa[group] is a list of functional aligners
    for group in groups:
        func1 = lambda sub_ind: pickle.load(open(ospath(f'{hutils.intermediates_path}/alignpickles3/{alignfile}/{hutils.all_subs[sub_ind]}.p'), "rb" ))
        if print_times: print(f'{c.time()}: Load aligners {group}', end=", ")
        all_aligners = Parallel(n_jobs=-1,prefer='threads')(delayed(func1)(sub_ind) for sub_ind in subs_inds[group]) #load each time because func(i) will modify arrays in all_aligners
        func2 = lambda aligner: get_template_aligners(aligner,slices,sorter,aligner2sparsearray=aligner2sparsearray,aligner_descale=aligner_descale,aligner_negatives=aligner_negatives,smoother=smoother_post)
        if print_times: print(f'{c.time()}: get_template_aligners {group}', end=", ")
        fa[group] = Parallel(n_jobs=-1,prefer='threads')(delayed(func2)(aligner) for aligner in all_aligners)
        if print_times: print(f'{c.time()}: downsample aligners {group}', end=", ")
        fa[group]=Parallel(n_jobs=-1,prefer='threads')(delayed(hutils.aligner_downsample)(estimator) for estimator in fa[group])
    return fa,all_aligners

def replace_half_with_negatives(X):
    #Given a 2D array containing positive values, replace a random half of the values with their negatives
    import random
    assert(np.all(X>=0))
    flat_X = X.flatten()
    num_to_negate = len(flat_X) // 2
    indices_to_negate = random.sample(range(len(flat_X)), num_to_negate)
    for idx in indices_to_negate:
        flat_X[idx] = -flat_X[idx]
    return flat_X.reshape(X.shape)

def randomise_but_preserve_row_col_sums_template(x):
    #Given TemplateAlignment object, for each estimator, run randomise_but_preserve_row_col_sums
    for nsubject in range(len(x.estimators)):
        for nparc in range(len(x.estimators[0].fit_)):
            R = x.estimators[nsubject].fit_[nparc].R
            Rabs = np.abs(R)
            Rabs_rand = randomise_but_preserve_row_col_sums(Rabs)
            R_rand = replace_half_with_negatives(Rabs_rand)
            x.estimators[nsubject].fit_[nparc].R = R_rand
    return x

def randomise_but_preserve_row_col_sums(r0):
    """
    Given a 2D array r0, randomise its elements while preserving the sum of each row and column.
    """
    assert(r0.min()>0)
    temp = r0.copy()
    sums_axis_0 = r0.sum(axis=0)
    sums_axis_1 = r0.sum(axis=1)
    sums_axis_1_normed = sums_axis_1 / sums_axis_1.sum()
    for j in range(r0.shape[1]):
        temp[:,j] = sums_axis_1_normed * sums_axis_0[j]
    return temp


def get_rowsums(fats,align_labels,axis=0):
    """
    Return a list (subjects) of spatial maps of functional aligner row (or column) sums 
    Parameters:
    fats: list (nsubjects) of functional aligners
    axis: whether to calculate row-sums or column-sums on functional aligner
    """
    nsubjects = len(fats)
    nparcs = len(fats[0].fit_)
    rowsums = [np.zeros(59412,dtype=np.float32) for i in range(nsubjects)] 
    for subject in range(nsubjects):
        for nparc in range(nparcs):
                R = divide_by_Frobenius_norm(fats[subject].fit_[nparc].R)
                #np.fill_diagonal(R,0) #TRY THIS
                sum = np.abs(R).sum(axis=axis)
                #sum = np.diagonal(R) #TRY THIS  
                rowsums[subject][align_labels==nparc] = sum
    return rowsums



def ident_from_data(structdata,funcdata,nsubjects,nparcs,nblocks,blocks,align_labels,arn):

    #fig,axs = plt.subplots(nsubjects,5)
    cc = np.zeros((nsubjects,nsubjects,nparcs),dtype=np.float32) #corrs bw structdata from one subject and funcdata from another subject, for a given parcel
    for subject_D in range(nsubjects):
        for subject_R in range(nsubjects):
            for nparc in range(nparcs):
                structdatum = structdata[subject_D][align_labels==nparc]
                funcdatum = funcdata[subject_R][align_labels==nparc]
                correlation = np.corrcoef(structdatum,funcdatum)[0,1]
                cc[subject_D,subject_R,nparc] = correlation

                """
                if subject_D==subject_R and nparc<5: #TRY THIS. Scatterplot of structdatum and funcdata. Rows are subjects, columns are parcels
                    axs[subject_D,nparc].scatter(structdatum,funcdatum,1)
                    axs[subject_D,nparc].set_title(f"r={correlation:.2f}")
                """
    #fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.2, wspace=0.2)
    #plt.show(block=False)

    
    corrs = np.diagonal(cc).T #corrs bw structdata and funcdata for each subject (same subject for both) and parcel
    corrsm = np.mean(corrs,axis=0)

    #Make a version of cc with nan diagonals
    ccx = cc.copy()
    for i in range(cc.shape[0]):
        for j in range(cc.shape[2]):
            ccx[i,i,j] = np.nan
    ccxr = np.ravel(ccx)
    ccxr2=ccxr[~np.isnan(ccxr)] 

    """
    block_sfc = np.zeros(blocks.shape[1],dtype=np.float32) #make a version of corrsm where we have a value for each block (by averaging the two parcels' values)
    for nblock in range(blocks.shape[1]):
        i,j = blocks[0,nblock], blocks[1,nblock]
        block_sfc[nblock] = np.mean([corrsm[i],corrsm[j]])

    sfc = np.zeros((nsubjects,nblocks),dtype=np.float32) #like above, but a version of corrs
    for subject in range(nsubjects):
        for nblock in range(blocks.shape[1]):
            i,j = blocks[0,nblock], blocks[1,nblock]
            sfc[subject,nblock] = np.mean([corrs[subject,i],corrs[subject,j]])
    """

    #element [i,j,k] holds the correlation between subject i's structdata and subject j's funcdata, for block k
    ccb = np.zeros((nsubjects,nsubjects,nblocks),dtype=np.float32)
    for subject_D in range(nsubjects):
        for subject_R in range(nsubjects):
            for nblock in range(blocks.shape[1]):
                i,j = blocks[0,nblock], blocks[1,nblock]
                ccb[subject_D,subject_R,nblock] = np.mean([cc[subject_D,subject_R,i],cc[subject_D,subject_R,j]])
    ccbn = subtract_nonscrambled_from_a(ccb)
    ccbrn = subtract_nonscrambled_from_a(regress(ccb))
    print(f'AV R {count_negs(ccbn):.1f}% of (sub-pairs)*blocks for connectome strength based identification')   
    print(f'AV R {count_negs(ccbrn):.1f}% of (sub-pairs)*blocks for connectome strength based identification (+reg)')   

    #In most parcels, those subject j's whose aligner strength was correlated most with subject i's connectome strength, were those subject pairings where tkalign works better :(
    ww=[] 
    fig,ax=plt.subplots(4)
    import pandas as pd
    for nblock in range(blocks.shape[1]):
        x=arn[:,:,nblock]
        y=ccbn[:,:,nblock]
        df = pd.DataFrame([x.ravel(),y.ravel()]).T
        ww.append(df.corr().iloc[0,1])


                
    # for most parcels and subjects, their own 'connectome strength' is correlated with their own 'aligner strength', while other subjects' are not. This was true within most parcels. 
    ax[0].hist(corrs.ravel())
    ax[0].set_title(f'matched corrs, {100*np.sum(corrs>0)/np.size(corrs):.0f}% > 0')
    ax[0].set_xlim([-.8,.8])
    ax[1].hist(ccxr2)
    ax[1].set_title(f'mismatchedcorrs, {100*np.sum(ccxr2>0)/np.size(ccxr2):.0f}% > 0')
    ax[1].set_xlim([-.8,.8])
    ax[2].imshow(cc.mean(axis=-1))
    ax[2].set_title('cc mean across parcs')
    ax[3].hist(ww)
    ax[3].set_title('Assoc bw arn and ccbn (within parc)')
    fig.tight_layout()
    plt.show(block=False)

    """

    p.plot(corrs[1,:] @ align_parc_matrix)
    p.plot(corrsm @ align_parc_matrix)     
    """

def get_tck_file():
    import socket
    hostname=socket.gethostname()
    if hostname=='DESKTOP-EGSQF3A': #home pc
        tckfile= 'tracks_5M_1M_end_sift2act.tck' #'tracks_5M_sift1M_200k.tck' #'tracks_5M_sift1M_200k.tck','tracks_5M.tck' 
    else: #service workbench
        tckfile='tracks_5M_1M_end.tck'
    return tckfile

def divide_by_Frobenius_norm(array):
    norm=np.linalg.norm(array)
    return array/norm

def get_smoother(fwhm):
    return sparse.load_npz(ospath(f'{hutils.intermediates_path}/smoothers/100610_{fwhm}_0.01.npz')).astype(np.float32)

"""
def smooth_aligners(list_of_aligners,smoother,sorter,slices):
    for i in range(len(slices)):
        for j in range(len(list_of_aligners)):
            smoother_parcel = smoother[slices[i],slices[i]]
            R_parcel = list_of_aligners[j].fit_[i].R
            list_of_aligners[j].fit_[i].R = smoother_parcel @ R_parcel# @ (smoother_parcel.T) #could also append #smoother_parcel which is equivalent to smoothing high-res connectome
    return list_of_aligners
"""

def makesorter(labels):
    """
    Given list of unordered labels (59412,), return sorting indices, unsorting indices, and labelslice. labelslice is list containing slice for each label in the sorted list
    Parameters:
    -----------
    labels: array of labels (59412,)    
    Returns:
    -----------
    sorter: array of indices to sort labels. e.g. labels[sorter] is sorted
    unsorter: array of indices to go from sorted labels to original ordering
    labelslice: list of slices for each label in the sorted list, e.g. [slice(0,767),slice(767,1534),...)]
    """

    if type(labels[0])==np.ndarray: #searchlight method
        #sorter and unsorter will be identity transformations
        sorter=np.array(range(len(labels))) #all integers from 0 to 59411
        unsorter=np.array(range(len(labels)))
        labelslice=None
    else: #default
        unique_labels=np.unique(labels)
        sorter=np.argsort(labels,kind='stable')
        unsorter=np.argsort(sorter)
        labelsort=labels[sorter]
        labelslice=np.zeros(len(unique_labels),dtype=object)
        for i in range(len(unique_labels)):
            where=np.where(labelsort==unique_labels[i])[0]
            labelslice[i]=slice(where[0],where[-1]+1)
    return sorter,unsorter,labelslice
    
def load_a(save_path):
    temp=np.load(save_path,allow_pickle=True)
    blocks=temp[()]['blocks']
    a=temp[()]['a']
    return blocks,a

def get_blocks(c,tckfile, MSMAll, sift2, align_parc_matrix, subs, block_choice,nblocks,parcel_pair_type,align_labels,nparcs,type_rowcol,par_prefer_hrc):
    """
    Return blocks
    """
    if type(align_labels[0])==np.ndarray: #searchlight method
        """
        Take mean high-res connectome, then 5mm smooth it. For each source vertex, calculate no. of streamlines going to each S300 parcel. For those S300 parcels which the source vertex belongs to, set no. of streamlines to zero. Then select the nblocks largest target S300 parcels, for each source vertex 
        """
        print_times = True
        if print_times: print(f'{c.time()}: BLOCKS get blocks start', end=", ")
        assert(block_choice=='few_from_each_vertex')
        connectomes_highres = hutils.get_highres_connectomes(c,subs,tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefer_hrc,n_jobs=-1) 
        print(f'{c.time()}: BLOCKS smooth connectomes', end=", ")
        connectomes_highres_sum = np.sum(connectomes_highres)
        from Connectome_Spatial_Smoothing import CSS as css    
        fwhm=5
        smoother=sparse.load_npz(ospath(f"{hutils.intermediates_path}/smoothers/100610_{fwhm}_0.01.npz"))
        connectomes_highres_sum=css.smooth_high_resolution_connectome(connectomes_highres_sum,smoother)
        print(f'{c.time()}: BLOCKS Find biggest blocks', end=", ")
        align_parc_matrix_S300 = hutils.parcellation_string_to_parcmatrix('S300')
        lines = align_parc_matrix_S300 @ connectomes_highres_sum #streamlines_per_parcel_and_vertex
        lines[align_parc_matrix_S300]=0
        #lines.eliminate_zeros()
        lines = lines.todense()
        inds = np.argpartition(lines,-nblocks,axis=0)[-nblocks:,:]
        temp0a=np.array([i for i in range(nparcs)])
        temp0b=np.tile(temp0a,(nblocks,1))

        print(f'{c.time()}: BLOCKS truncate', end=", ")
        truncate_to=100
        if truncate_to is not None:
            print(f'TRUNCATING blocks to first {truncate_to} source vertices')
            temp0b=temp0b[:,0:truncate_to]
            inds=inds[:,0:truncate_to]
            nparcs=truncate_to

        temp1=temp0b.ravel()   
        temp2 = inds.ravel()

        blocks=np.vstack([temp1,temp2]) #first row is searchlight parcel indices, second row is S300 parcel indices

    else: #default
        parcellated_connectomes = get_parcellated_connectomes(c,tckfile, MSMAll, sift2, align_parc_matrix, subs, par_prefer_hrc)    
        hps, hpsx, hpsxa = reduce_parcellated_connectomes(parcellated_connectomes)    
        blocks = get_blocks_from_parcellated_connectomes(block_choice, nblocks, parcel_pair_type, align_labels, nparcs, type_rowcol, hps, hpsx, hpsxa)
    return blocks, nparcs

def get_blocks_from_parcellated_connectomes(block_choice, nblocks, parcel_pair_type, align_labels, nparcs, type_rowcol, hps, hpsx, hpsxa):
    if block_choice=='largest': 
        blocks=get_blocks_largest(nblocks,parcel_pair_type,hps,hpsx)
    elif block_choice=='fromsourcevertex':
        blocks=get_blocks_source(align_labels,hpsxa,nblocks,nparcs,type_rowcol)
    elif block_choice=='all':
        blocks=get_blocks_all()
    elif block_choice=='few_from_each_vertex':
        blocks=get_blocks_few_from_each_vertex(nblocks,hpsxa,nparcs)
    return blocks

def get_parcellated_connectomes(c,tckfile, MSMAll, sift2, align_parc_matrix, subs, par_prefer_hrc):
    """
    Return parcellated connectomes
    """
    from Connectome_Spatial_Smoothing import CSS as css
    connectomes_highres = hutils.get_highres_connectomes(c,subs,tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefer_hrc,n_jobs=-1) 
    parcellated_connectomes=[css.downsample_high_resolution_structural_connectivity_to_atlas(hrs, align_parc_matrix) for hrs in connectomes_highres] 
    del connectomes_highres
    return parcellated_connectomes

def reduce_parcellated_connectomes(hp): 
    """
    Given a list of sparse arrays, return the following:
    Return:
    -------
    hps: elementwise sum of all sparse arrays in list
    hpsx: remove diagonal elements from hps
    hpsxa: hpsx converted to array
    """      
    hps=sum(hp)
    hpsx=hps.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        hpsx.setdiag(0)
        hpsx.eliminate_zeros() 
    hpsxa=hpsx.toarray()
    return hps,hpsx,hpsxa

def get_blocks_largest(nblocks,parcel_pair_type,hps,hpsx):
    if parcel_pair_type=='intra':
        temp=hutils.indices_of_largest_values_in_array(sparse.csr_matrix.diagonal(hps),nblocks)  
        blocks = np.vstack([temp,temp])
    elif parcel_pair_type=='inter': 
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
def get_blocks_few_from_each_vertex(nblocks,hpsxa,nparcs):
    nblocks_per_parcel=nblocks
    inds=np.argpartition(hpsxa,-nblocks_per_parcel,axis=0)[-nblocks_per_parcel:,:]
    temp2=inds.ravel()
    temp0a=np.array([i for i in range(nparcs)])
    temp0b=np.tile(temp0a,(nblocks_per_parcel,1))
    temp1=temp0b.ravel()   
    blocks=np.vstack([temp1,temp2])
    return blocks

def get_template_aligners(a,slices,sorter=None,aligner2sparsearray=False,aligner_descale=False,aligner_negatives='abs',smoother=None):
    """
    Get alignment transformations from a SurfaceAlignment object, and perform some processing. For RidgeAlignment, R.coef_ is (ntargetverts,nsourceverts). For all others, R is (nsourceverts,ntargetverts). That is, given X(nsamples,nsourceverts), X is transformed with XR by the transform method.
    Parameters:
    -----------
    a: SurfaceAlignment object
    slices: list of slices for each parcel in the ordered list
    aligner2sparsearray: if True, convert each aligner to a sparse array
    aligner_descale: if True, descale each aligner so that output is purely a rotation matrix
    aligner_negatives: options 'abs' (default), 'zero', 'leave'. What to do with negative values in R matrix. 'abs' to use absolute value. 'zero' to make it zero. 'leave' to leave as is.
    smoother: sparse array from function get_smoother
    """
    import fmralign
    if aligner2sparsearray:   
        from scipy.sparse import lil_matrix
        mat=lil_matrix((59412,59412),dtype=np.float32)
    for i in range(len(a.fit_)):
        pairwise_method=type(a.fit_[0])
        if pairwise_method in [fmralign.alignment_methods.ScaledOrthogonalAlignment , fmralign.alignment_methods.OptimalTransportAlignment,fmralign.alignment_methods.Hungarian]:
            R_i=a.fit_[i].R.T
            if aligner_descale:
                scale=a.fit_[i].scale
                R_i /= scale
        elif pairwise_method==fmralign.alignment_methods.RidgeAlignment:
            R_i=a.fit_[i].R.coef_  
        if aligner_negatives=='abs': 
            R_i=np.abs(R_i)
        elif aligner_negatives=='zero':
            R_i[R_i<0]=0
        if (smoother is not None) and (smoother.nnz!=59412):
            smoother_parcel = smoother[slices[i],slices[i]]
            R_i = smoother_parcel @ R_i# @ (smoother_parcel.T)
        if aligner2sparsearray: mat[slices[i],slices[i]]=R_i
        else: a.fit_[i].R=R_i        
    if aligner2sparsearray:
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
    n_subs=a_original.shape[0]
    a=np.copy(a_original)
    nparcs=a.shape[-1]
    for nD in range(n_subs):
        if a.ndim==3:
            for nblock in range(nparcs):
                a[nD,:,nblock]=fix(a[nD,:,nblock],nD)
        elif a.ndim==4: #from ident_grouped_type=='perparcel'
            for i,j in itertools.product(range(a.shape[-2]),range(a.shape[-1])):
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
def regress(y,include_axis_2=False):
    """
    Linear regression where each dimension of y is a different categorical variable, and the elements in y are the values of the single dependent variable. Returns residuals.
    For example, if y has shape (3,4,5), then the dependent variable is a function of 3 categorical variables, each with 3, 4, and 5 levels respectively.
    Parameters:
    -----------
    y: numpy array with 2 to 3 dimensions
    include_axis_2: if True, include axis 2 in the regression. If False, ignore axis 2 (only do axes 0 and 1)
    """
    
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

def ident_plot(X,xlabel,Y,ylabel,reg=True,normed = True, figsize=None,title=None):
    """
    X and Y are lists of vectors. Calculate the correlation between each pair of vectors, and plot the results
    """

    if normed:
        norm = lambda x: hutils.subtract_parcelmean(x,align_labels)
        X = [norm(i) for i in X]
        Y = [norm(i) for i in Y]
    corrs = np.zeros((len(Y),len(X)),dtype=np.float32)
    for i in range(len(Y)):
        for j in range(len(X)):
            corrs[i,j] = np.corrcoef(X[j],Y[i])[0,1]
    if reg:
        corrs = regress(corrs)
    fig,ax=plt.subplots(figsize=figsize)
    cax=ax.imshow(corrs,cmap='cividis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    nsubjects = corrs.shape[0]
    ax.set_xticks(range(nsubjects))
    ax.set_yticks(range(nsubjects))
    ax.set_xticklabels([i+1 for i in range(nsubjects)])
    ax.set_yticklabels([i+1 for i in range(nsubjects)])
    ax.set_title(title)
    fig.colorbar(cax,ax=ax)
    fig.tight_layout()
    return corrs

def get_max_within_parcel_corrs(mov,align_labels,nparcs):
    """
    Given an array of shape (ntimepoints,nvertices), return an array of shape (nvertices,) where each element is the maximum correlation between the timecourse of that vertex and any other vertex in the same parcel
    """
    result = np.zeros(59412,dtype=np.float32)
    for i in range(nparcs):
        array = mov[:,align_labels==i]
        values = np.max(hutils.diag0(np.corrcoef(array.T)),axis=0)
        result[align_labels==i] = values
    return result

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

"""
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
"""

def hist_ravel(axis,x,title='title',vline=0):
    if type(x)==np.ndarray: x=x.ravel()
    axis.hist(x,color='red',alpha=0.5)
    axis.set_title(title)
    axis.axvline(vline) #draw vertical line

def plot_id(axis,x,title='title'):
    colors=['tab:blue','tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','k','m','b']
    n_subs = x.shape[0]
    jitter = 0.15
    jitters = np.random.uniform(-jitter,+jitter,n_subs*n_subs)
    nsubs_nyD = min(10,x.shape[0]) #only show first 10 subjects
    marker_size=5
    index=0
    for nyD in range(nsubs_nyD):
        for nyR in range(n_subs):
            if nyD!=nyR:
                #marker_color=colors[nyR%len(colors)]
                axis.scatter(nyD+1+jitters[index], x[nyD, nyR], s=marker_size, c='b',alpha=0.5)
                index+=1
        #marker_color=colors[nyR%len(colors)]
        axis.scatter(nyD+1, x[nyD, nyD], s=marker_size, c='r')
    axis.set_title(title)          
    axis.get_xaxis().set_ticks([])
    axis.set_xlabel('Subjects (connectome)')
    axis.set_ylabel('Score (functional data)')

