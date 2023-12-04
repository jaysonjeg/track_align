"""
Formerly named tkfunc3
Script to relate high-resolution diffusion connectomes to functional alignment

For all 10 subjects...
Get high-resolution connectomes and smooth them
Get previously calculated template aligners based on movie-viewing fMRI
Calculate similarity between 2 subjects' connectomes aligned with their own aligner, vs. if one of them is aligned with a different subject's aligner.
The difference between 'unscrambled' and 'scrambled' similarities indicates extent to which individual diffusion maps relate to individual functional activations

We use a k-means parcellation to divide high-res connectome into k x k 'blocks'. Intraparcel blocks are those which connect parcel m to parcel m. Interparcel blocks connect parcel m to parcel n.
The functional aligner used the same k-means parcellation

We calculate connectome similarities only WITHIN blocks to ensure that fine resolution individual differences are being considered. Only a subset of all k x k blocks are used. Either we use the most densely connected n blocks, or we use all blocks containing a specific parcel m (e.g. primary visual cortex)
"""
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
"""
import numpy as np
from scipy.stats import spearmanr as sp

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
        ccs,_=sp(X.ravel(),Y.ravel())
    return np.nanmean(ccs)

def correlate_blocks(Xs,Ys,corr_args=['pearson','ravel']):
        #Do correlate_block for each corresponding item in Xs and Ys
        return [correlate_block(X,Y,corr_args=corr_args) for X,Y in zip(Xs,Ys)]

if __name__=='__main__':

    import os, pickle, warnings, itertools
    from scipy import sparse
    import hcpalign_utils as hutils
    from hcpalign_utils import ospath
    import matplotlib.pyplot as plt, tkalign_utils as tutils
    from tkalign_utils import values2ranks as ranks, regress as reg, identifiability as ident, count_negs
    from joblib import Parallel, delayed

    print(hutils.memused())
    c=hutils.clock()

    def func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,plot_type,pre_hrc_fwhm,post_hrc_fwhm,MSMAll, tckfile=None,text=''):
        """
        Parameters:
        -----------
        subs_inds: dict
            keys are 'temp', 'test', 'aligner'. Values are lists of subject indices in hutils.all_subs
        nblocks: int
        alignfile: str
            name of pickle file in intermediates/alignpickles containing functional aligners
        howtoalign: str
            'RDRT','RD','RD+','RT','RT+'
        block_choice: str
            'largest': choose blocks with largest number of streamlines
            'fromsourcevertex': use all blocks involving a single parcel
            'all': use all blocks
            'few_from_each_vertex': for each vertex, use nblocks blocks with largest number of streamlines
        save_file: bool
        load_file: bool
        to_plot: bool
        plot_type: str
            'save_as_html' or 'open_in_browser'
        pre_hrc_fwhm: int
        post_hrc_fwhm: int
        MSMAll: bool
        tckfile: str
            name of tck file in intermediates/highres_connectomes
        text: str
            text to add to end of save_prefix
        """

        ### Parameters 
        parcel_pair_type='inter' #'inter' for inter-parcel connections (default) or 'intra' for intra-parcel connections
        type_rowcol = 'col' # 'col' or 'row'. Only relevant for inferring directionality with block_choice=='fromsourcevertex' 

        get_similarity_average=True #correlations between X aligned with Y, and mean(all W aligned with W). Default True
        aligner2sparsearray=False #bool, default False. Make functional aligner into a sparse array. 
        aligner_descale=False #bool, Default False. make R transformations orthogonal again (doesn't make any difference to correlations). 
        aligner_negatives='abs' #str. What to do with negative values in R matrix. 'abs' to use absolute value (default). 'zero' to make it zero (performance slightly worse). 'leave' to leave as is.

        par_prefer_hrc='threads'  #'threads' (default) or 'processes' for getting high-res connectomes from file

        print(hutils.memused())
        if howtoalign!='RDRT':
            print(f'howtoalign is {howtoalign}')

        if tckfile is None:
            import socket
            hostname=socket.gethostname()
            if hostname=='DESKTOP-EGSQF3A': #home pc
                tckfile= 'tracks_5M_sift1M_200k.tck' #'tracks_5M_sift1M_200k.tck','tracks_5M.tck' 
            else: #service workbench
                tckfile='tracks_5M_1M_end.tck'

        sift2=not('sift' in tckfile) #True, unless there is 'sift' in tckfile


        if block_choice in ['fromsourcevertex','all']: 
            nblocks=0
        elif block_choice in ['all','few_from_each_vertex']: 
            ident_grouped_type='perparcel' #Do identifiability analyses at 'wholebrain' or 'perparcel' level?
        else:
            ident_grouped_type='wholebrain'

        aligned_method='template'

        ### Get parcellation 
        assert(parcellation_string in alignfile)
        align_labels=hutils.parcellation_string_to_parcellation(parcellation_string)
        align_parc_matrix=hutils.parcellation_string_to_parcmatrix(parcellation_string)
        nparcs=align_parc_matrix.shape[0]

        ### Set up subject lists 
        # Variable subs is a dict, keys are 'temp', 'test', and 'aligner'. Values are subject IDs as strings. 'aligner' subjects are used to determine the blocks with the most streamlines. 'temp' subjects are used to make the template connectome. 'test' subjects are aligned either with their own or other test subjects' aligners. Their correlation with the template connectome is calculated.
        groups=['temp','test']
        subs = {group: [hutils.all_subs[i] for i in subs_inds[group]] for group in subs_inds.keys()} 
        #assert( set(subs['temp']).isdisjoint(subs['test']) )

        ### Set up filenames and folders for saving and figures
        save_prefix = f"corrs_{subs_inds['aligner'].start}-{subs_inds['aligner'].stop}_{len(subs['temp'])}s_{subs_inds['test'].start}-{subs_inds['test'].stop}_{tckfile[:-4]}_{pre_hrc_fwhm}mm_{post_hrc_fwhm}mm_{parcellation_string}_{nblocks}b_{block_choice[0:2]}_{howtoalign}{text}"
        #save_prefix = f"corrs_0-{alignfile_nsubs}_{len(subs['temp'])}s_{min(subs_inds['test'])}-{max(subs_inds['test'])}_{tckfile[:-4]}_{pre_hrc_fwhm}mm_{post_hrc_fwhm}mm_{parcellation_string}_{nblocks}b_{block_choice[0:2]}_{howtoalign}{text}"
        #save_prefix = f'r{hutils.datetime_for_filename()}'
        print(save_prefix)

        figures_subfolder=ospath(f'{hutils.results_path}/figures/{save_prefix}') #save figures in this folder
        if to_plot and plot_type=='save_as_html': hutils.mkdir(figures_subfolder)
        p=hutils.surfplot(figures_subfolder,plot_type=plot_type)

        save_folder=f'{hutils.intermediates_path}/tkalign_corrs' #save results data in this folder
        hutils.mkdir(save_folder)
        save_path=ospath(f"{save_folder}/{save_prefix}.npy")
        print(save_path)

        ### Set up smoothing kernels
        sorter,unsorter,slices=tutils.makesorter(align_labels) # Sort vertices by parcel membership, for speed
        smoother_pre = tutils.get_smoother(pre_hrc_fwhm)[sorter[:,None],sorter]
        smoother_post = tutils.get_smoother(post_hrc_fwhm)[sorter[:,None],sorter]

        if load_file and os.path.exists(save_path):
            print('loading f and a')
            blocks,f,a = tutils.load_f_and_a(save_path)
        else:
            print(f'{c.time()}: Get parcellated connectomes and blocks',end=", ")  
            blocks = tutils.get_blocks(c,tckfile, MSMAll, sift2, align_parc_matrix, subs['aligner'], block_choice,nblocks,parcel_pair_type,align_labels,nparcs,type_rowcol,par_prefer_hrc)
            print(f'{c.time()}: Get high-res connectomes',end=", ")  
            hr = {group : hutils.get_highres_connectomes(c,subs[group],tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefer_hrc,n_jobs=-1) for group in groups} # Get high-res connectomes for test and template subjects. hr[group] is a list of sparse arrays, Each array is a connectome for a subject
            print(f'{c.time()}: Reorder connectomes', end=", ")
            hr = {group: [array[sorter[:,None],sorter] for array in hr[group]] for group in groups}
            print(f'{c.time()}: Smooth hrc', end=", ")
            hr={group : hutils.smooth_highres_connectomes(hr[group],smoother_pre) for group in groups}

            ### Get func aligners 
            print(f'{c.time()}: GetAligners', end=", ")
            aligner_file = f'{hutils.intermediates_path}/alignpickles/{alignfile}.p'
            #all_aligners = pickle.load( open( ospath(aligner_file), "rb" )) 
            if aligned_method=='template':                                                    
                fa={} #fa[group] is a list of functional aligners
                for group in groups:
                    all_aligners = pickle.load( open( ospath(aligner_file), "rb" )) #load each time because func(i) will modify arrays in all_aligners            
                    func = lambda nsub: tutils.get_template_aligners(all_aligners.estimators[nsub],slices,aligner2sparsearray=aligner2sparsearray,aligner_descale=aligner_descale,aligner_negatives=aligner_negatives)
                    fa[group] = [func(i) for i in subs_inds[group]]

            def get_aligner_parcel(group,sub_ind,nparcel):
                """       
                Parameters:
                ----------
                group: 'temp' or 'test'
                sub_ind: int
                    index of subject in subs[group]
                nparcel: int
                    index of parcel
                """
                if aligner2sparsearray:
                    return fa[group][sub_ind][slices[nparcel],slices[nparcel]].toarray()
                else:
                    return fa[group][sub_ind].fit_[nparcel].R    
            def get_vals(howtoalign,group,sub_ind_hr,sub_ind_fa,nblock):
                """
                Return the connectome of 'sub_ind_hr', the functional aligner of 'sub_ind_fa', and pre and post-multiplying smoothing matrices, for block 'nblock'. 'sub_ind_hr' and 'sub_ind_fa' are within subject group 'group'. 
                Parameters:
                -----------
                howtoalign: str
                    'RDRT','RD','RD+','RT','RT+'
                group: 'temp' or 'test'
                sub_ind_hr: int
                    index of high-res connectome subject in subs[group]
                sub_ind_fa: int
                    index of functional aligner subject in subs[group]
                nblock: int
                    index of block in blocks
                """
                i,j=blocks[0,nblock],blocks[1,nblock]
                D=hr[group][sub_ind_hr][slices[i],slices[j]] 
                if howtoalign is not None and '+' in howtoalign: #the other end will be self-aligned
                    Ri=get_aligner_parcel(group,sub_ind_hr,i)
                    Rj=get_aligner_parcel(group,sub_ind_hr,j)
                else:
                    Ri=np.eye(D.shape[0],dtype=np.float32)
                    Rj=np.eye(D.shape[1],dtype=np.float32)   
                if howtoalign in ['RD','RD+','RDRT']:
                    Ri=get_aligner_parcel(group,sub_ind_fa,i)
                if howtoalign in ['RT','RT+','RDRT']:
                    Rj=get_aligner_parcel(group,sub_ind_fa,j)
                if post_hrc_fwhm:
                    pre=smoother_post[slices[i],slices[i]]
                    post=smoother_post[slices[j],slices[j]]
                else:
                    pre=np.eye(D.shape[0],dtype=np.float32)
                    post=np.eye(D.shape[1],dtype=np.float32) 
                return i,j,D,Ri,Rj,pre,post 
            def yield_args(howtoalign,group,subject_pairs=False):
                """
                if subject_pairs==True, yield outputs of get_vals for each subject pair in 'group', for each block in blocks. 
                If subject_pairs==False, yield outputs of get_vals for each subject in 'group' for each block in blocks. Pass the same subject as both 'sub_ind_hr' and 'sub_ind_fa' to get_vals
                """ 
                n_subs=len(subs[group])
                for sub_ind_hr,nR in itertools.product(range(n_subs),range(n_subs)):
                    if subject_pairs or (sub_ind_hr==nR):
                        sub_ind_fa = tutils.get_key(aligned_method,subs[group],sub_ind_hr,nR)
                        for nblock in range(blocks.shape[1]):
                            yield get_vals(howtoalign,group,sub_ind_hr,sub_ind_fa,nblock)
            def get_aligned_block(i,j,D,Ri,Rj,pre,post):
                #Given a connectome D, and aligners Ri and Rj, return the normalized aligned connectome
                X=(Ri@D.toarray())@(Rj.T)
                if i==j: #intra-parcel blocks have diagonal elements as zeros
                    np.fill_diagonal(X,0) 
                X=(pre@X)@(post.T)           
                norm=np.linalg.norm(X)
                return (X/norm).astype(np.float32)

            par_prefer='threads' #default 'threads', but processes makes faster when aligned_method=='pairwise'
            aligned_blocks={} #aligned_blocks is a dict with keys ['test','template']. aligned_blocks['test'] is a 3-dim array with elements [nD,nR,nblock]. aligned_blocks['template'] is a 2-dim array with elements [nD,nblock]. Each element is an block of a connectome transformed with a functional aligner.
            if aligned_method=='template':
                for group in groups:
                    print(f'{c.time()}: GetAlignedBlocks{group}', end=", ")
                    temp=Parallel(n_jobs=-1,prefer=par_prefer)(delayed(get_aligned_block)(*args) for args in yield_args(howtoalign,group,subject_pairs={'temp':False,'test':True}[group]))
                    if group=='test':
                        aligned_blocks['test']=np.reshape(np.array(temp,dtype=object),(len(subs['test']),len(subs['test']),blocks.shape[1]))
                    elif group=='temp':
                        aligned_blocks['temp']=np.reshape(np.array(temp,dtype=object),(len(subs['temp']),blocks.shape[1]))
                        aligned_blocks_template_mean = np.mean(aligned_blocks['temp'],axis=0) #1-dim array with elements [nblock].
            del temp

            #aligned_blocks['test']=np.roll(aligned_blocks['test'],1,axis=2)

            def correlate(aligned_method,sub_ind_hr,sub_ind_fa,blocks,nxD=None,nxR=None):
                """
                Take subject nxD's diffusion map aligned with sub nxR's template aligner. Compare this to subject nyD's diffusion map aligned with sub nyR's template aligner. Only consider parcel x parcel blocks in 'blocks'
                nyD and nyR are indices in subs['test']
                nxD and nxR are indices in subs['temp']
                """                             
                N=blocks.shape[1]
                coeffs=np.zeros((N),dtype=np.float32)
                for nblock in range(N):
                    if aligned_method=='template':
                        Y=aligned_blocks['test'][sub_ind_hr,sub_ind_fa,nblock]
                    if nxD is None and nxR is None: #with average
                        X = aligned_blocks_template_mean[nblock]
                    else: #with pairwise
                        X = aligned_blocks['temp'][nxD,nblock]
                    coeffs[nblock]=correlate_block(X,Y) 
                return coeffs

            ### Get similarity between person X and Y's connectomes (functionally aligned)     
            if get_similarity_average:
                #a is an array of shape (test subjects, test subjects, nblocks). Stores correlations of each test_subject's connectome (dim 0) aligned with each subject's aligner (dim 1), for each block (dim 2), with the (mean of template subject's connectomes each aligned with their own aligner)
                print(f'{c.time()}: GetSimilarity (av)',end=", ")            
                a=np.zeros((len(subs['test']),len(subs['test']),blocks.shape[1]),dtype=np.float32) 
                for sub_ind_hr in range(len(subs['test'])):
                    for sub_ind_fa in range(len(subs['test'])):
                        a[sub_ind_hr,sub_ind_fa,:]=correlate(aligned_method,sub_ind_hr,sub_ind_fa,blocks)
            else: a=None

        if save_file:
            np.save(save_path,{'blocks':blocks,'f':f,'a':a})

        print(f'{c.time()}: Calculations',end='')
        if get_similarity_average:    
            if ident_grouped_type=='perparcel':
                a2=np.zeros(( a.shape[:-1] + (nparcs,nparcs)) , dtype=np.float32)
                a2[:]=np.nan
                for n in range(blocks.shape[1]):
                    i=blocks[0,n]
                    j=blocks[1,n]
                    a2[:,:,i,j]=a[:,:,n]
                a=a2
                del a2
                """
                Now a is n_subs_test * n_subs_test * nparcs * nparcs. If i=1, j=171 is in 'blocks', then a[any,any,1,171] is non-nan
                """     
                #a=np.transpose( np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs)) , (0,1,3,2)) #Now a is n_subs_test * n_subs_test * nparcs * nblocksperparc


            """
            maro means (m)ean across blocks of values in (a) which were (r)egressed then made (o)rdinal (ie ranked)
            arnm means values in (a) were (r)egressed, (n)ormalized against unscrambled, then (m)ean across subject pairs
            ari means identifiability of values in (a) which were (r)egressed
            """

            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                an=tutils.subtract_nonscrambled_from_a(a) #for those values where sub nyD was aligned with different subject nyR, subtract the nonscrambled value (where nyD==nyR)
                anm=np.nanmean(an,axis=(0,1)) #mean across subject-pairs
                man=np.nanmean(an,axis=-1) #mean across blocks
                ma=np.nanmean(a,axis=-1) #nsubs*nsubs*nparcs

                ao=ranks(a,axis=1) #values to ranks along nyR axis
                ar=reg(a)   
                arn=tutils.subtract_nonscrambled_from_a(ar)
                aro=ranks(ar,axis=1) 

                arnm=np.nanmean(arn,axis=(0,1)) #mean across subject-pairs
                marn=np.nanmean(arn,axis=-1) #mean across blocks

                mao=np.nanmean(ao,axis=-1) #mean along blocks
                maro=np.nanmean(aro,axis=-1)
                mar=np.nanmean(ar,axis=-1)

            if not(ident_grouped_type=='perparcel'):
                ai=ident(a)
                ari=ident(ar)
                aoi = ident(ao) #should be same as ai
                aroi=ident(aro) #should be same as ari

            mai=ident(ma)     
            maoi=ident(mao)
            maroi=ident(maro)

            if ident_grouped_type=='perparcel':
                #mai, maoi and maroi and maoi will be shape=(nparcs)
                anN= [count_negs(an[:,:,i,:]) for i in range(nparcs)]
                arnN=[count_negs(arn[:,:,i,:]) for i in range(nparcs)]            
        
            print(f'\n av grouped {ident_grouped_type}: ', end="")
            for string in ['mai','maoi','maroi']:
                print(f"{string} {eval(f'{string}.mean()'):.0f}, ", end="")
            
            if not(ident_grouped_type=='perparcel'):
                print('\n av blockwise: ', end="")
                for string in ['ai','ari','aoi','aroi']:
                    print(f"{string} {eval(f'{string}.mean()'):.0f}, ",end="")
            
            print('')

        print(f'{c.time()}: Calculations done, Ident start')
        if to_plot and ident_grouped_type=='perparcel':
            if get_similarity_average:
                hutils.plot_parc_multi(p,align_parc_matrix,['mai','maoi','maroi','anN','arnN'],[mai,maoi,maroi,anN,arnN])

            parc_sizes=np.array(align_parc_matrix.sum(axis=1)).squeeze()
            hutils.plot_parc(p,align_parc_matrix,parc_sizes,'parc_sizes') #confounder
            if 'all_aligners' in locals(): #confounder
                scales = np.vstack( [[all_aligners.estimators[i].fit_[nparc].scale for nparc in range(nparcs)] for i in subs_inds['test']] )   
                scales_mean=scales.mean(axis=0)    
                hutils.plot_parc(p,align_parc_matrix,scales_mean,'scales')        

        def plots_average():
            """
            if not(ident_grouped_type=='perparcel'):
                print(f'AV {count_negs(anm):.1f}% of blocks (mean across sub-pairs)')
                print(f'AV {count_negs(man):.1f}% of sub-pairs (mean across blocks)')      
            print(f'AV {count_negs(an):.1f}% of (sub-pairs)*blocks')
            """
            if not(ident_grouped_type=='perparcel'):
                print(f'AV R {count_negs(arnm):.1f}% of blocks (mean across sub-pairs)')
                print(f'AV R {count_negs(marn):.1f}% of sub-pairs (mean across blocks)')    
            print(f'AV R {count_negs(arn):.1f}% of (sub-pairs)*blocks')

            #print(f'Identifiability with mean template: {mai:.1f}%, per block average {ai.mean():.1f}')
            #print(f'reg: Identifiability with mean template: {mari:.1f}%, per block average {ari.mean():.1f}')
            if not(ident_grouped_type=='perparcel'):
                fig,axs=plt.subplots(5)
                #tutils.plot_id(axs[0],mao,title='mao')
                #tutils.plot_id(axs[1],maro,title='maro')
                tutils.plot_id(axs[0],ma,title='ma')
                tutils.plot_id(axs[1],mar,title='mar')
                tutils.plot_id(axs[2],maro,title='maro')
                tutils.plot_id(axs[3],a[:,:,0],title='a_block0')
                tutils.plot_id(axs[4],ar[:,:,0],title='ar_block0')
                plt.subplots_adjust(hspace=0.5) 
                fig.suptitle(f'Similarity average', fontsize=16)

        if get_similarity_average:
            plots_average()
            if to_plot and plot_type=='save_as_html': plt.savefig(f'{figures_subfolder}/av')
        if to_plot: plt.show()
        hutils.getloadavg()
        print(hutils.memused())


    nblocks=10 #how many (parcel x parcel) blocks to examine
    block_choice='largest' #'largest', 'fromsourcevertex', 'all','few_from_each_vertex'
    howtoalign = 'RDRT' #'RDRT','RD','RD+','RT','RT+'     
    pre_hrc_fwhm=3 #smoothing kernel (mm) for high-res connectomes. Default 3
    post_hrc_fwhm=3 #smoothing kernel after alignment. Default 3

    save_file=False  
    load_file=False
    to_plot=True
    plot_type='open_in_browser' #save_as_html

 
    """
    regv=0.2
    pcatemp=False
    FCparcellation in ['Schaefer1000'] #'kmeans3000'
    if regv==0: regstr=''
    else:regstr=f'_reg{regv}'
    if pcatemp: pcatempstr='_pcatemp'
    else: pcatrempstr=''
    alignfile = f'hcpalign_rest_FC_temp_scaled_orthogonal_50-4-7_TF_0_0_0_FFF_S300_False_FC{FCparcellation}_niter1{pcatempstr}{regstr}'
    """


    #alignfile='hcpalign_movie_temp_scaled_orthogonal_50-4-7_TF_0_0_0_FFF_S300_False_niter1'
    alignfile='hcpalign_movie_temp_scaled_orthogonal_10-4-7_TF_0_0_0_FFF_S300_False_niter1'
    parcellation_string = 'S300' #make sure it is the same as  alignfile
    MSMAll=False

    subs_aligner_range = tutils.extract_sub_range_alignpickles1(alignfile)
    for subs_test_range in [range(0,5)]:
        temp = [i for i in subs_aligner_range if i not in subs_test_range]
        subs_inds={'temp': temp, 'test': subs_test_range, 'aligner': subs_aligner_range}

        print('')
        print(f"{subs_inds['test']} - {howtoalign}")

        #func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,plot_type,pre_hrc_fwhm,post_hrc_fwhm,text=FCparcellation) #for FCparcellation 
        func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,plot_type,pre_hrc_fwhm,post_hrc_fwhm,MSMAll) #tckfile='tracks_5M_1M_end.tck'


    '''
    for test in [range(0,5)]:
        aligner_nsubs = tutils.extract_nsubs_alignpickles1(alignfile)
        temp = [i for i in range(aligner_nsubs) if i not in test]
        subs_inds={'temp': temp, 'test': test}

        print('')
        print(f"{subs_inds['test']} - {howtoalign}")

        #func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,plot_type,pre_hrc_fwhm,post_hrc_fwhm,text=FCparcellation) #for FCparcellation 
        func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,plot_type,pre_hrc_fwhm,post_hrc_fwhm) #tckfile='tracks_5M_1M_end.tck'
    '''