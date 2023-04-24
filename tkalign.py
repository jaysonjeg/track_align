"""
Formerly named tkfunc3
Script to relate high-resolution diffusion connectomes to functional alignment
hr: list of high resolution connectomes as sparse arrays (59412,59412)

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
    if corr_args==['pearson','ravel']: #4.5s
        ccs=[np.corrcoef(X.ravel(),Y.ravel())[0,1]]   
    elif corr_args==['spearman','ravel']: #21s
        ccs,_=sp(X.ravel(),Y.ravel())
    return np.nanmean(ccs)
def correlate_blocks(Xs,Ys,corr_args=['pearson','ravel']):
        return [correlate_block(Xs[n],Ys[n],corr_args=corr_args) for n in range(Xs.shape[-1])]

if __name__=='__main__':

    import os, pickle, warnings, itertools
    from Connectome_Spatial_Smoothing import CSS as css
    from scipy import sparse
    import hcpalign_utils as hutils
    from hcpalign_utils import ospath
    import matplotlib.pyplot as plt, tkalign_utils as tutils
    from tkalign_utils import values2ranks as ranks, regress as reg, identifiability as ident, count_negs
    from joblib import Parallel, delayed

    print(hutils.memused())
    c=hutils.clock()

    def func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,save_plots,pre_hrc_fwhm,post_hrc_fwhm,tckfile=None):
        print(hutils.memused())
        if howtoalign!='RDRT':
            print(f'howtoalign is {howtoalign}')

        get_offdiag_blocks=True #pre-emptively calculate and cache aligned_blocks (faster but more RAM)
        plot_type={True:'save_as_html', False:'open_in_browser'}[save_plots]

        if tckfile is None:
            import socket
            hostname=socket.gethostname()
            if hostname=='DESKTOP-EGSQF3A': #home pc
                tckfile= 'tracks_5M_sift1M_200k.tck' #'tracks_5M_sift1M_200k.tck','tracks_5M.tck' 
            else: #service workbench
                tckfile='tracks_5M_1M_end.tck'

        sift2=not('sift' in tckfile) #default True
        MSMAll=False
        """
        We can either go with the largest n parcel x parcel blocks, for intra-parcel and inter-parcel blocks separately, OR alternatively we can pick a single parcel and consider all the blocks involving that parcel, ie all the links between that parcel and every other parcel. For the latter option, doing type_rowcol='row' and 'col' are relevant as comparing the two infers directionality
        """
        parcel_pair_type='o' #'o' for inter-parcel connections or 'i' for intra-parcel connections

        if block_choice=='fromsourcevertex': type_rowcol = 'col'
        if block_choice in ['fromsourcevertex','all']: nblocks=0
        elif block_choice in ['all','maxhpmult']: 
            ident_grouped_type='perparcel' #'wholebrain' , 'perparcel'
        else:
            ident_grouped_type='wholebrain'

        get_similarity_pairwise=False #correlations bw nxD*nxR (matched), and nyD*(all possible nyDs)
        get_similarity_average=True #correlation between mean(nxD*nxR(matched)), and nyD*(all possible nyDs)

        align_nparcs=tutils.extract_nparcs(alignfile)
        align_labels=hutils.Schaefer(align_nparcs)
        align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)
        
        corr_args=['pearson','ravel'] #first arg 'pearson' or 'spearman', second arg 'columnwise' or 'ravel' \
        fa_sparse=False #default False
        aligned_descale=False #Default False. make R transformations orthogonal again (doesn't make any difference to correlations). 
        aligned_negs='abs' #options 'abs' (default), 'zero' (performance slightly worse), 'leave'. What to do with negative values in R matrix. 'abs' to use absolute value. 'zero' to make it zero. 'leave' to leave as is.
        show_same_aligner=False #plots show nxR==nyR (both aligned by same aligner). Default False

        groups=['temp','test']
        subs = {group: [hutils.subs[i] for i in subs_inds[group]] for group in groups}
        alignfile_nsubs=tutils.extract_nsubs(alignfile)
        subs['aligner'] = [hutils.subs[i] for i in range(alignfile_nsubs)]


        #save_prefix = f'r{hutils.datetime_for_filename()}'
        save_prefix = f"corrs_0-{alignfile_nsubs}_{len(subs['temp'])}s_{min(subs_inds['test'])}-{max(subs_inds['test'])}_{tckfile[:-4]}_{pre_hrc_fwhm}mm_{post_hrc_fwhm}mm_{align_nparcs}p_{nblocks}b_{block_choice[0:2]}_{howtoalign}"
        results_subfolder=ospath(f'{hutils.results_path}/{save_prefix}')
        if to_plot and save_plots: hutils.mkdir(results_subfolder)
        p=hutils.surfplot(results_subfolder,plot_type=plot_type)
        save_folder=f'{hutils.intermediates_path}/tkalign_corrs'
        hutils.mkdir(save_folder)
        save_path=ospath(f"{save_folder}/{save_prefix}.npy")


        aligned_method = 'template' if (('temp' in alignfile) or ('Temp' in alignfile)) else 'pairwise'
        if aligned_method=='pairwise': 
            get_similarity_pairwise=True
            get_similarity_average=False
            assert( set(subs['temp']).isdisjoint(subs['test']) )
        nparcs=align_parc_matrix.shape[0]
        sorter,unsorter,slices=tutils.makesorter(align_labels) # Sort vertices by parcel membership, for speed
        post_skernel=sparse.load_npz(ospath(f'{hutils.intermediates_path}/smoothers/100610_{post_hrc_fwhm}_0.01.npz'))[sorter[:,None],sorter]

        if load_file and os.path.exists(save_path):
            print('loading f and a')
            blocks,f,a = tutils.load_f_and_a(save_path)
        else:
            #Get high-res connectomes
            print(f'{c.time()}: Get highres connectomes and downsample',end=", ")
            par_prefr_hrc='threads'        
            hr_for_hp = hutils.get_highres_connectomes(c,subs['aligner'],tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefr_hrc,n_jobs=-1)
            hp=[css.downsample_high_resolution_structural_connectivity_to_atlas(hrs, align_parc_matrix) for hrs in hr_for_hp] #most connected parcel pairs are determined from template subjects
            del hr_for_hp
            hps,hpsx,hpsxa = tutils.get_hps(hp)
            hr = {group : hutils.get_highres_connectomes(c,subs[group],tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefr_hrc,n_jobs=-1) for group in groups} 

            print(f'{c.time()}: Reorder', end=", ")
            hr = {group: [array[sorter[:,None],sorter] for array in hr[group]] for group in groups}

            smoother=sparse.load_npz(ospath(f'{hutils.intermediates_path}/smoothers/100610_{pre_hrc_fwhm}_0.01.npz')).astype(np.float32)[sorter[:,None],sorter]   

            print(f'{c.time()}: Smooth hrc', end=", ")
            hr={group : hutils.smooth_highres_connectomes(hr[group],smoother) for group in groups}

            print(f'{c.time()}: GetBlocks', end=", ")       
            if block_choice=='largest': 
                blocks=tutils.get_blocks_largest(nblocks,parcel_pair_type,hps,hpsx)
            elif block_choice=='fromsourcevertex':
                blocks=tutils.get_blocks_source(align_labels,hpsxa,nblocks,nparcs,type_rowcol)
            elif block_choice=='all':
                blocks=tutils.get_blocks_all()
            elif block_choice=='maxhpmult':
                blocks=tutils.get_blocks_maxhpmult(nblocks,hpsxa,nparcs)

            ### Get func aligners ###
            print(f'{c.time()}: GetAligners', end=", ")
            aligner_file = f'{hutils.intermediates_path}/alignpickles/{alignfile}.p'
            #all_aligners = pickle.load( open( ospath(aligner_file), "rb" )) 
            if aligned_method=='template':                                                    
                fa={}
                for group in groups:
                    all_aligners = pickle.load( open( ospath(aligner_file), "rb" )) #load each time because func(i) will modify arrays in all_aligners            
                    func = lambda nsub: tutils.aligner2sparse(all_aligners.estimators[nsub],slices,fa_sparse=fa_sparse,aligned_descale=aligned_descale,aligned_negs=aligned_negs)
                    fa[group] = [func(i) for i in subs_inds[group]]
            elif aligned_method=='pairwise':          
                all_aligners = pickle.load( open( ospath(aligner_file), "rb" )) 
                allowed_keys = [f'{i}-{j}' for i in subs['test'] for j in subs['temp'] if i!=j] #only aligners which transform test subjects to template subjects
                fa={'test':{key:value for key,value in all_aligners.items() if key in allowed_keys}}

            def get_aligner_parcel(group,key,i):
                #group is 'temp' or 'test'
                if fa_sparse:
                    return fa[group][key][slices[i],slices[i]].toarray()
                else:
                    return fa[group][key].fit_[i].R    
            def get_vals(howtoalign,group,nD,key,n):
                #if group=='temp': howtoalign='RDRT' #RDRT to make template hrc with RDRT, or equals howtoalign
                i,j=blocks[0,n],blocks[1,n]
                D=hr[group][nD][slices[i],slices[j]] 
                if howtoalign is not None and '+' in howtoalign: #the other end will be self-aligned
                    Ri=get_aligner_parcel(group,nD,i)
                    Rj=get_aligner_parcel(group,nD,j)
                else:
                    Ri=np.eye(D.shape[0],dtype=np.float32)
                    Rj=np.eye(D.shape[1],dtype=np.float32)   
                if howtoalign in ['RD','RD+','RDRT']:
                    Ri=get_aligner_parcel(group,key,i)
                if howtoalign in ['RT','RT+','RDRT']:
                    Rj=get_aligner_parcel(group,key,j)
                if post_hrc_fwhm:
                    pre=post_skernel[slices[i],slices[i]]
                    post=post_skernel[slices[j],slices[j]]
                else:
                    pre=np.eye(D.shape[0],dtype=np.float32)
                    post=np.eye(D.shape[1],dtype=np.float32) 
                return i,j,D,Ri,Rj,pre,post 
            def yield_args(howtoalign,group,get_offdiag_blocks=False): 
                n_subs=len(subs[group])
                for nD,nR in itertools.product(range(n_subs),range(n_subs)):
                    if get_offdiag_blocks or (nD==nR):
                        key = tutils.get_key(aligned_method,subs[group],nD,nR)
                        for n in range(blocks.shape[1]):
                            yield get_vals(howtoalign,group,nD,key,n)
            def get_aligned_block(i,j,D,Ri,Rj,pre,post):
                X=(Ri@D.toarray())@(Rj.T)
                if i==j: #intra-parcel blocks have diagonal elements as zeros
                    np.fill_diagonal(X,0) 
                X=(pre@X)@(post.T)           
                norm=np.linalg.norm(X)
                return (X/norm).astype(np.float32)

            get_offdiag_blocks_all={'temp':False,'test':True}
            par_prefer='processes' #default 'threads', but processes makes faster when aligned_method=='pairwise'
            aligned_blocks={}
            if aligned_method=='template':
                for group in groups:
                    print(f'{c.time()}: GetAlignedBlocks{group}', end=", ")
                    temp=Parallel(n_jobs=-1,prefer=par_prefer)(delayed(get_aligned_block)(*args) for args in yield_args(howtoalign,group,get_offdiag_blocks=get_offdiag_blocks_all[group]))
                    if group=='test':
                        aligned_blocks['test']=np.reshape(np.array(temp,dtype=object),(len(subs['test']),len(subs['test']),blocks.shape[1]))
                    elif group=='temp':
                        aligned_blocks['temp']=np.reshape(np.array(temp,dtype=object),(len(subs['temp']),blocks.shape[1]))
                        aligned_blocks_template_mean = np.mean(aligned_blocks['temp'],axis=0) 
            elif aligned_method=='pairwise':
                def yield_args_pairwise():
                    for nDtarget,nD,nR in itertools.product(range(len(subs['temp'])),range(len(subs['test'])),range(len(subs['test']))):
                        key = f"{subs['test'][nR]}-{subs['temp'][nDtarget]}"
                        for n in range(blocks.shape[1]):
                            yield get_vals(howtoalign,'test',nD,key,n)       
                temp=Parallel(n_jobs=-1,prefer=par_prefer)(delayed(get_aligned_block)(*args) for args in yield_args_pairwise())
                aligned_blocks['test']=np.reshape(np.array(temp,dtype=object),(len(subs['temp']),len(subs['test']),len(subs['test']),blocks.shape[1])) #element nD,nR,nDtarget,nblock has nD's connectivity for nblock transformed using the aligner that takes subject 'nR' to 'nDtarget'.
                temp=Parallel(n_jobs=-1,prefer=par_prefer)(delayed(get_aligned_block)(*args) for args in yield_args(None,'temp',get_offdiag_blocks=False)) #howtoalign set to None so we don't transform template hrcs at all
                aligned_blocks['temp']=np.reshape(np.array(temp,dtype=object),(len(subs['temp']),blocks.shape[1]))
            del temp

            #aligned_blocks['test']=np.roll(aligned_blocks['test'],1,axis=2)

            def correlate(aligned_method,nyD,nyR,blocks,nxD=None,nxR=None):
                """
                Take subject nxD's diffusion map aligned with sub nxR's template aligner. Compare this to subject nyD's diffusion map aligned with sub nyR's template aligner. Only consider parcel x parcel blocks in 'blocks'
                nyD and nyR are indices in subs['test']
                nxD and nxR are indices in subs['temp']
                """                             
                N=blocks.shape[1]
                coeffs=np.zeros((N),dtype=np.float32)
                for n in range(N):
                    if aligned_method=='template':
                        Y=aligned_blocks['test'][nyD,nyR,n]
                    elif aligned_method=='pairwise':
                        Y=aligned_blocks['test'][nxD,nyD,nyR,n] #hrc of nyD is transformed using nyR->nxD (from subs['test']) aligner
                    if nxD is None and nxR is None: #with average
                        X = aligned_blocks_template_mean[n]
                    else: #with pairwise
                        X = aligned_blocks['temp'][nxD,n]
                    coeffs[n]=correlate_block(X,Y) 
                return coeffs

            ### Get similarity between person X and Y's connectomes (functionally aligned)

            """
            Template:
            For each pair of subjects nxD and nyD, use his own aligner (nxR) on nxD, and iterate through every subject's nyR to align nyD. Of course, there's no point comparing 2 subjects aligned with the same aligner (nxR==nyR) as this will be be very similar to original connectomes (except maybe with some smoothing). For example, if len(subs['test'])==5, nxD==2 and nyD==4, then align nxD with his own aligner (so nxR==2) but try nyR in [0,1,3,4]. nyR==4 will be the 'unscrambled' version which is hopefully than the other 3 options

            Pairwise:
            Subject nxD's diffusion map compared to subject nyD's diffusion map (aligned via nY->nX aligner)
            """          
            if get_similarity_pairwise:
                print(f'{c.time()}: GetSimilarity (pair)',end=", ")
                #store correlation for each block in each subject-pair 
                pair_type=0 #0 and 1, similar speed, 1 with 'processes' seems slower
                if pair_type==0:
                    f=np.zeros((len(subs['temp']),len(subs['test']),len(subs['test']),blocks.shape[1]),dtype=np.float32) 
                    f[:]=np.nan 
                    def yield_indices():
                        for nyD,nyR in itertools.product(range(len(subs['test'])),range(len(subs['test']))):
                            for nxD in range(len(subs['temp'])):
                                if subs['temp'][nxD] != subs['test'][nyD]:
                                    nxR = nxD
                                    yield nyD,nyR,nxD,nxR
                    def enter_values(nyD,nyR,nxD,nxR,blocks):
                        f[nxD,nyD,nyR,:]=correlate(aligned_method,nyD,nyR,blocks,nxD,nxR)
                    Parallel(n_jobs=-1,require='sharedmem')(delayed(enter_values)(*args,blocks) for args in yield_indices())
                
                elif pair_type==1:
                    print("code not edited for len(subs['temp']) vs len(subs['test'])")
                    assert(get_offdiag_blocks)
                    def yield_aligned_blocks():
                        for nxD, nyD, nyR in itertools.product(range(len(subs['test'])),range(len(subs['test'])),range(len(subs['test']))):  
                            nxR=nxD
                            Xs = aligned_blocks['test'][nxD,nxR,:]
                            Ys = aligned_blocks['test'][nyD,nyR,:]
                            yield Xs,Ys    
                    temp=Parallel(n_jobs=-1,prefer='threads')(delayed(correlate_blocks)(*args) for args in yield_aligned_blocks())
                    f=np.reshape(np.array(temp,dtype=np.float32),(len(subs['test']),len(subs['test']),len(subs['test']),blocks.shape[1]))  
                    f[np.eye(f.shape[0],dtype=bool),:,:]=np.nan #set diagonals in dims 0 and 1 to np.nan  
            else: f=None           

            
            if get_similarity_average:
                print(f'{c.time()}: GetSimilarity (av)',end=", ")            
                a=np.zeros((len(subs['test']),len(subs['test']),blocks.shape[1]),dtype=np.float32) #stores correlations or nyD*nyR with the average of other subjects nxDs each aligned with their own aligner nxR
                for nyD in range(len(subs['test'])):
                    for nyR in range(len(subs['test'])):
                        a[nyD,nyR,:]=correlate(aligned_method,nyD,nyR,blocks)
            else: a=None

        if save_file:
            np.save(save_path,{'blocks':blocks,'f':f,'a':a})

        print(f'{c.time()}: Calculations',end='')
        if get_similarity_pairwise:   
            fn=tutils.subtract_nonscrambled_from_z(f) #4-dim array with elements [nxD,nyD,nyR,nblock]. Nonscrambled elements set to nan
            fn2=tutils.unaligned_to_nan(fn,subs['test'],subs['temp']) #Set elements where X and Y are aligned with same aligner to nan. Size (nxD,nyD,nyR,block) = (3,3,3,200)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                #average across nyR (different subjects' aligners). Ignore nans. That means where same aligner used for nxD and nyD (nxR==nyR), and where aligner belongs to the same subject so it is not scrambled (nyR==nyD). 3-dim array with elements [nxD, nyD, nblock]. If 'template', this is upper triangular. If 'pairwise', this has diagonal zeros
                fnm=np.nanmean(fn2,axis=2) #size (nxD,nyD, block)=(3,3,200)
                fnmm=np.nanmean(fnm,axis=(0,1)) #Mean across subjects nxD and nyD . Size (block) = (200)  
                
                if ident_grouped_type=='perparcel':
                    f2=np.zeros(( f.shape[:-1] + (nparcs,nparcs)) , dtype=np.float32)
                    f2[:]=np.nan
                    for n in range(blocks.shape[1]):
                        i=blocks[0,n]
                        j=blocks[1,n]
                        f2[:,:,:,i,j]=f[:,:,:,n]
                    f=f2
                    del f2

                fx=tutils.unaligned_to_nan(f,subs['test'],subs['temp'])

                #Average across blocks. Ignore nans. That means blocks with no connections in one of the subject pairs. 3-dim array with elements [nxD, nyD, nyR]
                mf=np.nanmean(f,axis=3) #size (nxD,nyD,nyR)=(3,3,3)
                mfn=np.nanmean(fn,axis=3)
                m_unscram_ranks=tutils.get_unscram_ranks(ranks(mf),aligned_method) 


                mean0 = lambda array: np.nanmean(array,axis=0)
                mfx=np.nanmean(fx,axis=3)
                mfom=mean0(ranks(mf))
                mfxmr=reg(mean0(mfx))
                
                mfomi=ident(mean0(ranks(mf)))
                mfmri=ident(reg(mean0(mf)))
                mfxmi=ident(mean0(mfx))
                mfxomi=ident(mean0(ranks(mfx)))
                mfxmri=ident(reg(mean0(mfx)))   
                mfxomri=ident(reg(mean0(ranks(mfx))))
                mfxromi=ident(mean0(ranks(reg(mfx))))                

                
                if not(ident_grouped_type=='perparcel'):
                    fomi=ident(mean0(ranks(f)))
                    fmri=ident(reg(mean0(f)))
                    fxmi=ident(mean0(fx))
                    fxomi=ident(mean0(ranks(fx))) #might be best
                    fxmri=ident(reg(mean0(fx)))   
                    fxomri=ident(reg(mean0(ranks(fx)))) #no.2
                    fxromi=ident(mean0(ranks(reg(fx)))) #no.2

                fxo=ranks(fx)
                fxoM=np.nanmean(fxo,axis=3)
                fxoMm=mean0(fxoM)
                fxoMmi = ident(fxoMm) 

            
            print(f' pairs grouped {ident_grouped_type}: ', end="")
            for string in ['mfomi','mfmri','mfxmi','mfxomi','mfxmri','mfxomri','mfxromi','fxoMmi']:
                print(f"{string} {eval(f'{string}.mean()'):.0f}, ", end="")    
            
            if not(ident_grouped_type=='perparcel'):
                print('\n pairs blockwise: ', end="")
                for string in ['fomi','fmri','fxmi','fxomi','fxmri','fxomri','fxromi']:
                #for string in ['fmri','fxromi']: 
                    print(f"{string} {eval(f'{string}.mean()'):.0f}, ",end="")

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


            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                an=tutils.subtract_nonscrambled_from_a(a)
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

        print(f'{c.time()}: Calculations done')
        if to_plot and ident_grouped_type=='perparcel':

            if get_similarity_pairwise:
                hutils.plot_parc_multi(p,align_parc_matrix,['mfxomi','fxoMmi'],[mfxomi,fxoMmi])
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
                fig,axs=plt.subplots(4)
                #tutils.plot_id(axs[0],mao,title='mao')
                #tutils.plot_id(axs[1],maro,title='maro')
                tutils.plot_id(axs[0],ma,title='ma')
                tutils.plot_id(axs[1],mar,title='mar')
                tutils.plot_id(axs[2],a[:,:,0],title='a_block0')
                tutils.plot_id(axs[3],ar[:,:,0],title='ar_block0')
                plt.subplots_adjust(hspace=0.5) 
                fig.suptitle(f'Similarity average', fontsize=16)



        def plots_pairwise(show_same_aligner):         
            
            print(f"PAIR { 100*(np.sum(np.array(m_unscram_ranks)>=(len(subs['test'])-2)) / len(m_unscram_ranks)):.1f} % of sub-pairs had orig> all scram possibilities")
            print(f'PAIR {count_negs(fnmm):.1f}% of blocks (mean across sub-pairs, nyR) had original > scrambled')        
            print(f'PAIR {count_negs(fnm):.1f} % of (sub-pairs)*blocks (mean across nyR)') 
            
            print(f'PAIR {count_negs(fn2):.1f} % of (sub-pairs)*nyR*blocks') 
            #print(f'Identifiability with pairs {mfomi:.1f}%, per block average {fmri.mean():.1f}%')
            if not(ident_grouped_type=='perparcel'):
                fig,axs=plt.subplots(5)
                tutils.plotter(axs[0],mf,aligned_method,show_same_aligner,'m',subs['test'],subs['temp'])
                tutils.plot_id(axs[1],mfom,title='mfom: pairwise')
                tutils.plot_id(axs[2],mfxmr,title='mfxmr: pairwise')
                tutils.plot_id(axs[3],fxoMm,title='fxoMm: pairwise') 
                axs[4].hist(fnmm)
                axs[4].set_title('Distribution of scrambled minus nonscrambled \nacross blocks')
                #axs[0].hist(m_unscram_ranks)
                #axs[0].set_title('m: Rank order of unscrambled \namong scrambled: Bigger is better')
                #tutils.plotter(axis_mn,mfn,aligned_method,show_same_aligner,'mfn',,subs_test,subs_temp,drawhorzline=True)    
                plt.subplots_adjust(hspace=0.5)
                fig.suptitle(f'Similarity pairwise', fontsize=16)

        if get_similarity_pairwise:
            plots_pairwise(show_same_aligner)
            if to_plot and save_plots: plt.savefig(f'{results_subfolder}/pair')
        if get_similarity_average:
            plots_average()
            if to_plot and save_plots: plt.savefig(f'{results_subfolder}/av')
        if to_plot: plt.show()
        hutils.getloadavg()
        print(hutils.memused())


    nblocks=5 #how many (parcel x parcel) blocks to examine
    block_choice='maxhpmult' #'largest', 'fromsourcevertex', 'all','maxhpmult'
    save_file=True  
    load_file=False
    to_plot=True
    save_plots=True

    pre_hrc_fwhm=5 #smoothing kernel (mm) for high-res connectomes. Default 3
    post_hrc_fwhm=5 #smoothing kernel after alignment. Default 3

    howtoalign='RDRT' #'RDRT','RD','RD+','RT','RT+'

    for regv in [0.05]:
        if regv==0: regstr=''
        else:regstr=f'_reg{regv}'
        alignfile = f'hcpalign_movie_temp_scaled_orthogonal_50-4-7_TF_0_0_0_FFF_S300_False_niter1{regstr}'
        for test in [range(0,10),range(10,20),range(20,30),range(30,40),range(40,50)]:
            aligner_nsubs = tutils.extract_nsubs(alignfile)
            temp = [i for i in range(aligner_nsubs) if i not in test]
            subs_inds={'temp': temp, 'test': test}

            print('')
            print(f"{subs_inds['test']} - {howtoalign}")
            print(regv)
            func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,save_plots,pre_hrc_fwhm,post_hrc_fwhm) #tckfile='tracks_5M_1M_end.tck'
