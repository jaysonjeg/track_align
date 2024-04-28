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


def correlate_blocks(Xs,Ys,corr_args=['pearson','ravel']):
        #Do correlate_block for each corresponding item in Xs and Ys
        return [tutils.correlate_block(X,Y,corr_args=corr_args) for X,Y in zip(Xs,Ys)]

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

    def func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,plot_type,pre_hrc_fwhm,post_hrc_fwhm,MSMAll, tckfile=None,align_template_to_imgs=False,text=''):
        """
        Parameters:
        -----------
        subs_inds: dict
            keys are 'temp', 'test', 'blocks'. Values are lists of subject indices in hutils.all_subs
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
        align_template_to_imgs: bool
            whether template was aligned to images or vice versa
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
        sift2=not('sift' in tckfile) #True, unless there is 'sift' in tckfile

        print(hutils.memused())
        if howtoalign!='RDRT':
            print(f'howtoalign is {howtoalign}')

        if block_choice in ['fromsourcevertex','all']: 
            nblocks=0

        aligned_method='template'

        ### Get parcellation 
        assert(parcellation_string in alignfile)
        align_labels=hutils.parcellation_string_to_parcellation(parcellation_string)
        align_parc_matrix=hutils.parcellation_string_to_parcmatrix(parcellation_string)
        nparcs=align_parc_matrix.shape[0]

        ### Set up subject lists 
        # Variable subs is a dict, keys are 'temp', 'test', and 'aligner'. Values are subject IDs as strings. 'aligner' subjects are used to determine the blocks with the most streamlines. 'temp' subjects are used to make the template connectome. 'test' subjects are aligned either with their own or other test subjects' aligners. Their correlation with the template connectome is calculated.
        subs = {group: [hutils.all_subs[i] for i in subs_inds[group]] for group in subs_inds.keys()} 
        #assert( set(subs['temp']).isdisjoint(subs['test']) )

        ### Set up filenames and folders for saving and figures
        save_prefix = f"{alignfile}_B{nblocks}{block_choice[0:2]}{subs_inds['blocks'].start}-{subs_inds['blocks'].stop}_D{tckfile[:-4]}_{pre_hrc_fwhm}mm_{post_hrc_fwhm}mm_{howtoalign}{text}_S{len(subs['temp'])}_{subs_inds['test'].start}-{subs_inds['test'].stop}"
        print(save_prefix)

        figures_subfolder=ospath(f'{hutils.results_path}/figures/{save_prefix}') #save figures in this folder
        if to_plot and plot_type=='save_as_html': hutils.mkdir(figures_subfolder)
        p=hutils.surfplot(figures_subfolder,plot_type=plot_type)

        save_folder=f'{hutils.intermediates_path}/tkalign_corrs2' #save results data in this folder
        hutils.mkdir(save_folder)
        save_path=ospath(f"{save_folder}/{save_prefix}.npy")

        if parcellation_string[0]=='R': #searchlight
            S300_labels=hutils.parcellation_string_to_parcellation('S300')
            S300_parcmatrix = hutils.parcellation_string_to_parcmatrix('S300')
            sorter_S300,unsorter_S300,slices_S300=tutils.makesorter(S300_labels)
            alignfile_S300 = tutils.replace_S300_alignpickles3(alignfile)

        ### Set up smoothing kernels
        sorter,unsorter,slices=tutils.makesorter(align_labels) # Sort vertices by parcel membership, for speed
        smoother_pre = tutils.get_smoother(pre_hrc_fwhm)[sorter[:,None],sorter]
        smoother_post = tutils.get_smoother(post_hrc_fwhm)[sorter[:,None],sorter]
        if load_file and os.path.exists(save_path):
            print('loading a')
            blocks,a = tutils.load_a(save_path)
        else:
            print(f'{c.time()}: Get parcellated connectomes and blocks',end=", ")  
            blocks,nparcs = tutils.get_blocks(c,tckfile, MSMAll, sift2, align_parc_matrix, subs['blocks'], block_choice,nblocks,parcel_pair_type,align_labels,nparcs,type_rowcol,par_prefer_hrc)
            print(f'{c.time()}: Get high-res connectomes',end=", ")  
            hr = {group : hutils.get_highres_connectomes(c,subs[group],tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefer_hrc,n_jobs=-1) for group in ['temp','test']} # Get high-res connectomes for test and template subjects. hr[group] is a list of sparse arrays, Each array is a connectome for a subject
            print(f'{c.time()}: Reorder connectomes', end=", ")
            hr = {group: [array[sorter[:,None],sorter] for array in hr[group]] for group in ['temp','test']}
            print(f'{c.time()}: Smooth hrc', end=", ")
            hr={group : hutils.smooth_highres_connectomes(hr[group],smoother_pre) for group in ['temp','test']}

            print(f'{c.time()}: GetAligners', end=", ")
            if align_template_to_imgs:
                groups=['test']
            else:
                groups=['test','temp']    
            if aligned_method=='template':                                                    
                fa, all_aligners = tutils.new_func(c,subs_inds, alignfile, aligner2sparsearray, aligner_descale, aligner_negatives, sorter, slices, smoother_post, groups)
                #scales = np.vstack( [[all_aligners[i].fit_[nparc].scale for nparc in range(nparcs)] for i in range(len(all_aligners))] )   
                del all_aligners

                if parcellation_string[0]=='R': #searchlight
                    fa_S300, _ = tutils.new_func(c,subs_inds, alignfile_S300, aligner2sparsearray, aligner_descale, aligner_negatives, sorter_S300, slices_S300, None, groups)

            
            #TRY THIS
            """
            r0 = fa['test'][0].fit_[0].R
            r0t = tutils.randomise_but_preserve_row_col_sums(r0)
            print(np.corrcoef(r0.ravel(),r0t.ravel())[0,1])
            print(np.corrcoef(r0.sum(axis=0),r0t.sum(axis=0))[0,1])
            print(np.corrcoef(r0.sum(axis=1),r0t.sum(axis=1))[0,1])
            for subject in range(len(fa['test'])):
                for nparc in range(len(fa['test'][0].fit_)):
                    fa['test'][subject].fit_[nparc].R = tutils.randomise_but_preserve_row_col_sums(fa['test'][subject].fit_[nparc].R)
            """

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

            def get_aligner_parcel_S300(group,sub_ind,nparcel):
                return fa_S300[group][sub_ind].fit_[nparcel].R   

            def get_aligned_block(i,j,D,Ri,Rj,pre=None,post=None):
                #Given a connectome D, and aligners Ri and Rj, return the normalized aligned connectome
                if type(D)==np.ndarray:
                    X=(Ri@D)@(Rj.T)
                else: #D is sparse array
                    X=(Ri@D.toarray())@(Rj.T)
                if i==j: #intra-parcel blocks have diagonal elements as zeros
                    np.fill_diagonal(X,0) 
                #X=(pre@X)@(post.T) #moved to earlier in code, with function get_template_aligners   
                norm=np.linalg.norm(X)
                return (X/norm).astype(np.float32)

            par_prefer='threads' #default 'threads', but processes makes faster when aligned_method=='pairwise'

            if align_template_to_imgs==True:
                #aligned_blocks is a 2-dim array with elements [nR,nblock]. Each element is a block of the mean template connectome, aligned to subject nR's functional space. nR is a test subject
                def get_template_connectome_parcel(nblock):
                    i,j=blocks[0,nblock],blocks[1,nblock]
                    if parcellation_string[0]=='R': #searchlight
                        temp = [hr['temp'][sub_ind][align_labels[i],:][:,S300_labels==j] for sub_ind in range(len(subs['temp']))]
                    else:
                        temp = [hr['temp'][sub_ind][slices[i],slices[j]] for sub_ind in range(len(subs['temp']))] #list of sparse arrays, each array is a connectome block for a template subject
                    temp = [array.toarray().astype(np.float32) for array in temp] #convert to dense arrays
                    temp = [tutils.divide_by_Frobenius_norm(array) for array in temp]
                    return np.mean(temp,axis=0) #mean across template subjects
                def get_vals2(howtoalign,sub_ind_fa,nblock):
                    D = template_Ds[nblock]
                    i,j=blocks[0,nblock],blocks[1,nblock]
                    Ri=np.eye(D.shape[0],dtype=np.float32)
                    Rj=np.eye(D.shape[1],dtype=np.float32)   
                    if howtoalign in ['RD','RDRT']:
                        Ri=get_aligner_parcel('test',sub_ind_fa,i)
                    if howtoalign in ['RT','RDRT']:
                        if parcellation_string[0]=='R': #searchlight
                            Rj=get_aligner_parcel_S300('test',sub_ind_fa,j)
                        else:
                            Rj=get_aligner_parcel('test',sub_ind_fa,j)
                    return i,j,D,Ri,Rj 
                def yield_args2(howtoalign):
                    for sub_ind_fa in range(len(subs['test'])):
                        for nblock in range(blocks.shape[1]):
                            yield get_vals2(howtoalign,sub_ind_fa,nblock)
                print(f'{c.time()}: GetTemplateConnectome', end=", ")
                template_Ds = Parallel(n_jobs=-1,prefer='threads')(delayed(get_template_connectome_parcel)(nblock) for nblock in range(blocks.shape[1]))
                print(f'{c.time()}: GetAlignedBlocks', end=", ")
                temp=Parallel(n_jobs=-1,prefer=par_prefer)(delayed(get_aligned_block)(*args) for args in yield_args2(howtoalign))
                aligned_blocks = np.reshape(np.array(temp,dtype=object),(len(subs['test']),blocks.shape[1])) 


            if align_template_to_imgs==True:
                def get_test_connectome_parcel(sub_ind,nblock):
                    i,j=blocks[0,nblock],blocks[1,nblock]
                    if parcellation_string[0]=='R': #searchlight
                        D = hr['test'][sub_ind][align_labels[i],:][:,S300_labels==j]
                    else:
                        D = hr['test'][sub_ind][slices[i],slices[j]]
                    D = D.toarray().astype(np.float32)
                    return D
                def yield_subs_and_blocks():
                    for sub_ind in range(len(subs['test'])):
                        for nblock in range(blocks.shape[1]):
                            yield sub_ind,nblock
                print(f'{c.time()}: GetTestConnectomes', end=", ")
                temp = Parallel(n_jobs=-1,prefer='threads')(delayed(get_test_connectome_parcel)(*args) for args in yield_subs_and_blocks())
                test_Ds_array = np.reshape(np.array(temp,dtype=object),(len(subs['test']),blocks.shape[1])) 
                        
            if get_similarity_average:
                #a is an array of shape (test subjects, test subjects, nblocks). Stores correlations between template connectome aligned with each test subject's aligner (dim 1), and each test subject's connectome (dim 0), for each block (dim 2)
                print(f'{c.time()}: GetSimilarity (av)',end=", ")                
                def correlate_single_block(aligned_method,align_template_to_imgs,aligned_blocks_single,test_Ds_array_single):
                    nsubjects=len(aligned_blocks_single)
                    coeffs = np.zeros((nsubjects,nsubjects),dtype=np.float32)
                    for sub_ind_hr in range(nsubjects):
                        for sub_ind_fa in range(nsubjects):  
                            if aligned_method=='template':
                                if align_template_to_imgs==True:
                                    Y=aligned_blocks_single[sub_ind_fa] #template connectome aligned with sub_ind_fa
                                    X = test_Ds_array_single[sub_ind_hr] #test subject sub_ind_hr's connectome
                                elif align_template_to_imgs==False:
                                    assert(0) #code not ready for this yet
                                    #Y=aligned_blocks['test'][sub_ind_hr,sub_ind_fa] #test subject sub_ind_hr's connectome aligned with sub_ind_fa
                                    #X = aligned_blocks_template_mean[nblock] #mean aligned template connectome
                            coeffs[sub_ind_hr,sub_ind_fa]=tutils.correlate_block(X,Y)    
                    return coeffs             
                a = Parallel(n_jobs=-1,prefer='processes')(delayed(correlate_single_block)(aligned_method,align_template_to_imgs,aligned_blocks[:,nblock],test_Ds_array[:,nblock]) for nblock in range(blocks.shape[1]))
                a=np.dstack(a)
                
            else: a=None
        if save_file:
            np.save(save_path,{'blocks':blocks,'a':a})

        print(f'{c.time()}: Calculations',end='')
        if get_similarity_average:    
            """
            maro means (m)ean across blocks of values in (a) which were (r)egressed then made (o)rdinal (ie ranked)
            arnm means values in (a) were (r)egressed, (n)ormalized against unscrambled, then (m)ean across subject pairs
            ari means identifiability of values in (a) which were (r)egressed
            az means values in (a) were averaged across each partner parcel
            """
            if False: #transpose a so that fa is on axis 0, hr on axis 1. Identifiability then means whether the hr is identifiable given fa
                if a.ndim==3:
                    a=np.transpose(a,[1,0,2]) 
                elif a.ndim==4:
                    a=np.transpose(a,[1,0,2,3])

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

                arnm=np.nanmean(arn,axis=(0,1)) #mean across subject-pairs (within block)
                marn=np.nanmean(arn,axis=-1) #mean across blocks (within subject-pairs)

                arnc=[tutils.count_negs(arn[:,:,i]) for i in range(arn.shape[2])] #no of negative values in each block (across sub-pairs)
                carn=[tutils.count_negs(arn[i,:,:]) for i in range(len(subs['test']))] #no of negative values for subject's connectome (across blocks and subjects for fa)
                anc=[tutils.count_negs(an[:,:,i]) for i in range(arn.shape[2])] 
                can=[tutils.count_negs(an[i,:,:]) for i in range(len(subs['test']))] 

                mao=np.nanmean(ao,axis=-1) #mean along blocks
                maro=np.nanmean(aro,axis=-1)
                mar=np.nanmean(ar,axis=-1)

                mai=ident(ma)     
                maoi=ident(mao)
                mari=ident(mar)
                maroi=ident(maro)

                #identifiability for each block separately
                ai=ident(a)
                ari=ident(ar)
                aoi = ident(ao) #should be same as ai
                aroi=ident(aro) #should be same as ari

                if block_choice=='few_from_each_vertex':
                    az = np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs))
                    arz = np.reshape(ar,(ar.shape[0],ar.shape[1],nblocks,nparcs))

                    #Now repeat all above with a3 instead of a. Need an, arn, 'mai','mari','maoi','maroi'
                    azn=tutils.subtract_nonscrambled_from_a(az) #for those values where sub nyD was aligned with different subject nyR, subtract the nonscrambled value (where nyD==nyR)
                    maz=np.nanmean(az,axis=2) #nsubs*nsubs*nparcs

                    azo=ranks(az,axis=1) #values to ranks along nyR axis
                    azr=reg(az)
                    arzn=tutils.subtract_nonscrambled_from_a(arz) 
                    arzo=ranks(arz,axis=1)

                    mazo=np.nanmean(azo,axis=2)
                    marzo=np.nanmean(arzo,axis=2)
                    marzn=np.nanmean(arzn,axis=2)
                    marz=np.nanmean(arz,axis=2)

                    mazi=ident(maz)
                    mazoi=ident(mazo)
                    marzi=ident(marz)
                    marzoi=ident(marzo)

                    aznc= [count_negs(azn[:,:,:,i]) for i in range(nparcs)] 
                    arznc=[count_negs(arzn[:,:,:,i]) for i in range(nparcs)]   

            print(f'\n av grouped wholebrain: ', end="")
            for string in ['mai','mari','maoi','maroi']:
                print(f"{string} {eval(f'{string}.mean()'):.0f}, ", end="")
            print('\n av blockwise: ', end="")
            for string in ['ai','ari','aoi','aroi']:
                print(f"{string} {eval(f'{string}.mean()'):.0f}, ",end="")   

            if block_choice=='few_from_each_vertex': 
                print(f'\n av grouped perparcel: ', end="")
                for string in ['mazi','marzi','mazoi','marzoi']:
                    print(f"{string} {eval(f'{string}.mean()'):.0f}, ", end="")

            print('')

        print(f'{c.time()}: Calculations done, Ident start')
        if to_plot and block_choice=='few_from_each_vertex':
            if get_similarity_average:
                hutils.plot_parc_multi(p,align_parc_matrix,['mazi','mazoi','marzoi','aznc','arznc'],[mazi,mazoi,marzoi,aznc,arznc])     
            
            """
            parc_sizes=np.array(align_parc_matrix.sum(axis=1)).squeeze()
            hutils.plot_parc(p,align_parc_matrix,parc_sizes,'parc_sizes') #confounder
            if 'all_aligners' in locals(): #confounder
                scales = np.vstack( [[all_aligners[i].fit_[nparc].scale for nparc in range(nparcs)] for i in range(len(all_aligners))] )   
                scales_mean=scales.mean(axis=0)    
                hutils.plot_parc(p,align_parc_matrix,scales_mean,'scales')        
            """
        def plots_average():             
            if to_plot:
                fig,axs=plt.subplots(5,3)
                tutils.plot_id(axs[0,0],ma,title='ma')
                tutils.plot_id(axs[1,0],mar,title='mar')
                tutils.plot_id(axs[2,0],maro,title='maro')
                tutils.plot_id(axs[3,0],a[:,:,0],title='a_block0')
                tutils.plot_id(axs[4,0],ar[:,:,0],title='ar_block0')
                tutils.hist_ravel(axs[0,1],-an,'an')
                tutils.hist_ravel(axs[1,1],-arn,'arn')
                tutils.hist_ravel(axs[2,1],-arnm,'arnm: block means (across sub-pairs)')
                tutils.hist_ravel(axs[3,1],-marn,'marn: sub-pair means (across blocks)')
                tutils.hist_ravel(axs[0,2],carn,'carn: counts for each connectome subject',vline=50)
                tutils.hist_ravel(axs[1,2],arnc,'arnc: counts for each block (across sub-pairs)',vline=50)
                tutils.hist_ravel(axs[2,2],anc,'anc: counts for each connectome subject',vline=50)
                tutils.hist_ravel(axs[3,2],can,'can: counts for each block (across sub-pairs)',vline=50)
                plt.subplots_adjust(hspace=0.5) 
                fig.suptitle(f'Similarity average', fontsize=16)

            print(f'AV R {count_negs(arnm):.1f}% of blocks (mean across sub-pairs)')
            print(f'AV R {count_negs(marn):.1f}% of sub-pairs (mean across blocks)')    
            print(f'AV R {count_negs(arn):.1f}% of (sub-pairs)*blocks')   

        if get_similarity_average:
            plots_average()
            if to_plot and plot_type=='save_as_html': plt.savefig(f'{figures_subfolder}/av')
        if to_plot: plt.show()
        hutils.getloadavg()
        print(hutils.memused())
        return sorter, unsorter, slices, smoother_pre, smoother_post, figures_subfolder, p, c, subs, align_labels, align_parc_matrix, nparcs, blocks, hr, fa, a, an, arn


    load_file=False
    save_file=False  
    to_plot=True
    plot_type='open_in_browser' #save_as_html

    if load_file:
        load_file_path = 'Amovf0123t0_S300_Tmovf0123t0sub30to40_G0ffrr_TempScal_gam0.2_B100la40-90_Dtracks_5M_1M_end_3mm_3mm_RDRT_S40_40-50'
        load_file_path = 'Amovf0123t0_S300_Tmovf0123t0sub20to30_RG0ffrr_TempScal_gam0.3_B100la20-30_Dtracks_5M_1M_end_3mm_0mm_RDRT_S10_30-80'
        #load_file_path = 'Amovf0123t0_S300_Tmovf0123t0sub20to30_RG0ffrr_TempScal_gam0.3_B5fe20-30_Dtracks_5M_1M_end_3mm_0mm_RDRT_S10_30-60'
        nblocks,block_choice,howtoalign,pre_hrc_fwhm,post_hrc_fwhm,alignfiles,tckfile,subs_test_range_start, subs_test_range_end = tutils.extract_tkalign_corrs2(load_file_path)
        save_file=False

    else:
        nblocks=20 #how many (parcel x parcel) blocks to examine
        block_choice='largest' #'largest', 'fromsourcevertex', 'all','few_from_each_vertex'
        howtoalign = 'RDRT' #'RDRT','RD','RD+','RT','RT+'     
        
        pre_hrc_fwhm=3 #smoothing kernel (mm) for high-res connectomes. Default 3
        post_hrc_fwhm=3 #smoothing kernel after alignment. Default 3

        tckfile = tutils.get_tck_file()
        #tckfile = 'tracks_5M_1M_end.tck'
        
        #Multiple gamma values
        """
        alignfile_pre = 'Amovf0123t0_S300_Tmovf0123t0sub0to10_G0ffrr_TempScal_'
        alignfile_pre = 'Aresfcf0123t0S1000t_S300_Tresfcf0123t0S1000tsub0to10_G0ffrr_TempScal_'
        alignfile_pre = 'Amovf0123t0_S300_Tmovf0123t0sub0to10_RG0ffrr_TempScal_'
        alignfile_pre = 'Aresfcf0123t0S1000t_S300_Tresfcf0123t0S1000tsub0to10_RG0ffrr_TempScal_
        alignfiles = [f'{alignfile_pre}gam{gam}' for gam in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]]
        alignfiles = [alignfile_pre] + alignfiles
        """
        #alignfiles = ['Amovf0123t0_S300_Tmovf0123t0sub0to3_G0ffrr_TempScal_gam0.2'] #DESKTOP
        #alignfiles = ['Amovf0123t0_S300_Tmovf0123t0sub0to3_RG0ffrr_TempScal_gam0.2'] #DESKTOP R
        #alignfiles = ['Amovf0t0_R5_Tmovf0t0sub0to2_RG0ffrr_TempScal_gam0.3'] #DESKTOP R searchlight
        #alignfiles = ['Amovf0123t0_S300_Tmovf0123t0sub30to40_G0ffrr_TempScal_gam0.2'] #MOVIE
        #alignfiles = ['Amovf0123t0_S300_Tmovf0123t0sub20to30_RG0ffrr_TempScal_gam0.3'] #MOVIE R
        #alignfiles = ['Aresfcf0123t0S1000t_S300_Tresfcf0123t0S1000tsub20to30_RG0ffrr_TempScal_gam0.3'] #REST R
        #alignfiles = ['Amovf0123t0_R10_Tmovf0123t0sub20to30_RG0ffrr_TempScal_gam0.3'] #MOVIE R searchlight

        alignfiles = ['Amovf0t0_S300_Tmovf0t0sub0to5_RG0ffrr_TempScal_'] #bad for some reason
        #alignfiles = ['Amovf0t3_S300_Tmovf0t3sub5to10_RG0ffrr_TempScal_gam0.2'] 
        alignfiles = ['Amovf0t0_S300_Tmovf0t0sub5to10_RG0ffrr_TempScal_gam0.2']
    
    for alignfile in alignfiles:

        parcellation_string = tutils.extract_alignpickles3(alignfile,'parcellation_string')
        MSMAll = tutils.extract_alignpickles3(alignfile,'MSMAll')
        if tutils.extract_alignpickles3(alignfile,'template_making_string')[0]=='R': 
            align_template_to_imgs=True
            template_sub_start = tutils.extract_alignpickles3(alignfile,'template_sub_start')
            template_sub_end = tutils.extract_alignpickles3(alignfile,'template_sub_end')
            print("Align template to imgs")
        else: 
            align_template_to_imgs=False
        subs_blocks_range = range(0,10) #subjects to use to determine blocks with most streamlines, range(10,30)
        for subs_test_range in [range(0,5)]: #range(20,30)
            temp = [i for i in subs_blocks_range if i not in subs_test_range]
            #temp = [i for i in range(0,5)]
            if load_file:
                assert(subs_test_range_start==subs_test_range.start and subs_test_range_end==subs_test_range.stop)
            if align_template_to_imgs: #that template_subs are consistent between the aligner and this script
                assert(min(temp)==template_sub_start and max(temp)==template_sub_end-1)
            subs_inds={'temp': temp, 'test': subs_test_range, 'blocks': subs_blocks_range}
            print(f'{c.time()}: Getting data start', end=", ")
            sorter, unsorter, slices, smoother_pre, smoother_post, figures_subfolder, p, c, subs, align_labels, align_parc_matrix, nparcs, blocks, hr, fa, a, an, arn = func(subs_inds,nblocks,alignfile,howtoalign,block_choice,save_file,load_file,to_plot,plot_type,pre_hrc_fwhm,post_hrc_fwhm,MSMAll, tckfile=tckfile, align_template_to_imgs=align_template_to_imgs)

            assert(0)

            ### DOOMSDAY TESTING ###
            
            norm = lambda x: hutils.subtract_parcelmean(x,align_labels)
            from tkalign_utils import ident_plot

            print(f'{c.time()}: Calculating spatial maps', end=", ")

            nsubjects = len(subs_inds['test'])
            subjects = [hutils.all_subs[i] for i in subs_inds['test']]
            hr_strengths = [np.squeeze(np.array(i.sum(axis=0)))[unsorter] for i in hr['test']] #list (subjects) of high-res connectome node strengths
            rowsums = tutils.get_rowsums(fa['test'],align_labels,axis=0)   

            max_sub = 4 #only use first max_sub subjects (or None)
            if max_sub is not None:
                nsubjects = max_sub
                subjects = subjects[0:max_sub]
                hr_strengths = hr_strengths[0:max_sub]
                rowsums = rowsums[0:max_sub]

            assert(0)

            import getmesh_utils
            meshes = [getmesh_utils.get_verts_and_triangles(subject,'white') for subject in subjects]
            sulcs = [hutils.get_sulc(i) for i in subjects] #list (subjects) of sulcal depth maps

            print(f'{c.time()}: Get movies', end=", ")       
            movs,_ = hutils.get_movie_or_rest_data(subjects,'movie',runs=[0,1],fwhm=0,clean=True,MSMAll=False)          

            print(f'{c.time()}: Get maxcorr maps', end=", ")
            movs_maxcorr = Parallel(n_jobs=-1,prefer='processes')(delayed(tutils.get_max_within_parcel_corrs)(mov,align_labels,nparcs) for mov in movs) #list (subjects) of maxcorr maps. For each vertex, maximum correlation with any other vertex in the same parcel
            print(f'{c.time()}: Get nearest vertices', end=", ")  
            mask = hutils.get_fsLR32k_mask()
            import biasfmri_utils as butils
            temp = Parallel(n_jobs=-1,prefer='processes')(delayed(butils.get_nearest_vertices)(meshes[i],mask) for i in range(nsubjects))
            nearest_vertices,nearest_distances = [list(item) for item in zip(*temp)]
            print(f'{c.time()}: corr neighbour start', end=", ")   
            movs_nextcorr = Parallel(n_jobs=-1,prefer='processes')(delayed(butils.get_corr_with_neighbours)(nearest_vertices[i],movs[i]) for i in range(nsubjects))

            """
            print(f'{c.time()}: Get temporal autocorrs', end=", "
            movs_autocorr = Parallel(n_jobs=-1,prefer='processes')(delayed(hutils.temporal_autocorr)(i,1) for i in imgs)     
            print(f'{c.time()}: Get vertex areas start', end=", ")
            vertex_areas = Parallel(n_jobs=-1,prefer='processes')(delayed(hutils.get_vertex_areas)(meshes[i]) for i in range(nsubjects))
            """

            structdata = sulcs
            funcdata = movs_maxcorr

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
            cci=tutils.identifiability(cc)
            

        '''
        for mesh_sub in [0,1]:
            p.mesh = meshes[mesh_sub]
            p.plot(sulcs[0],cmap='bwr')
            p.plot(norm(hutils.mylog10(hr_strengths[0])),cmap='bwr')
            p.plot(norm(rowsums[0]),cmap='bwr')
            p.plot(sulcs[1],cmap='bwr')
            p.plot(norm(hutils.mylog10(hr_strengths[1])),cmap='bwr')
            p.plot(norm(rowsums[1]),cmap='bwr') 
            p.plot(movs_nextcorr[0],cmap='inferno')
            p.plot(movs_nextcorr[1],cmap='inferno')
            p.plot(movs_maxcorr[0],cmap='inferno')
            p.plot(movs_maxcorr[1],cmap='inferno')

            #Plots of intersubject identifiability (all parcels together)
            """
            ident_plot(sulcs,'sulcs',rowsums,'rowsums')
            ident_plot(hr_strengths,'hr_strengths',rowsums,'rowsums')
            ident_plot(sulcs,'sulcs',vertex_areas,'vertex_areas')
            ident_plot(hr_strengths,'hr_strengths',vertex_areas,'vertex_areas')
            ident_plot(sulcs,'sulcs',nearest_distances,'nearest_distances')
            ident_plot(hr_strengths,'hr_strengths',nearest_distances,'nearest_distances')
            ident_plot(sulcs,'sulcs',movs_autocorr,'movs_autocorr')
            ident_plot(hr_strengths,'hr_strengths',movs_autocorr,'movs_autocorr')
            ident_plot(sulcs,'sulcs',movs_nextcorr,'movs_nextcorr')
            ident_plot(hr_strengths,'hr_strengths',movs_nextcorr,'movs_nextcorr')

            ident_plot(rowsums,'rowsums',vertex_areas,'vertex_areas')
            ident_plot(rowsums,'rowsums',nearest_distances,'nearest_distances')
            ident_plot(rowsums,'rowsums',movs_autocorr,'movs_autocorr')
            ident_plot(rowsums,'rowsums',movs_nextcorr,'movs_nextcorr')

            ident_plot(vertex_areas,'areas',nearest_distances,'nearest_distances')

            ident_plot(vertex_areas,'areas',movs_autocorr,'movs_autocorr')
            ident_plot(nearest_distances,'nearest_distances',movs_autocorr,'movs_autocorr')
            ident_plot(vertex_areas,'areas',movs_nextcorr,'movs_nextcorr')
            ident_plot(nearest_distances,'nearest_distances',movs_nextcorr,'movs_nextcorr')
           
            ident_plot(movs_autocorr,'movs_autocorr',movs_nextcorr,'movs_nextcorr')
            """
            #structdata = sulcs
            #structdata = hr_strengths
            structdata = sulcs 
            funcdata = movs_nextcorr 

            i=0
            for structdata in [sulcs,hr_strengths]:
                for funcdata in [rowsums,movs_nextcorr,movs_maxcorr]:
                    tutils.ident_from_data(structdata,funcdata,nsubjects,nparcs,nblocks,blocks,align_labels,arn)
                    i+=1
                    print(f"i={i}")

        '''


