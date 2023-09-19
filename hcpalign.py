if __name__=='__main__':

    import nibabel as nib
    nib.imageglobals.logger.level = 40  #suppress pixdim error msg
    import hcp_utils as hcp
    import numpy as np
    import os
    from sklearn.svm import LinearSVC
    
    import hcpalign_utils as hutils
    from hcpalign_utils import ospath
    from Connectome_Spatial_Smoothing import CSS as css
    from my_surf_pairwise_alignment import MySurfacePairwiseAlignment, LowDimSurfacePairwiseAlignment
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    import psutil

    hcp_folder=hutils.hcp_folder
    intermediates_path=hutils.intermediates_path
    results_path=hutils.results_path
    available_movies=hutils.movies 
    available_rests=hutils.rests
    available_tasks=hutils.tasks 

    def func(\
        c,t,subs,nalign,align_string, ndecode, decode_string, n_jobs=-1,parcellation_string='S300',method='pairwise',alignment_method='scaled_orthogonal',alignment_kwargs={}, kfolds=5, lowdim_vertices=False, lowdim_ncomponents=300, MSMAll=False, descale_aligner=False, absValueOfAligner=False, scramble_aligners=False,post_decode_fwhm=0, load_pickle=False, save_pickle=False,\
        plot_any=False, plot_impulse_response=False, plot_contrast_maps=False, plot_scales=False,return_aligner=False,\
        args_template={'n_iter':2,'scale':False,'method':1,'nsubsfortemplate':'all','pca_template': False},\
        args_maxcorr={'max_dist':10,'typeof':3},\
        reg=0):
        """
        c: a hcpalign_utils.clock object
        t: If a hcpalign_utils.cprint object is given here, will print to both console and text file 
        subs: list of subject IDs
        nalign: list of alignment data (nsubs) (nvertices,nsamples) for each subject
        nalign_string: description string for the type of alignment data
        ndecode: list of decoding data (nsubs) (ncontrasts,nvertices) for each subject
        decode_string: description string for the type of decoding data
        n_jobs: processor cores for a single source to target pairwise aligment (1 parcel per thread) (my PC has 12), ie for the 'inner loop'. -1 means use all cores
        nparcs: no. of parcels. Subjects aligned within each parcel
        method: anat, intra_subject, pairwise, template
        alignment_method: scaled_orthogonal, permutation, optimal_tarnsport, ridge_cv
        alignment_kwargs: dict
            Additional keyword arguments to pass to the alignment method
        kfolds: default 5 (for classification)
        lowdim_vertices: to use PCA/ICA to reduce nalign dimensionality along the nvertices axis 
        lowdim_ncomponents: default 300, works with lowdim_samples or lowdim_vertices
        MSMAll: False for MSMSulc
        descale_aligner: remove scale factor from aligner
        absValueOfAligner: to take elementwise absolute value of the aligner
        scramble_aligners:  to transform subjects' task maps using someone else's aligner  
        post_decode_fwhm smooths the left-out contrast maps predicted through alignment
        plot_any #plot anything at all, or not
        plot_impulse_response #to plot aligned image of circular ROIs
        plot_contrast_maps #to plot task contrast maps (predicted through alignment)
        plot_scales #to plot scale parameter for Procrustes aligner
        save_pickle: to save aligner as pickle
        args_diffusion: only relevant if align_with='diffusion'
            sift2: True or False
            tckfile e.g. 'tracks_5M' (default with sift2=True),'tracks_5M_sift1M.tck' (default with sift2=False) 'tracks_5M_sift1M.tck' 'tracks_5M_50k.tck', 'tracks_5M_sift1M_200k.tck'
            diffusion_targets_nparcs: False, or a number of parcels [100,300,1000,3000,10000] Number of connectivity targets for DA     
            diffusion_targets_nvertices: number of random vertices to use as targets for diffusion alignment
            fwhm_circ: 
        args_template: only relevant for template alignment 
            n_iter: default 2
            scale: True or False for scale_template
            method: 1, 2, or 3 
            nsubsfortemplate: 'all' or an iterable, e.g. [0,2,4], or range(3)
            pca_template: True to use new PCA-derived template
        reg: regularization parameter for MySurfPairwiseAlignment: default 0
        """
        
        #so that print will print to text file too
        if t is not None:
            print=t.print 
        

        n_subs=len(subs)
        nsubs = np.arange(len(subs)) #number of subjects
        nlabels = [np.array(range(i.shape[0])) for i in ndecode]

        if lowdim_vertices:
            lowdim_method='pca' #'pca','ica'
            lowdim_ncomponents=lowdim_ncomponents 
        if lowdim_vertices: assert(method in ['pairwise','template'])
   
        post_decode_smooth=hutils.make_smoother_100610(post_decode_fwhm)

        save_suffix=f"A{align_string}_D{decode_string}_{method[0:4]}_{alignment_method}_{parcellation_string}_{n_subs}_{post_decode_fwhm}{str(descale_aligner)[0]}{str(absValueOfAligner)[0]}{str(scramble_aligners)[0]}"

        """
        if not(args_template['nsubsfortemplate']=='all'):
            save_suffix=f"{save_suffix}_template{len(args_template['nsubsfortemplate'])}"
        if not(args_template['n_iter']==2):
            save_suffix=f"{save_suffix}_niter{args_template['n_iter']}"
        if not (args_template['method']==1):
            save_suffix=f"{save_suffix}_meth{args_template['method']}"
        if args_template['pca_template']==True:
            save_suffix=f"{save_suffix}_pcatemp"
        """
        if reg:
            save_suffix=f"{save_suffix}_reg{reg}"
        #Set up for saving pickled data and for plotting
        if alignment_method=='optimal_transport':
            import dill
            pickle=dill
        else:
            import pickle      
        hutils.mkdir(f'{intermediates_path}/alignpickles')
        save_pickle_filename=ospath(f'{intermediates_path}/alignpickles/hcpalign_{save_suffix}.p')                          
        if plot_any:
            plot_dir=f'/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/figures/hcpalign/{save_suffix}'
            p=hutils.surfplot(plot_dir,plot_type='open_in_browser')

        print(save_suffix)


        if load_pickle: assert(os.path.exists(ospath(save_pickle_filename)))


        #Get parcellation and classifier
        clustering = hutils.parcellation_string_to_parcellation(parcellation_string)
        classifier=LinearSVC(max_iter=10000,dual='auto')     
        n_splits=min(kfolds,n_subs) #for cross validation
        kf=KFold(n_splits=n_splits)
        classification_scores=[]

        ###PAIRWISE ALIGNMENT###
        if method == 'pairwise':      
            if load_pickle:
                print('loading aligners')
                aligners = pickle.load(open(ospath(save_pickle_filename), "rb" ))
            else:        
                if alignment_method=='maxcorr':
                    typeof=args_maxcorr['typeof']
                    max_dist=args_maxcorr['max_dist']                  
                    if typeof==1:
                        import get_gdistances
                        dists=get_gdistances.get_gdistances('100610','midthickness',max_dist,load_from_cache=True,save_to_cache=False)  
                    else:
                        from make_gdistances_full import get_saved_gdistances_full
                        x=get_saved_gdistances_full('100610','midthickness')
                        if typeof==3:
                            from scipy.sparse import csr_matrix
                            x = x < max_dist
                            dists = csr_matrix((x[x.nonzero()],x.nonzero()),shape=x.shape)
                    print(f'nnz is {dists.getnnz()}, first row size {dists.indptr[1]}')

                def initialise_aligner():
                    if alignment_method=='maxcorr':                          
                        aligner=hutils.maxcorr(dists=dists,max_dist=max_dist,typeof=typeof)
                    elif lowdim_vertices: 
                        aligner=LowDimSurfacePairwiseAlignment(alignment_method=alignment_method, clustering=clustering,n_jobs=n_jobs,reg=reg,n_components=lowdim_ncomponents,lowdim_method=lowdim_method)
                    else: 
                        aligner=MySurfacePairwiseAlignment(alignment_method=alignment_method, clustering=clustering,n_jobs=n_jobs,alignment_kwargs=alignment_kwargs,reg=reg)  #faster if fmralignbench/surf_pairwise_alignment.py/fit_parcellation uses processes not threads. MAKE IT PROCESSES!???
                    return aligner
                def fit_aligner(source_align, target_align, absValueOfAligner, descale_aligner,aligner):
                    aligner.fit(source_align, target_align)
                    if absValueOfAligner: hutils.aligner_absvalue(aligner)
                    if descale_aligner: hutils.aligner_descale(aligner)
                    return aligner

                print(hutils.memused())  
                print(f'{c.time()}: Calculate aligners')
                import itertools       
                temp=Parallel(n_jobs=-1,prefer='processes')(delayed(fit_aligner)(nalign[source],nalign[target], absValueOfAligner, descale_aligner,initialise_aligner()) for source,target in itertools.permutations(nsubs,2))
                aligners={}
                index=0
                for source,target in itertools.permutations(nsubs,2):
                    aligners[f'{subs[source]}-{subs[target]}'] = temp[index]
                    index +=1
                del temp
                if save_pickle: 
                    pickle.dump(aligners,open(save_pickle_filename,"wb"))

            print(hutils.memused())  
            print(f'{c.time()}: Aligning decode data start')                    
            all_aligned_sources_decode=[]
            for target in nsubs:
                sources=[i for i in nsubs if i!=target]  
                aligned_sources_decode=[]
                for source in sources:
                    source_decode = ndecode[source]
                    aligner=aligners[f'{subs[source]}-{subs[target]}'] 
                    aligned_sources_decode.append( post_decode_smooth(aligner.transform(source_decode)) )
                aligned_sources_decode = np.vstack(aligned_sources_decode)
                all_aligned_sources_decode.append(aligned_sources_decode)

            print(hutils.memused())  
            print(f'{c.time()}: Classification start')  
            from sklearn.base import clone
            def classify_pairwise(target,nsubs,nlabels,target_decode,target_labels,aligned_sources_decode,classifier):
                sources=[i for i in nsubs if i!=target]
                sources_labels=[nlabels[i] for i in sources] 
                clf=clone(classifier)
                clf.fit(aligned_sources_decode, np.hstack(sources_labels))
                return clf.score(target_decode, target_labels)
            classification_scores = Parallel(n_jobs=-1)(delayed(classify_pairwise)(target,nsubs,nlabels,ndecode[target],nlabels[target],all_aligned_sources_decode[target],classifier) for target in nsubs)               

        elif method=='template':
        ###TEMPLATE ALIGNMENT###           
            if load_pickle:
                print('loading aligner')
                aligners = pickle.load(open(ospath(save_pickle_filename), "rb" ))
            else:       
                from my_template_alignment import MyTemplateAlignment, LowDimTemplateAlignment, get_template            
                if lowdim_vertices: aligners=LowDimTemplateAlignment(alignment_method,clustering=clustering,n_jobs=n_jobs,n_iter=args_template['n_iter'],n_components=lowdim_ncomponents,lowdim_method=lowdim_method,scale_template=args_template['scale'],template_method=args_template['method'],reg=reg)
                else: aligners=MyTemplateAlignment(alignment_method,clustering=clustering,n_jobs=n_jobs,n_iter=args_template['n_iter'],scale_template=args_template['scale'],template_method=args_template['method'],reg=reg)
                print(hutils.memused()) 

                if args_template['pca_template']==True:
                    if args_template['nsubsfortemplate']=='all':
                        aligners.template=get_template(clustering,nalign)
                    else:
                        aligners.template=get_template(clustering,[nalign[i] for i in args_template['nsubsfortemplate']])
                    aligners.fit_to_template(nalign)
                elif args_template['pca_template']==False:
                    if args_template['nsubsfortemplate']=='all':
                        aligners.fit(nalign) 
                    else:
                        aligners.fit([nalign[i] for i in args_template['nsubsfortemplate']])
                        print(f'{c.time()}: Fitting rest of imgs to template')
                        aligners.fit_to_template(nalign)
                print(hutils.memused()) #XXX
                if absValueOfAligner:
                    [aligners.estimators[j].absValue() for j in range(len(aligners.estimators))]
                if descale_aligner:
                    [aligners.estimators[j].descale() for j in range(len(aligners.estimators))]
                print('{} Template align done'.format(c.time())) 
                if return_aligner: 
                    print(hutils.memused())
                    return aligners
                if save_pickle: 
                    print('saving aligner')
                    pickle.dump(aligners,open(save_pickle_filename,"wb"))
                
            #Taligner.estimators is list(nsubs) of SurfacePairwiseAlignments, from each subj to template       
            ndecode_aligned=[post_decode_smooth(aligners.transform(ndecode[i],i)) for i in range(len(ndecode))]
            if plot_any:
                if plot_impulse_response:
                    source=0 #aligner for plot is sub 0 to template
                    aligner = aligners.estimators[source] 
                    hutils.do_plot_impulse_responses(p,'',aligner,method,lowdim_vertices)
                    assert(0) #XXXXX
                if plot_contrast_maps:
                    for i in range(1):
                        for contrast in [3]: #Visualise predicted contrast map   
                            p.plot(ndecode[i][contrast,:],'Con{}_sub{}'.format(contrast,i))
                            p.plot(ndecode_aligned[i][contrast,:],'_Con{}_subTemplate_from_sub{}'.format(contrast,i))
                if plot_scales:
                    p.plot(hutils.aligner_get_scale_map(clustering,aligners.estimators[source]),'scale') #plot scale parameter for Procrustes aligner

            del aligners
            print('{} Template transform done'.format(c.time()))
            hutils.getloadavg()
           
        elif method=='anat':
            ndecode_aligned=[post_decode_smooth(ntask) for ntask in ndecode]
        del nalign

        if method in ['template','anat']: 
            
            #To see accuracy for all subjects
            
            X=np.vstack(ndecode_aligned)
            y=np.hstack(nlabels)
            num_of_tasks=ndecode_aligned[0].shape[0]
            subjects=np.hstack([np.tile([i],num_of_tasks) for i in range(n_subs)])
            from sklearn.model_selection import GroupKFold,cross_val_score, GridSearchCV
            gkf=GroupKFold(n_splits=n_splits)       
            classification_scores=cross_val_score(classifier,X,y,groups=subjects,cv=gkf,n_jobs=-1)       
            
            
            #To see accuracy for subsets of subjects
            """
            for gp in [range(10,20)]:
                X=np.vstack([ndecode_aligned[i] for i in gp])
                y=np.hstack([nlabels[i] for i in gp])
                num_of_tasks=ndecode_aligned[0].shape[0]
                subjects=np.hstack([np.tile([i],num_of_tasks) for i in range(len(gp))])
                from sklearn.model_selection import GroupKFold,cross_val_score, GridSearchCV
                gkf=GroupKFold(n_splits=n_splits)       
                classification_scores=cross_val_score(classifier,X,y,groups=subjects,cv=gkf,n_jobs=-1) 
                print('Subgroup mean classification accuracy {:.2f}'.format(np.mean([np.mean(i) for i in classification_scores])))  
            """

        print(f'{c.time()} Classifications done')
        print('Mean classification accuracy {:.2f}'.format(np.mean([np.mean(i) for i in classification_scores])))
        
        if plot_any and plot_impulse_response and method != 'anat': 
            hutils.do_plot_impulse_responses(p,'',aligner,method,lowdim_vertices)
        
        print(hutils.memused())
        return classification_scores


###########################################################
    
    resultsfilepath=ospath(f'{results_path}/r{hutils.datetime_for_filename()}.txt')
    with open(resultsfilepath,'w') as resultsfile:
        t=hutils.cprint(resultsfile) 
        c=hutils.clock()   


        subs=hutils.subs[slice(0,3)]
        method='pairwise' #anat, intra_subject, pairwise, template
        alignment_method='scaled_orthogonal' #scaled_orthogonal, permutation, optimal_transport, ridge_cv
        alignment_kwargs = {'scaling':True}
        parcellation_string = 'S300' #S300, K1000, MMP
        save_pickle=False
        load_pickle=False #use saved aligner
        MSMAll=False
        args_template = {'n_iter':1,'do_level_1':False,'normalize_imgs':None,'normalize_template':None,'remove_self':False,'level1_equal_weight':False}


        #Get alignment data. List (nsubjects) of alignment data (ntimepoints, nvertices)
        print(f"{c.time()} Get alignment data start")
        if method=='anat' or load_pickle:
            nalign = [[] for sub in subs] #irrelevant anyway
            align_string = ''
        else:
            nalign,align_string = hutils.get_movie_or_rest_data(subs,'movie',runs=[0],fwhm=0,clean=True,MSMAll=MSMAll)
            #nalign,align_string = hutils.get_aligndata_highres_connectomes(c,subs,MSMAll,{'sift2':False , 'tckfile':'tracks_5M_sift1M.tck' , 'targets_nparcs':False , 'targets_nvertices':16000 , 'fwhm_circ':3 })
        print(f"{c.time()} Get alignment data done")      
        print(hutils.memused())

        if False: #circular shift nalign to scramble
            nalign.append(nalign.pop(0)) 
            nalign.append(nalign.pop(0))
        if False: #reduce dimensionality of alignment data in ntimepoints/nsamples axis using PCA
            nalign, string = hutils.reduce_dimensionality_samples(c,nalign,ncomponents=300,method='pca')
            align_string = f'{align_string}{string}'
            
        #Get decoding data. List (nsubjects) of decode data (ncontrasts, nvertices)
        print(f"{c.time()} Get decoding data start")
        ndecode,decode_string = hutils.get_task_data(subs,available_tasks[0:7],MSMAll=MSMAll)
        print(f"{c.time()} Get decoding data done")

        #decode movie viewing data instead
        """
        ndecode,decode_string = hutils.get_movie_or_rest_data(subs,'movie',runs=[1],fwhm=0,clean=True,MSMAll=MSMAll)
        ndecode, string = hutils.reduce_dimensionality_samples(c,ndecode,ncomponents=20,method='pca')
        decode_string = f'{decode_string}{string}'
        """

        for reg in [0]:
         
            print(hutils.memused())   
            func(c,t,subs, nalign, align_string, ndecode, decode_string, parcellation_string=parcellation_string,method=method ,alignment_method=alignment_method,alignment_kwargs=alignment_kwargs,post_decode_fwhm=0,save_pickle=save_pickle,load_pickle=load_pickle,n_jobs=+1,args_template=args_template,plot_any=False, plot_impulse_response=False, plot_contrast_maps=False,reg=reg)
            print(hutils.memused())
            t.print('')  

    
    print('\a') #beep sounds 