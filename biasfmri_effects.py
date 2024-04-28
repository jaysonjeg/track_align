"""
Code for gyral bias in fMRI surface data
Use env py390

The code is divided into sections which you can (in)activate by setting the relevant boolean to True or False
Can use either real HCP fMRI data or noise data
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
import biasfmri_utils as butils
import nilearn.plotting as plotting
import nibabel as nib
import pickle
import hcp_utils as hcp
from scipy import stats

if __name__=='__main__':

    c = hutils.clock()
    #Set paths
    hcp_folder=hutils.hcp_folder
    intermediates_path=hutils.intermediates_path
    results_path=hutils.results_path
    project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
    biasfmri_intermediates_path = ospath(f'{project_path}/intermediates')

    ### GENERAL PARAMETERS
    sub_slice = slice(0,1)
    real_or_noise = 'real' # 'real' or 'noise
    this_parc = 1 #which parcel for within-parcel analysis
    parc_string='S300'
    surface = 'midthickness' #which surface for calculating distances, e.g. 'white','inflated','pial','midthickness'
    which_subject_visual = '100610' #which subject for visualization. '100610', '102311', 'standard'
    surface_visual = 'white'
    MSMAll = False

    ### PARAMETERS FOR NULL TESTS OF CORRELATIONS BETWEEN TWO MAPS
    null_method = 'spin_test' #spin_test, no_null, eigenstrapping
    n_perm = 100 

    ### PARAMETERS FOR SPECIFIC SUB-SECTIONS

    which_neighbours = 'local' #'local','distant'
    distance_range=(1,10) #Only relevant if which_neighbours=='distant'. Geodesic distance range in mm, e.g. (0,4), (2,4), (3,5), (4,6)
    load_neighbours = False
    save_neighbours = True

    ### PARAMETERS FOR NOISE OR REAL DATA
    subjects=hutils.all_subs[sub_slice]
    nsubjects = len(subjects)
    print(f'{c.time()}: subs {sub_slice}, data is {real_or_noise}, parc {parc_string}, surface {surface}, visualise on {which_subject_visual} {surface_visual}')
    if real_or_noise == 'noise':
        noise_source = 'surface' #'volume' or 'surface'. 'volume' means noise data in volume space projected to 'surface'. 'surface' means noise data generated in surface space
        smooth_noise_fwhm = 2 #mm of surface smoothing. Try 0 or 2
        ntimepoints = 1000 #number of timepoints
        print(f'{c.time()}: Noise data, source: {noise_source}, smooth: {smooth_noise_fwhm}mm, ntimepoints: {ntimepoints}, test first-half, retest second-half')
    elif real_or_noise == 'real':
            img_type = 'movie' #'movie' or 'rest'
            runs = [0]
            #assert(len(runs)%2==0) #even number of runs, to split into test and retest
            print(f'{c.time()}: Real HCP data, img_type: {img_type}, MSMAll: {MSMAll}, runs {runs}, test is first half, retest is second half')


    ### GET DATA
    import getmesh_utils
    if which_subject_visual =='standard':
        p=hutils.surfplot('',mesh = hcp.mesh[surface_visual], plot_type='open_in_browser')
    else:
        vertices_visual,faces_visual = getmesh_utils.get_verts_and_triangles(which_subject_visual,surface_visual,MSMAll)
        p = hutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')
    mask = hutils.get_fsLR32k_mask() #boolean mask of gray matter vertices. Excludes medial wall
    parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
    parc_matrix = hutils.parcellation_string_to_parcmatrix(parc_string)
    nparcs = parc_labels.max()+1


    print(f'{c.time()}: Get meshes')
    import getmesh_utils
    meshes = [getmesh_utils.get_verts_and_triangles(subject,surface,MSMAll) for subject in subjects]
    meshes = [(hutils.cortex_64kto59k(vertices),hutils.cortex_64kto59k_for_triangles(faces)) for vertices,faces in meshes] #downsample from 64k to 59k
    all_vertices, all_faces = zip(*meshes)
    sulcs = [butils.get_sulc(i)[mask] for i in subjects] #list (subjects) of sulcal depth maps

    print(f'{c.time()}: Get fMRI data')   
    if real_or_noise == 'noise':
        if noise_source=='volume':
            #Get noise data in volume space which has been projected to surface
            def get_vol2surf_noisedata(which_subject):
                noise_data_prefix = f'{which_subject}.32k_fs_LR.func'
                noise_data_path = ospath(f'{project_path}/{noise_data_prefix}.npy')
                noise = np.load(noise_data_path).astype(np.float32) #noise data in volume space projected to surface (vol 2 surf)
                if noise.shape[1] > 59412:
                    print("Removing non-gray vertices from noise fMRI data")
                    noise = noise[:,mask]
                return noise     
            ims = Parallel(n_jobs=-1,prefer='threads')(delayed(get_vol2surf_noisedata)(subject) for subject in subjects)
        elif noise_source=='surface':
            #Generate noise data in surface space
            ims = [np.random.randn(ntimepoints,59412).astype(np.float32) for i in range(nsubjects)]
        if smooth_noise_fwhm>0: #smooth the noise data using geodesic distances from subject-specific meshes
            fwhm_values_for_gdist = np.array([3,5,10]) #fwhm values for which geodesic distances have been pre-calculated
            fwhm_for_gdist = fwhm_values_for_gdist[np.where(fwhm_values_for_gdist>smooth_noise_fwhm)[0][0]] #find smallest value greater than fwhm in the above list
            print(f'{c.time()}: Generate smoothers start')
            skernels = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.get_smoothing_kernel)(subject,surface,fwhm_for_gdist,smooth_noise_fwhm,MSMAll) for subject in subjects)
            print(f'{c.time()}: Smooth noise data start')
            ims = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.smooth)(im,skernel) for im,skernel in zip(ims,skernels))
            print(f'{c.time()}: Smoothing finished')
    elif real_or_noise == 'real':
        ims,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs,fwhm=0,clean=True,MSMAll=MSMAll)
    print(f'{c.time()}: Each subject data shape is {ims[0].shape}')

    ### Bias in fMRI-based parcellation (do left hemisphere alone)
    do_bias_parcellation = False
    if do_bias_parcellation:
        ngrayl = len(hcp.vertex_info.grayl) #left hemisphere only
        
        tstats_all_subjects = []
        cohend_all_subjects = []
        for nsubject in range(nsubjects): #range(nsubjects)

            use_precomputed_parcellation=False
            if use_precomputed_parcellation:
                #Use standard mesh sulc vaule, and pre-computed parcellation labels
                parcellation_string = 'M'
                print(f"Using standard mesh sulc values and pre-computed parcellation labels from {parcellation_string}")
                labels = hutils.parcellation_string_to_parcellation(parcellation_string)[0:ngrayl]
                labels = np.expand_dims(labels,1) #change from (nvertices,) to (nvertices,1)
                sulc = -hcp.mesh.sulc[mask] #because this standard sulc values are flipped
                faces = butils.triangles_removenongray(hcp.mesh.midthickness_left[1],mask)
                _,edges = butils.faces2connectivity(faces)
            else:
                #Use clustering to generate your own functional parcellation
                n_clusters = 50 #how many functional clusters
                n_repeats_per_subject = 1 #how many different random parcellations to find per subject
                imgt = ims[nsubject].T
                sulc = sulcs[nsubject] #could put (nsubjects+1) to compare to the sulc map of a different subject
                faces = meshes[nsubject][1]
                print(f'{c.time()}: Faces to structural adjacency matrix')
                connectivity,edges = butils.faces2connectivity(faces)
                imgt = imgt[0:ngrayl,:]
                connectivity = connectivity[:,0:ngrayl][0:ngrayl,:]    
                print(f'{c.time()}: Agglomerative clustering')
                def do_agglomerative_clustering():
                    from sklearn.cluster import AgglomerativeClustering
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', connectivity=connectivity, linkage='ward').fit(imgt)
                    return clustering.labels_
                labels = Parallel(n_jobs=1,prefer='threads')(delayed(do_agglomerative_clustering)() for i in range(n_repeats_per_subject))
                labels = np.vstack(labels).T


            #Remove edges which are outside the left hemisphere
            sulc_left = sulc[0:ngrayl] #left-sided sulcal depth map 
            edges_left = butils.find_edges_left(edges, ngrayl)

            #Get border vertices
            print(f'{c.time()}: Find border vertices start')
            border = butils.get_border_vertices(ngrayl,edges_left,labels)
            border = np.sum(border,axis=1)
            border_bool = border>0
            labels = labels[:,0] #first repetition

            #Convert L hemisphere data to bihemispheric brain data (with zeros in the R hemi)
            border_full = np.zeros(59412)
            border_full[0:ngrayl] = border
            parcels_full = np.zeros(59412)
            parcels_full[0:ngrayl] = labels
            sulc_border = sulc_left[border_bool] #sulcal depth values at the border of parcels
            sulc_nonborder = sulc_left[~border_bool] #sulcal depth values not at the border of parcels
        

            cohend_all_subjects.append(butils.get_cohen_d(sulc_border,sulc_nonborder))
            tstats_all_subjects.append(stats.ttest_ind(sulc_border,sulc_nonborder)[0])

            plot_single_subject_parcellations = False
            if plot_single_subject_parcellations:
                p.plot(sulc)
                p.plot(border_full,cmap='inferno') #Plot border points
                p.plot(parcels_full,cmap='tab20') #Plot parcellation
                fig,axs=plt.subplots(figsize=(4,3))
                ax=axs
                vp = ax.violinplot([sulc_border,sulc_nonborder],showmeans=True,showextrema=True)
                ax.set_ylabel('Sulcal depth')
                ax.set_xticks([1,2],labels=['Border','Non-border'])
                ax.set_xlabel('Vertex location')
                """
                axs[1].scatter(sulc_left,border,1,alpha=0.05,color='k')
                axs[1].set_xlabel('Sulcal depth')
                axs[1].set_ylabel('Distance from parcel boundary')
                """
                fig.tight_layout()
                plt.show(block=False)
                sulc_both = np.zeros(59412)
                sulc_both[0:ngrayl] = sulc_left
                print(f'{c.time()}: Do spin test start')
                sulc_both_nulls = butils.do_spin_test(sulc_both,mask,n_perm)
                print(f'{c.time()}: Do spin test done')
                sulc_left_nulls = sulc_both_nulls[0:ngrayl]
                cohen_d, t_stat, p_value = butils.ttest_ind_with_nulldata_given(border,sulc_left,sulc_left_nulls)
                print(f"Sulcal depth at parcel borders {np.mean(sulc_border):.3f} vs non-border vertices {np.mean(sulc_nonborder):.3f}: cohens d {cohen_d:.3f}, t(29694)={t_stat:.3f}, spin test p={p_value:.3f}")
                import neuromaps
                corr,pval = neuromaps.stats.compare_images(sulc_left,-border,nulls=sulc_left_nulls,metric='pearsonr')
                print(f"Correlation is {corr:.3f}, spin test p={pval:.3f}")
        
        print(f'{c.time()}: Func parcellation. Cohen\'s d of all participants:')
        print(cohend_all_subjects)
        print(f'Func parcellation. t-statistic of all participants:')
        print(tstats_all_subjects)
        result = stats.ttest_1samp(tstats_all_subjects,0,alternative='greater')
        print(f'T({result.df})={result.statistic:.3f}, p={result.pvalue:.3f}')

        assert(0)

    ### Will the largest PCA components be biased towards the sulci?
    do_pca = False
    if do_pca:
        #calculate PCA components for each array in ims separately 
        print(f'{c.time()}: PCA start')
        from sklearn.decomposition import PCA
        nsubject = 0
        img = ims[nsubject]
        sulc = sulcs[nsubject]
        pca = PCA(n_components=50)
        pca.fit(img)
        pca.components_.shape #shape (ncomponents,nvertices)

        corrs_with_sulc = []
        for component in pca.components_:
            #p.plot(component)
            corrs_with_sulc.append(np.corrcoef(component,sulc)[0,1])
        print(f"How many components are gyral biased? {np.sum(np.array(corrs_with_sulc)>0)}/{len(corrs_with_sulc)}")
        assert(0)

    get_vertex_areas = False
    if get_vertex_areas:
        print(f'{c.time()}: Get vertex areas start')
        vertex_areas = Parallel(n_jobs=-1,prefer='threads')(delayed(hutils.get_vertex_areas)(mesh) for mesh in meshes)
        print(f'{c.time()}: Get vertex areas end')

        plot_vertex_areas = False
        if plot_vertex_areas:
            nsubject=0
            #p.plot(vertex_areas[nsubject])
            fig,axs=plt.subplots(figsize=(4,4))
            axs.scatter(sulcs[nsubject],vertex_areas[nsubject],1,alpha=0.05,color='k')
            axs.set_xlabel('Sulcal depth')
            axs.set_ylabel('Vertex area')
            axs.set_title(f'Correlation: {np.corrcoef(sulcs[nsubject],vertex_areas[nsubject])[0,1]:.3f}')
            fig.tight_layout()
            plt.show(block=False) 
    else:
        vertex_areas = [None]*nsubjects 

    ### Parcellate
    do_parcellate = True
    if do_parcellate:
        print(f'{c.time()}: Parcellate')    
        ims_parc = Parallel(n_jobs=1,prefer='threads')(delayed(butils.parcel_mean)(im,parc_matrix,vertex_area) for im, vertex_area in zip(ims,vertex_areas)) #list (nsubjects) of parcellated time series (ntimepoints,nparcels)
        ims_singleparc = [im[:,parc_labels==1] for im in ims] #subjects' single-parcel time series (ntimepoints,nverticesInParcel)
        sulcs_parc = [butils.parcel_mean(sulc,parc_matrix) for sulc in sulcs] #subjects' parcellated sulcal depth maps (nparcels,)
        sulcs_singleparc = [sulc[parc_labels==1] for sulc in sulcs] #subjects' single-parcel sulcal depth maps (nverticesInParcel,)


    ### Parcel-mean values might be a bit biased towards the sulci within each parcel
    do_parcelmean = False
    if do_parcelmean:
        print(f'{c.time()}: Parcel-mean values biased towards the sulci within each parcel')
        corrs_vert_parc = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.corr_between_vertex_and_parcelmean)(img,img_parc,parc_labels) for img,img_parc in zip(ims,ims_parc)) #correlations between each vertex and its parcel's mean
        for nsubject in range(nsubjects):
            cc = corrs_vert_parc[nsubject]
            #ccs=butils.smooth(cc,skernels[nsubject]) #smooth the correlation values
            sulc = sulcs[nsubject]
            sulc_ranks = np.argsort(np.argsort(sulc)) #ranks of sulcal depth values (0 is sulcus, max is gyrus)
            ccn = butils.subtract_parcelmean(cc,parc_matrix)
            sulcn = butils.subtract_parcelmean(sulc,parc_matrix)
            #p.plot(cc)
            #p.plot(sulcn)
            for pairs in ([['sulc','cc']]): #,['sulc_ranks','cc'],['sulcn','ccn']]):
                xname = pairs[0]
                yname = pairs[1]
                xvals = eval(xname)
                yvals = eval(yname)
                p.plot(yvals,cmap='inferno')

                corr,pval = butils.corr_with_nulls(xvals,yvals,mask,null_method,n_perm)

                #stat = stats.pearsonr(xvals,yvals)
                print(f'{xname} vs {yname}: R={corr:.3f}, p={pval:.3f}')
                fig,ax=plt.subplots(figsize=(3.5,3.5))
                ax.scatter(xvals,yvals,1,alpha=0.05,color='k')
                ax.set_xlabel('Sulcal depth')
                ax.set_ylabel('Correlation with parcel mean')
                fig.tight_layout()
        assert(0)

    ### Get neighbours and correlations
    do_neighbours = True
    if do_neighbours:
        func = lambda subject,mesh: butils.get_subjects_neighbour_vertices(c, subject,surface,mesh,biasfmri_intermediates_path, which_neighbours, distance_range,load_neighbours, save_neighbours,MSMAll)
        print(f'{c.time()}: Get neighbour vertices')  
        temp = Parallel(n_jobs=-1,prefer='threads')(delayed(func)(subject,mesh) for subject,mesh in zip(subjects,meshes))
        all_neighbour_vertices, all_neighbour_distances, all_neighbour_distances_mean = zip(*temp)
        print(f'{c.time()}: Get corr neighbour start')   
        ims_adjcorr, ims_adjcorr_full = zip(*Parallel(n_jobs=1,prefer='threads')(delayed(butils.get_corr_with_neighbours)(all_neighbour_vertices[i],ims[i],parallelize=True) for i in range(nsubjects)))
        ims_adjcorr_parc = [butils.parcel_mean(im,parc_matrix) for im in ims_adjcorr] #subjects' parcel-mean local neighbourhood correlations (nparcels,)
        ims_adjcorr_singleparc = [im[parc_labels==1] for im in ims_adjcorr] #subjects' single-parcel local neighbourhood correlations (nverticesInParcel,)
        print(f'{c.time()}: Get corr neighbour end')   

        ### Interpolate neighbourhood correlations to get plot of correlation vs. geodesic distance
        do_interpolate_corrs = False
        if do_interpolate_corrs:
            print(f'{c.time()}: Interpolate start') 
            from scipy.interpolate import interp1d
            nsubject = 0      
            nvertices = len(ims_adjcorr[0])

            from scipy.optimize import curve_fit
            def exp_func(x, a, b, c):
                return a * np.exp(-b * x) + c
            expfit_params = np.zeros((nvertices,3)) #parameters for exponential fit to each vertex's correlation-distance curve


            minval = int(np.ceil(distance_range[0]))
            minval = max(minval,4) #only interpolate for distances 4mm and above
            maxval = int(np.floor(distance_range[1]+1))
            interp_dists = np.array(range(minval,maxval)) #distance values at which to interpolate, 0mm, 1mm, 2mm, 3mm, etc
            interp_corrs = np.zeros((nvertices,len(interp_dists))) #save interpolated correlations
            for nvertex in range(nvertices):
                if nvertex%10000==0: print(f'{c.time()}: Vertex {nvertex}/{nvertices}')
                distsx = all_neighbour_distances[nsubject][nvertex]
                corrsx = ims_adjcorr_full[nsubject][nvertex]
                f = interp1d(distsx, corrsx, kind='linear', fill_value='extrapolate')
                interp_corrs[nvertex,:] = f(interp_dists)

                #popt, pcov = curve_fit(exp_func, distsx, corrsx,p0=[1.5,0.5,0.1],bounds=((0.5,0.001,-0.1),(2.5,1.0,0.5)))
                #expfit_params[nvertex,:] = popt

            interp_corrs[interp_corrs<0] = 0 #set min and max interpolated values to 0 and 1
            interp_corrs[interp_corrs>1] = 1
            print(f'{c.time()}: Interpolate end')  

            #Test whether interpolated correlations at any distance is associated with sulcal depth, and plot
            print(f"{c.time()}: Checking for associations with sulcal depth. interp_dists is {interp_dists}")
            ndists = len(interp_dists)
            nrows = int(np.ceil(np.sqrt(ndists)))
            fig,axs = plt.subplots(nrows,nrows,figsize=(10,7))
            for i in range(ndists): #iterate through distances, 0mm, 1mm, etc.
                corrs = interp_corrs[:,i]
                p.plot(corrs) #plot interpolated correlation on brain surface
                ax=axs.flatten()[i]
                valid = ((corrs!=0) & (corrs!=1))
                ax.scatter(sulcs[nsubject][valid],corrs[valid],1,alpha=0.05,color='k')
                ax.set_xlabel('Sulcal depth')
                ax.set_ylabel('Interpolated correlation')
                corr,pval = butils.corr_with_nulls(sulcs[nsubject],corrs,mask,null_method,n_perm)
                ax.set_title(f'Distance {interp_dists[i]}mm\nR={corr:.3f}, p={pval:.3f}')
            fig.tight_layout()

            # Plot correlation vs. geodesic distance for some example vertices
            samplevertices = np.linspace(0,nvertices-1,16).astype(int)
            nrows = int(np.ceil(np.sqrt(len(samplevertices))))
            fig,axs = plt.subplots(nrows,nrows,figsize=(10,7))
            for i,nvertex in enumerate(samplevertices):
                ax=axs.flatten()[i]
                distsx = all_neighbour_distances[nsubject][nvertex]
                corrsx = ims_adjcorr_full[nsubject][nvertex]
                interp_corrs_vertex = interp_corrs[nvertex,:]
                ax.scatter(distsx,corrsx,20,alpha=1,color='k')
                ax.plot(interp_dists, interp_corrs_vertex, '-', color='r')
                ax.set_xlim(distance_range[0],distance_range[1])
                ax.set_ylim(0,1)
                ax.set_xlabel('Distance from source vertex (mm)')
                ax.set_ylabel('Correlation')
                ax.set_title(f'Source vertex is {nvertex}')
            fig.tight_layout()
            plt.show(block=False)

            #plot exponential fit parameters
            expfit_param_names = ['Amplitude','Decay rate','Bias']
            fig,axs = plt.subplots(3,figsize=(4,9))
            for i in range(3):
                values = expfit_params[:,i]
                p.plot(values) 
                ax=axs.flatten()[i]
                ax.scatter(sulcs[nsubject],values,1,alpha=0.05,color='k')
                ax.set_xlabel('Sulcal depth')
                ax.set_ylabel(f'{expfit_param_names[i]}')
                corr,pval = butils.corr_with_nulls(sulcs[nsubject],values,mask,null_method,n_perm)
                ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
            fig.tight_layout()

            assert(0)


        plot_neighbours = True
        if plot_neighbours:
            for i in range(nsubjects):
                ims_adjcorr[i][np.isnan(ims_adjcorr[i])] = 0 #replace NaNs with 0s
                #ims_adjcorr[i][ims_adjcorr[i]<0] = 0 #set negative correlations to 0

            parc_means = butils.parcel_mean(np.reshape(ims_adjcorr[0],(1,-1)),parc_matrix)
            p.plot(parc_means @ parc_matrix) #Plot parcel-means of local correlations for subject 0

            for to_subtract_parcelmean in [True]:
                if to_subtract_parcelmean:
                    ims_adjcorr = [butils.subtract_parcelmean(i,parc_matrix) for i in ims_adjcorr]
                    all_neighbour_distances_mean = [butils.subtract_parcelmean(i,parc_matrix) for i in all_neighbour_distances_mean]       
                    sulcs = [butils.subtract_parcelmean(i,parc_matrix) for i in sulcs]         
                for i in [0,1]:
                    p.plot(ims_adjcorr[i],cmap='inferno')
                    p.plot(all_neighbour_distances_mean[i],cmap='inferno')
                    p.plot(sulcs[i])
                    fig,axs=plt.subplots(2,2,figsize=(7,7))
                    ax=axs[0,0]
                    ax.scatter(all_neighbour_distances_mean[i],ims_adjcorr[i],1,alpha=0.05,color='k')
                    ax.set_xlabel('Inter-vertex distance (mm)')
                    ax.set_ylabel('Correlation with neighbours')
                    corr,pval = butils.corr_with_nulls(all_neighbour_distances_mean[i],ims_adjcorr[i],mask,null_method,n_perm)
                    ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
                    #ax.set_title(f'Correlation: {np.corrcoef(all_neighbour_distances_mean[i],ims_adjcorr[i])[0,1]:.3f}')
                    ax=axs[0,1]
                    ax.scatter(sulcs[i],all_neighbour_distances_mean[i],1,alpha=0.05,color='k')
                    ax.set_xlabel('Sulcal depth')
                    ax.set_ylabel('Inter-vertex distance (mm)')
                    corr,pval = butils.corr_with_nulls(sulcs[i],all_neighbour_distances_mean[i],mask,null_method,n_perm)
                    ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
                    #ax.set_title(f'Correlation: {np.corrcoef(sulcs[i],all_neighbour_distances_mean[i])[0,1]:.3f}')
                    ax=axs[1,0]
                    ax.scatter(sulcs[i],ims_adjcorr[i],1,alpha=0.05,color='k')
                    ax.set_xlabel('Sulcal depth')
                    ax.set_ylabel('Correlation with neighbours')
                    corr,pval = butils.corr_with_nulls(sulcs[i],ims_adjcorr[i],mask,null_method,n_perm)
                    ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
                    #ax.set_title(f'Correlation: {np.corrcoef(sulcs[i],ims_adjcorr[i])[0,1]:.3f}')
                    ax=axs[1,1]
                    ax.scatter(sulcs[i],ims_adjcorr[(i+1)%nsubjects],1,alpha=0.05,color='k')
                    ax.set_xlabel('Sulcal depth')
                    ax.set_ylabel('Correlation with neighbours')
                    corr,pval = butils.corr_with_nulls(sulcs[i],ims_adjcorr[(i+2)%2],mask,null_method,n_perm)
                    ax.set_title(f'Correlation of this sub\'s sulc with \nlocalcorrs of {subjects[(i+1)%nsubjects]}: R={corr:.3f}, p={pval:.3f}')
                    #ax.set_title(f'Correlation of this sub\'s sulc with \nlocalcorrs of {subjects[(i+1)%2]}: {np.corrcoef(sulcs[i],ims_adjcorr[(i+1)%2])[0,1]:.3f}')
                    fig.suptitle(f'Subject {subjects[i]}')
                    fig.tight_layout()
                    plt.show(block=False)

            assert(0) #because we've modified the data

        ### Is there a bias in the group mean local neighbourhood correlations?
        do_groupmean = False
        if do_groupmean:
            neighbour_distances_submean = np.mean(all_neighbour_distances_mean,axis=0) #mean of all subjects' local neighbourhood distances
            ims_adjcorr_mean=np.mean(ims_adjcorr,axis=0) #mean of all subjects' local neighbourhood correlations
            p.plot(butils.fillnongray(ims_adjcorr_mean,mask))
            p.plot(butils.fillnongray(neighbour_distances_submean,mask))

        #Correlation between sulc and local correlations. Looking at parcel-mean data, or single-parcel data
        do_correlations=False
        if do_correlations:
            corrs_sulc_adjcorr_parc = [np.corrcoef(sulc,adjcorr)[0,1] for sulc,adjcorr in zip(sulcs_parc,ims_adjcorr_parc)]
            corrs_sulc_adjcorr_singleparc = [np.corrcoef(sulc,adjcorr)[0,1] for sulc,adjcorr in zip(sulcs_singleparc,ims_adjcorr_singleparc)]

            print('Correlation (within-subject, across parcels) of (parcel-averaged) sulcal depth vs local correlations')
            print(corrs_sulc_adjcorr_parc)

            print('Correlation (within-subject, within single parcel, across vertices) of sulcal depth vs local correlations')
            print(corrs_sulc_adjcorr_singleparc)

    ### Analyse reliability and identifiability of functional connectivity
    do_FC = False
    if do_FC:
        print(f'{c.time()}: Calculate FC')   

        ntimepoints_per_half = ims[0].shape[0]//2 #number of time points in the test and retest halves of each subject's data

        #split each subject's data into test and retest halves
        ims_parc = [array[0:ntimepoints_per_half,:] for array in ims_parc] + [array[ntimepoints_per_half:,:] for array in ims_parc] 
        ims_singleparc = [array[0:ntimepoints_per_half,:] for array in ims_singleparc] + [array[ntimepoints_per_half:,:] for array in ims_singleparc]

        ims_pfc = Parallel(n_jobs=-1,prefer='processes')(delayed(np.corrcoef)(im.T) for im in ims_parc) #list (nsubjects) of parcellated FC Matrices (nparcels,nparcels)
        ims_sfc = Parallel(n_jobs=-1,prefer='processes')(delayed(np.corrcoef)(im.T) for im in ims_singleparc) #list (nsubjects) of within-single-parcel FC Matrices (nvertices,nvertices)

        ims_pfcv = [i.ravel() for i in ims_pfc] #list (nsubjects) of parcellated FC matrices vectorized, shape (nparcls*nparcels,)
        ims_sfcv = [i.ravel() for i in ims_sfc] #list (nsubjects) of single parcel FC matrices vectorized, shape (nverticesInParcel*nverticesInParcel,)

        #re-split into run 1 and run 2
        ims_pfcvx = ims_pfcv[0:nsubjects]
        ims_pfcvy = ims_pfcv[nsubjects:]
        ims_sfcvx = ims_sfcv[0:nsubjects]
        ims_sfcvy = ims_sfcv[nsubjects:]

        print(f'{c.time()}: Get identifiability')  
        #Correlation between test and retest scans, across all subject pairs
        import tkalign_utils as tutils
        figsize = (4,4)
        x_axis_label = 'Test scans'
        y_axis_label = 'Retest scans'
        for reg in [False,True]:
            prefix = f'Parcellated FC (reg={reg})'
            corrs = tutils.ident_plot(ims_pfcvx,x_axis_label,ims_pfcvy,y_axis_label,reg=reg,normed=False,figsize=figsize,title=prefix) 
            print(f'{prefix}, test-retest identifiability is {tutils.identifiability(corrs):.2f}%')
            prefix = f'Single parcel FC (reg={reg})'
            corrs = tutils.ident_plot(ims_sfcvx,x_axis_label,ims_sfcvy,y_axis_label,reg=reg,normed=False,figsize=figsize,title=prefix)
            print(f'{prefix}, test-retest identifiability is {tutils.identifiability(corrs):.2f}%')
        print(f'{c.time()}: Finished with identifiability')  
        plt.show(block=False)