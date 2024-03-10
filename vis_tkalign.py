"""
Script to visualize how individual differences in high-res structural connectivity links to individual differences in functional topography
"""

import numpy as np
import hcpalign_utils as hutils
import tkalign_utils as tutils
from hcpalign_utils import ospath
import warnings
import matplotlib.pyplot as plt, tkalign_utils as tutils
from tkalign_utils import values2ranks as ranks, regress as reg, identifiability as ident, count_negs
import hcp_utils as hcp
from nilearn import plotting

### PARAMETERS
load_file_path = 'Amovf0123t0_S300_Tmovf0123t0sub20to30_RG0ffrr_TempScal_gam0.3_B100la20-30_Dtracks_5M_1M_end_3mm_0mm_RDRT_S10_30-80'
aligner_negatives = 'abs'

### INITIAL SETUP
c=hutils.clock()
figures_subfolder=ospath(f'{hutils.results_path}/figures/vis_tkalign')
p=hutils.surfplot(figures_subfolder,plot_type='open_in_browser')

### GET MORE PARAMETERS FROM LOAD_FILE_PATH
nblocks,block_choice,howtoalign,pre_hrc_fwhm,post_hrc_fwhm,alignfiles,tckfile,subs_test_range_start, subs_test_range_end = tutils.extract_tkalign_corrs2(load_file_path)
alignfile = alignfiles[0]
parcellation_string = tutils.extract_alignpickles3(alignfile,'parcellation_string')
MSMAll = tutils.extract_alignpickles3(alignfile,'MSMAll')
sift2=not('sift' in tckfile) #True, unless there is 'sift' in tckfile


### SETUP PARCELLATION, SORTER, AND SMOOTHERS
align_labels=hutils.parcellation_string_to_parcellation(parcellation_string)
align_parc_matrix=hutils.parcellation_string_to_parcmatrix(parcellation_string)
nparcs=align_parc_matrix.shape[0]
subs_inds = {}
subs_inds['test'] = np.arange(subs_test_range_start,subs_test_range_end)
subject_ids = hutils.all_subs[subs_test_range_start:subs_test_range_end]
### Set up smoothing kernels
sorter,unsorter,slices=tutils.makesorter(align_labels) # Sort vertices by parcel membership, for speed
smoother_pre = tutils.get_smoother(pre_hrc_fwhm)[sorter[:,None],sorter]
smoother_post = tutils.get_smoother(post_hrc_fwhm)[sorter[:,None],sorter]


### LOAD ALIGNERS
print(f'{c.time()}: Load aligners start, with negatives to {aligner_negatives}')
fa, _ = tutils.new_func(c,subs_inds, alignfile, False, False, aligner_negatives, sorter, slices, smoother_post, ['test'])
print(f'{c.time()}: Load aligners end')
fat=fa['test']

save_folder=f'{hutils.intermediates_path}/tkalign_corrs2' #save results data in this folder
save_path=ospath(f"{save_folder}/{load_file_path}.npy")
blocks,a=tutils.load_a(save_path)

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
    carn=[tutils.count_negs(arn[i,:,:]) for i in range(arn.shape[0])] #no of negative values for subject's connectome (across blocks and subjects for fa)
    anc=[tutils.count_negs(an[:,:,i]) for i in range(arn.shape[2])] 
    can=[tutils.count_negs(an[i,:,:]) for i in range(arn.shape[0])] 

    mao=np.nanmean(ao,axis=-1) #mean along blocks
    maro=np.nanmean(aro,axis=-1)
    mar=np.nanmean(ar,axis=-1)
    
    a2=a.copy()
    for i in range(a2.shape[-1]): np.fill_diagonal(a2[:,:,i], np.nan)

    n_elements = 50
    ae = tutils.get_extremal_indices(a,n_elements,kind='largest')
    a2e = tutils.get_extremal_indices(a2,n_elements,kind='largest')
    are = tutils.get_extremal_indices(ar,n_elements,kind='largest')
    arne = tutils.get_extremal_indices(arn,n_elements,kind='smallest')
    ane = tutils.get_extremal_indices(an,n_elements,kind='smallest')

    aR = reg(a,include_axis_2=True)
    aRn = tutils.subtract_nonscrambled_from_a(aR)
    aRe = tutils.get_extremal_indices(aR,n_elements,kind='largest')
    aRne = tutils.get_extremal_indices(aRn,n_elements,kind='smallest')


### GET THE SUBJECTS AND PARCELS OF INTEREST
nblock=12
subject_indices = [16,0] #these are indices in test participants. First one is whose connectome was used. Second one iw whose functional aligner really underperforms when applied to subx's connectome
both_subject_ids = [subject_ids[i] for i in subject_indices]

nsubjects = 2
nparcels = 2
parcs = [blocks[i,nblock] for i in range(2)] #indices of the two parcels involved
parcs_bool = [align_labels==index for index in parcs] #boolean arrays for the two parcels 
fas = [[fat[subject].fit_[index].R for index in parcs] for subject in subject_indices] #list (nsubjects) of list (parcels) of functional aligners, e.g. fas[0][1] is subject 0's functional aligner for parcel 1
[[np.fill_diagonal(array,0) for array in arrays] for arrays in fas]



### LOAD CONNECTOMES
hr = hutils.get_highres_connectomes(c,both_subject_ids,tckfile,MSMAll=MSMAll,sift2=sift2,prefer='threads',n_jobs=-1) 
hr = [array[sorter[:,None],sorter] for array in hr]
hr= hutils.smooth_highres_connectomes(hr,smoother_pre)
hrs = [hr[subject][slices[parcs[0]],slices[parcs[1]]].toarray() for subject in range(nsubjects)] #list (nsubjects). Each element is the structural connectivity linking parcel 0 to parcel 1

### VISUALISE

gray=hutils.vertexmap_59kto64k()
#coords = hcp.mesh.midthickness[0][gray]
coords_flat = hcp.mesh.flat[0][gray]
parcs_coords = [coords_flat[parcs_bool[i]] for i in range(2)]

smoother = tutils.get_smoother(3)[sorter[:,None],sorter]
parcs_smoothers = [smoother[slices[parcs[i]],slices[parcs[i]]] for i in range(2)]


fas_smoothed = [[parcs_smoothers[i] @ fas[subject][i] for i in range(nparcels)] for subject in range(nsubjects)]
vmax = np.max([np.max(fas_smoothed[sub][nparc]) for sub in range (nsubjects) for nparc in range(2)]) #common colormap for subplots
vmin = np.min([np.min(fas_smoothed[sub][nparc]) for sub in range (nsubjects) for nparc in range(2)])


subject = 0
nparc = 0
template_vertices = [0,50,100,150]
vertices_source_subject = [[[fas_smoothed[subject][nparc][:,vert] for vert in template_vertices] for nparc in range(nparcels)] for subject in range(nsubjects)]  #nested list (subject, parcel, template_vertex). For a given vertex in the template, give the spatial map of which vertices in the source subject it is mapped to

 give the mapping onto vertices in the source subject.
vertices_source_subject_proj = vertices_source_subject #initialise with same size. Will be above spatial maps multiplied by the same subject's hrc

for subject in range(nsubjects):
    for nparc in range(nparcels):
        for index,vertex in enumerate(template_vertices):
            if nparc==0: hrc = hrs[subject]
            elif nparc==1: hrc = hrs[subject].T
            vertices_source_subject_proj[subject][nparc][index] = vertices_source_subject[subject][nparc][index] @ hrc


#VISUALISE HIGH_RES STREAMLINES
fig,axs=plt.subplots(nsubjects,nparcels)
for nparcel in range(nparcels):
    for nsubject in range(nsubjects):
        ax = axs[nsubject,nparcel]
        x = parcs_coords[nparcel][:,1]
        y = parcs_coords[nparcel][:,2]
        z = hrs[nsubject].sum(axis=1-nparcel)
        ax.plot(x,y,'o',markersize=2,color='grey')
        im = ax.tripcolor(x,y,z,shading='gouraud',cmap='viridis',vmin=None,vmax=None)
        #cax=fig.add_axes([.25,.05,.5,.05])
        #fig.colorbar(im,cax=cax,orientation='horizontal')
        ax.set_title(f'parc {nparcel}, sub {nsubject}')
    fig.colorbar(im)
    fig.suptitle(f'Connectivity: total no of streamlines per vertex')
plt.show(block=False)


### VISUALISE FUNCTIONAL ALIGNER FOR A SPECIFIC PARCEL IN A SINGLE SUBJECT
for nparcel in range(nparcels):
    fig,axs=plt.subplots(nsubjects,len(template_vertices))
    for index,vertex in enumerate(template_vertices):
        for nsubject in range(nsubjects):
            ax = axs[nsubject,index]
            x = parcs_coords[nparcel][:,1]
            y = parcs_coords[nparcel][:,2]
            z = fas_smoothed[nsubject][nparcel][:,vertex]
            ax.plot(x,y,'o',markersize=2,color='grey')
            ax.plot(x[vertex],y[vertex],'o',markersize=5,color='red')
            im = ax.tripcolor(x,y,z,shading='gouraud',cmap='viridis',vmin=None,vmax=None)
            #cax=fig.add_axes([.25,.05,.5,.05])
            #fig.colorbar(im,cax=cax,orientation='horizontal')
            ax.set_title(f'vert {vertex} sub {nsubject}')
    fig.suptitle(f'Func aligners: parcel {nparcel}\nRed dots are template verts')
    fig.colorbar(im)
plt.show(block=False)

### VISUALISE FUNC ALIGNER SUM FROM A SUBJECT PROJECTED VIA THEIR OWN STREAMLINES TO OTHER PARCEL
for nparcel in range(nparcels):
    fig,axs=plt.subplots(nsubjects,len(template_vertices))
    for index,vertex in enumerate(template_vertices):
        for nsubject in range(nsubjects):
            ax = axs[nsubject,index]
            x = parcs_coords[1-nparcel][:,1]
            y = parcs_coords[1-nparcel][:,2]
            z = vertices_source_subject_proj[nsubject][nparcel][index]
            ax.plot(x,y,'o',markersize=2,color='grey')
            ax.plot(x[vertex],y[vertex],'o',markersize=5,color='red')
            im = ax.tripcolor(x,y,z,shading='gouraud',cmap='viridis',vmin=None,vmax=None)
            #cax=fig.add_axes([.25,.05,.5,.05])
            #fig.colorbar(im,cax=cax,orientation='horizontal')
            ax.set_title(f'vert {vertex} sub {nsubject}')
    fig.suptitle(f'Func aligners: parcel {nparcel} self-proj to parcel {1-nparcel}\nRed dots are template verts.')
    fig.colorbar(im)
plt.show(block=False)



assert(0)




vmax=0.04
image_num=0
for source_vertex in [0]:
    for nparcel in range(2):
        for nsubject in range(2):
            image_num+=1
            title = f'P{nparcel}_V{source_vertex}_S{nsubject}'
            print(title)
            x=np.zeros(59412,dtype=np.float32)
            x[parcs_bool[nparcel]]=fas[nsubject][nparcel][:,source_vertex]
            new_data = hcp.cortex_data(x)
            output_file = ospath(f'{figures_subfolder}/{title}')
            view=plotting.plot_surf(hcp.mesh.inflated,new_data,output_file=output_file,symmetric_cmap=True,cmap='gist_ncar',view=(0,0),bg_map=hcp.mesh.sulc, threshold=1e-8,vmin=-vmax,vmax=vmax)  
#plotting.show()

