#Test eigenstrapping

"""
from eigenstrapping.datasets import load_surface_examples
surf_lh, surf_rh, data_lh, data_rh, emodes_lh, emodes_rh, evals_lh, evals_rh = load_surface_examples(with_surface=True)
#Issue 1: FileNotFoundError: No such file or no access: 'C:/Users/Jayson/miniconda3/envs/eigen/lib/site-packages/eigenstrapping/datasets/brainmaps/fsaverage5_lh.gii'
#No 'brainmaps' folder

from eigenstrapping import datasets
z = datasets.load_distmat('fsaverage',den='10k',hemi='lh',data_dir = data_dir) 
#Question: What other meshes are available?

#Issue 2: "import eigenstrapping" does something rather weird to my interactive python console. After doing that, printing anything to the python console doesnt work anymore. Some other actions on "strings" dont work anymore either. Is this a windows problem.
"""

"""
import hcpalign_utils as hutils
from hcpalign_utils import ospath
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
biasfmri_intermediates_path = ospath(f'{project_path}/intermediates')
data_dir = ospath(f'{biasfmri_intermediates_path}/temp')

sub='100610'
hemi="L"
#mesh_path = surf_file=ospath(f"{hutils.hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.{hemi}.{'midthickness'}.32k_fs_LR.surf.gii")
mesh_path_left="C:\\Users\\Jayson\\neuromaps-data\\atlases\\fsLR\\tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii"
mesh_path_right="C:\\Users\\Jayson\\neuromaps-data\\atlases\\fsLR\\tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii"

import numpy as np
data = np.random.random((59412))

import biasfmri_utils as butils
nulls=butils.eigenstrap_bilateral(mesh_path_left,mesh_path_right,data,num_modes=1000,num_nulls=10)
p=hutils.surfplot('',plot_type='open_in_browser')
p.plot(data)
p.plot(nulls[0,:])
assert(0)

from eigenstrapping import stats
corr,pval = stats.compare_maps(data,data2,nulls=nulls)
print(f'r = {corr:.3f}, p = {pval:.3f}')

assert(0)
"""

#######################
#Visualize a small part of a mesh

import trimesh
import hcpalign_utils as hutils
from hcpalign_utils import ospath
import hcp_utils as hcp
import numpy as np
#import getmesh_utils
import biasfmri_utils as butils
import generic_utils as gutils
import brainmesh_utils as bmutils

#Code to run windows command line commands
"""
import subprocess
import os
import sys

hcp_folder=hutils.hcp_folder
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
biasfmri_intermediates_path = ospath(f'{project_path}/intermediates')
temp_path = ospath(f'{biasfmri_intermediates_path}/temp')

sub='100610'
hemi="L"
pt_hemi_32k_mesh_path = surf_file=ospath(f"{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.{hemi}.{'sphere'}.32k_fs_LR.surf.gii")

onavg_folder = f"{project_path}\\intermediates\\tpl-onavg-main"
onavg_hemi_10k_file = f"tpl-onavg_hemi-{hemi}_den-10k_sphere.surf.gii"
onavg_hemi_10k_path = f"{onavg_folder}\\{onavg_hemi_10k_file}"

command = f"wb_command -surface-affine-regression {pt_hemi_32k_mesh_path} {onavg_hemi_10k_path} {f'{temp_path}/affine'}"
print(command)
os.system(command)
assert(0)
"""

"""
      <source> - the surface to warp
      <target> - the surface to match the coordinates of
      <affine-out> - output - the output affine file
"""

#Trimesh stuff. Dont need anymore..
c=gutils.clock()
which_subject='100610'
surface_type = 'white'
MSMAll=False
hemi="L"

#vertices,faces = getmesh_utils.get_verts_and_triangles(which_subject,surface_type,MSMAll)
#vertices_64k,faces_64k = getmesh_utils.get_verts_and_triangles_hemi(which_subject,hemi,surface_type,MSMAll=MSMAll)
vertices,faces = bmutils.hcp_get_mesh_hemi(which_subject,hemi,surface_type,MSMAll=MSMAll)
mask = hutils.get_fsLR32k_mask(hemi=hemi)

#Only include vertices within these coordinates
print(np.min(vertices,axis=0))
print(np.max(vertices,axis=0))
coordinate_range = ((-70,-20),(-50,20),(-np.inf,np.inf))
for dim in range(3):
    mask = mask & ((vertices[:,dim]>coordinate_range[dim][0]) & (vertices[:,dim]<coordinate_range[dim][1]))

vertices = vertices[mask,:]
#faces = hutils.cortex_64kto59k_for_triangles(faces_64k,hemi=hemi)
faces = bmutils.triangles_removenongray(faces,mask)

#mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],faces=[[0, 1, 2]])
mesh = trimesh.Trimesh(vertices=vertices,faces=faces)
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
biasfmri_intermediates_path = ospath(f'{project_path}/intermediates')

neighbour_vertices, neighbour_distances, neighbour_distances_mean=butils.get_subjects_neighbour_vertices(c, which_subject,'midthickness',None, biasfmri_intermediates_path, 'local', None, True, False,MSMAll=MSMAll)
color_values = neighbour_distances_mean[0:len(vertices)]

sulc = butils.hcp_get_sulc(which_subject,version='fsaverage_LR32k')
#color_values = sulc[hutils.get_fsLR32k_mask(hemi='both')][0:len(vertices)]

colors = trimesh.visual.color.interpolate(color_values, color_map='viridis')

points=trimesh.points.PointCloud(vertices)
#points.colors=[100,100,100,150]
points.colors=colors
points.show()
assert(0)

values = vertices[:,1]
vertices_mask = values>np.quantile(values,0.96)
mesh.update_vertices(vertices_mask)
color_values = color_values[vertices_mask]
colors = trimesh.visual.color.interpolate(color_values, color_map='viridis')
assert(0)
mesh.visual.vertex_colors = colors
mesh.show()
assert(0)


"""


print(f"{c.time()}: {len(mesh.vertices)} vertices")
meshes = [trimesh.creation.uv_sphere(radius=0.3,count=(3,1)) for i in range(len(mesh.vertices))]
color = trimesh.visual.random_color() 
for i, m in enumerate(meshes):
    m.apply_translation(mesh.vertices[i,:])
    m.visual.vertex_colors = colors[i]
"""

#colors=trimesh.visual.color.interpolate(sulc,color_map='viridis')
#colors = trimesh.visual.color.interpolate(vertices[:,0], color_map='prism')
#np.random.shuffle(colors)
#mesh.visual.vertex_colors=colors

#slice = mesh.section(plane_origin=mesh.centroid,plane_normal=[0,0,1]) #Path3D
#slice_2D, to_3D = slice.to_planar()
#slice_2D.show()
#slice.show()

print(f'{c.time()}: Done loading mesh')
trimesh.Scene([mesh]+meshes).show()

#mesh.show() 
assert(0)

#######################

import numpy as np
import os
from sklearn.svm import LinearSVC
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from hcpalign_utils import ospath
from joblib import Parallel, delayed
#from my_surf_pairwise_alignment import MySurfacePairwiseAlignment, LowDimSurfacePairwiseAlignment









#Plot ISC spatial maps from file
filenames=['20240428_212410','20240428_212357','20240428_223912','20240428_222854','20240428_223932','20240428_222934']
titles=['sulc','all','sulc_func','sulc_comb','all_func','all_comb']
for filename in filenames:
    plot_dir=ospath(f'{hutils.results_path}/figures/hcpalign/{filename}')
    values = np.load(ospath(f'{plot_dir}\\ISCs_vertexmeans.npy'))
    p=hutils.surfplot(plot_dir,plot_type='open_in_browser')
    values[values<0] = 0
    p.plot(values,'ISCs_vertexmeans',vmax=1)
assert(0)

#Get individualized parcellations
"""
nparcs = 300
subjects = hutils.all_subs[0:3]
labels,colors=hutils.get_individualized_parcellation(nparcs,subjects)
p=hutils.surfplot('')
p.plot(labels[0,:],cmap='tab20')
"""

### Get parcellated connectomes for Anna Behler ###
'''
import tkalign_utils as tutils
import hcpalign_utils as hutils
from hcpalign_utils import ospath
c=hutils.clock()

#Settable parameters
subjects = hutils.all_subs[0:2]
MSMAll=False
sift2=True
tckfile = tutils.get_tck_file()
par_prefer_hrc='threads'  #'threads' (default) or 'processes' for getting high-res connectomes from file
sift2=not('sift' in tckfile) #True, unless there is 'sift' in tckfile

parcellated = False
if parcellated:
    parcellation_string = 'M'
    #Get connectome
    align_parc_matrix=hutils.parcellation_string_to_parcmatrix(parcellation_string)
    connectomes = tutils.get_parcellated_connectomes(c,tckfile, MSMAll, sift2, align_parc_matrix, subjects, par_prefer_hrc)  
    #Save file
    for subject,connectome in zip(subjects,connectomes):
        file_name = f'{tckfile[:-4]}_{parcellation_string}_{subject}.npy'
        save_path = ospath(f'{hutils.intermediates_path}/parcellated_connectomes/{file_name}')
        np.save(save_path,connectome.toarray())
else:
    connectomes = hutils.get_highres_connectomes(c,subjects,tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefer_hrc,n_jobs=-1)
    from scipy import sparse
    for fwhm in [3]:
        for subject,connectome in zip(subjects,connectomes):
            if fwhm==0: 
                file_name = f'{tckfile[:-4]}_{subject}.npz'
            else:
                skernel=sparse.load_npz(ospath(f'{hutils.intermediates_path}/smoothers/{subject}_{fwhm}_0.01.npz'))
                from Connectome_Spatial_Smoothing import CSS as css   
                connectome = css.smooth_high_resolution_connectome(connectome,skernel)
                file_name = f'{tckfile[:-4]}_{subject}_smooth_{fwhm}mm.npz'

            save_path = ospath(f'{hutils.intermediates_path}/highres_connectomes/{file_name}')
            sparse.save_npz(save_path,connectome)
            #Load with x=sparse.load_npz(save_path)
'''



"""
from make_gdistances_full import get_searchlights
parcels = get_searchlights(1)
hr=hutils.get_highres_connectomes(None,['100610'],'tracks_5M_50k.tck')[0]

parcel = parcels[0]
hrp = hr[:,parcel][parcel,:]
"""


#Test GroupedFeaturesEstimator
"""
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=5000, noise=0.0)

feature_groups = np.array([0]*2500 + [1]*2500)

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR

base_estimator = Ridge(alpha=1000)
base_estimator = SVR(kernel='linear')
final_estimator = SVR(kernel='linear')

n_jobs_outer = -1
n_jobs_inner = -1


from predict_utils import GroupedFeaturesEstimator
est = GroupedFeaturesEstimator(cv=2,n_jobs=n_jobs_inner,base_estimator=base_estimator,final_estimator=final_estimator,feature_groups=feature_groups)

from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)
r2_scores = cross_val_score(est, X, y, cv=kf, scoring='r2',n_jobs=n_jobs_outer)
print(r2_scores)
assert(0)
"""


p=hutils.surfplot('',plot_type='open_in_browser')
c=hutils.clock()
hcp_folder=hutils.hcp_folder
intermediates_path=hutils.intermediates_path
results_path=hutils.results_path

subs=hutils.all_subs[slice(0,2)]
subs_template=hutils.all_subs[slice(0,5)]
nsubs = np.arange(len(subs)) #number of subjects

post_decode_smooth=hutils.make_smoother_100610(0)

runs = [0]
imgs_align,align_string = hutils.get_movie_or_rest_data(subs,'movie',runs=runs,fwhm=0,clean=True,MSMAll=False)
imgs_decode,decode_string = hutils.get_task_data(subs,hutils.tasks[0:7],MSMAll=False)
labels = [np.array(range(i.shape[0])) for i in imgs_decode] 

parcellation_string='S300'
clustering = hutils.parcellation_string_to_parcellation(parcellation_string)
nparcs = clustering.max()+1

alignment_kwargs = {}
per_parcel_kwargs = {}

'''

#from make_gdistances_full import get_searchlights
#clustering = get_searchlights(5)

classifier=LinearSVC(max_iter=10000,dual='auto')     
print(f'{c.time()}: Done loading data')

#Example: Template alignment

from fmralign.template_alignment import TemplateAlignment
imgs_template,template_align_string = hutils.get_movie_or_rest_data(subs_template,'movie',runs=runs,fwhm=0,clean=True,MSMAll=False)
aligners = TemplateAlignment('scaled_orthogonal',clustering=clustering,alignment_kwargs={'scaling':True})
#args_template_dict = {'hyperalignment':{'n_iter':1,'do_level_1':True, 'normalize_imgs':'zscore', 'normalize_template':'zscore', 'remove_self':True, 'level1_equal_weight':False},'GPA': {'n_iter':1,'do_level_1':False,'normalize_imgs':'rescale','normalize_template':'rescale','remove_self':False,'level1_equal_weight':False}}
args_template = {'n_iter':0,'do_level_1':False,'normalize_imgs':'rescale','normalize_template':'rescale','remove_self':False,'level1_equal_weight':False}
aligners.make_template(imgs_template,**args_template)
#aligners.make_template(imgs_template,n_iter=1,do_level_1=False,level1_equal_weight=False,normalize_imgs='zscore',normalize_template='zscore',remove_self=False,gamma=0)
#aligners.template = np.mean(imgs_template,axis=0)
#aligners.make_lowdim_template(imgs_template,clustering,n_bags=1)
print(f'{c.time()}: Start fitting')
#aligners.fit_to_template(imgs_align)

aligners.fit_template_to_imgs(imgs_align)
assert(0)

im=imgs_align[0]
t = aligners.template
t2=aligners.estimators[0].transform(t)
print(np.corrcoef(im.ravel(),t2.ravel()))

assert(0)



imgs_decode_aligned=[aligners.transform(imgs_decode[i],i) for i in range(len(imgs_decode))]

def ploto(img,vmax=None):
    p.plot(img,vmax=vmax)

ploto(imgs_align[0][100,:],vmax=None) #movie
ploto(imgs_align[1][100,:],vmax=None) #movie
ploto(imgs_align[2][100,:],vmax=None) #movie
ploto(aligners.template[100,:],vmax=None) #template movie

ploto(imgs_decode[0][8+3]) #RH
ploto(imgs_decode[1][8+3]) 
ploto(imgs_decode[2][8+3])
ploto(imgs_decode[0][12]) #Emotion Faces
ploto(imgs_decode[1][12]) 
ploto(imgs_decode[2][12]) 
ploto(imgs_decode[0][13]) #Emotion Shapes
ploto(imgs_decode[1][13]) 
ploto(imgs_decode[2][13]) 

ploto(imgs_decode_aligned[0][8+3]) #RH
ploto(imgs_decode_aligned[1][8+3]) 
ploto(imgs_decode_aligned[2][8+3]) 
ploto(imgs_decode_aligned[0][12]) #Emotion faces
ploto(imgs_decode_aligned[1][12]) 
ploto(imgs_decode_aligned[2][12]) 
ploto(imgs_decode_aligned[0][13]) #Emotion faces
ploto(imgs_decode_aligned[1][13]) 
ploto(imgs_decode_aligned[2][13]) 

assert(0)

print(aligners.estimators[0].fit_[0].R[0:2,0:2])
ratio_within_roi = hutils.do_plot_impulse_responses(p,'',aligners.estimators[0])
print(f'Ratio within ROI: {ratio_within_roi:.2f}')


'''

#Preparation for ProMises model
nparcs=parcellation_string[1:]
gdists_path=hutils.ospath(f'{hutils.intermediates_path}/geodesic_distances/gdist_full_100610.midthickness.32k_fs_LR.S{nparcs}.p') #Get saved geodesic distances between vertices (for vertices in each parcel separately)
import pickle
with open(gdists_path,'rb') as file:
    gdists = pickle.load(file)
promises_k=0.3 #k parameter in ProMises model
promises_F = [np.exp(-i) for i in gdists] #local distance matrix in ProMises model
alignment_kwargs = {'promises_k':promises_k}
per_parcel_kwargs = {'promises_F':promises_F}


#Procrustes alignment 
from fmralign.surf_pairwise_alignment import SurfacePairwiseAlignment
aligner = SurfacePairwiseAlignment(alignment_method='scaled_orthogonal',clustering=clustering,alignment_kwargs=alignment_kwargs, per_parcel_kwargs=per_parcel_kwargs) 
aligner.fit(imgs_align[0],imgs_align[1]) 

assert(0)

#FOR METHODS FIGURE
ims=imgs_decode[0]
imst=aligner.transform(ims)
p.plot(ims[8+3]) #right hand
p.plot(imst[8+3])
hutils.do_plot_impulse_responses(p,'',aligner,radius=1,vertices=None)
assert(0)

print(f'Corr bw images: {np.corrcoef(ims[0],ims[1]):.3f}')
print(f'Corr bw transformed images: {np.corrcoef(imst[0],imst[1]):.3f}')

imsp = [ims[:,clustering==i] for i in np.unique(clustering)]
imstp = [imst[:,clustering==i] for i in np.unique(clustering)]


imspc = [np.corrcoef(imsp[i][0,:],imsp[i][10,:])[0,1] for i in range(len(np.unique(clustering)))]
imstpc = [np.corrcoef(imstp[i][0,:],imstp[i][10,:])[0,1] for i in range(len(np.unique(clustering)))]
print(f'Within-parcel corr bw images: {np.mean(imspc):.3f}')
print(f'Within-parcel corr bw transformed images: {np.mean(imstpc):.3f}')

scmap=hutils.aligner_get_scale_map(aligner)
p.plot(scmap)





print(aligner.fit_[0].R.sum(axis=1)[0:3])
print(aligner.fit_[0].R.sum(axis=0)[0:3])


print(aligner.fit_[0].R[0:3,0:3])

ratio_within_roi = hutils.do_plot_impulse_responses(p,'',aligner)
print(f'Ratio within ROI: {ratio_within_roi:.2f}')

assert(0)

print(f'{c.time()}: Done fitting alignment')

#Old methods
"""
from my_surf_pairwise_alignment import MySurfacePairwiseAlignment
aligner=MySurfacePairwiseAlignment(alignment_method='scaled_orthogonal', clustering=clustering,n_jobs=-1,reg=0)  #faster if fmralignbench/surf_pairwise_alignment.py/fit_parcellation uses processes not threads
aligner.fit(imgs_align[0],imgs_align[1])
aligner.transform(imgs_decode[0])

from my_template_alignment import MyTemplateAlignment, get_template  
aligners= MyTemplateAlignment('scaled_orthogonal',clustering=clustering,n_jobs=1,n_iter=2,scale_template=False,template_method=1,reg=0)
aligners.fit(imgs_align) 
imgs_decode_new = aligners.transform(imgs_decode[0],0)
print(aligners.estimators[0].fit_[0].R[0:2,0:2])
assert(0)
"""


"""
#Generic brain plot for figure
import hcpalign_utils as hutils
import numpy as np
p=hutils.surfplot('')
x=hutils.makesurfmap([])

for i in np.arange(0,1,0.2):
    x[:]=i
    p.plot(x,vmin=0,vmax=1,cmap='Greys')
"""

"""
n=[5,10,20,50,100]
anat=[.68, .84, .88, .91, .93]
temp=[.81,.91,.87,.88,.89]
temp_niter1 = [.79, .91, .94, .93, .95]

temp_restFC=[0.88,0.89,0.92] #don't have 0 and 100 subs values yet

import matplotlib.pyplot as plt
import matplotlib

fig,ax=plt.subplots(1)
ax.plot(n[1:],anat[1:],'k-o',markersize=8)
ax.plot(n[1:],temp_niter1[1:],'r-o',markersize=8)
#ax.plot(n[1:-1],temp_restFC,'b-o',markersize=8)
ax.set_xlabel('Number of subjects')
ax.set_ylabel('Classification accuracy')
#ax.legend(['Standard co-registration', 'Functional alignment - movie', 'Functional alignment - rsfMRI'])
ax.legend(['Standard co-registration', 'Functional alignment'])
ax.set_xscale('log')
ax.set_xticks(n[1:])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()
"""

'''
import hcp_utils as hcp
import matplotlib.pyplot as plt
import numpy as np
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


import socket
hostname=socket.gethostname()
if hostname=='DESKTOP-EGSQF3A': #home pc
    tckfile= 'tracks_5M_sift1M_200k.tck' #'tracks_5M_sift1M_200k.tck','tracks_5M.tck' 
else: #service workbench
    tckfile='tracks_5M_1M_end.tck'

sift2=not('sift' in tckfile) #default True
MSMAll=False

pre_hrc_fwhm = 0
align_nparcs=300
align_labels=hutils.Schaefer(align_nparcs)
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)

sub = hutils.subs[0]

print(f'{c.time()}: Get highres connectomes and downsample',end=", ")
par_prefer_hrc='threads'        
hrs = hutils.get_highres_connectomes(c,[sub],tckfile,MSMAll=MSMAll,sift2=sift2,prefer=par_prefer_hrc,n_jobs=-1)
hr=hrs[0]

print(f'{c.time()}: Get geodesic distances')
from make_gdistances_full import get_saved_gdistances_full
gdists = get_saved_gdistances_full(sub,'midthickness')
gdistsL = gdists[hcp.struct.cortex_left,hcp.struct.cortex_left].astype(np.float32)
print(f'{c.time()}: Get geodesic distances done', end=", ")


def fulldistance_to_gaussian(fwhm):
    from Connectome_Spatial_Smoothing import CSS as css
    sigma=css._fwhm2sigma(fwhm)    
    gaussian = np.exp(-(gdistsL**2 / (2 * (sigma ** 2))))
    from sklearn.preprocessing import normalize
    gaussian=normalize(gaussian,norm='l1',axis=0) #so each column has sum 1.
    return gaussian


for pre_hrc_fwhm in [2,3,7,10]:
    smoother=sparse.load_npz(ospath(f'{hutils.intermediates_path}/smoothers/100610_{pre_hrc_fwhm}_0.01.npz')).astype(np.float32) 
    print(f'{c.time()}: {pre_hrc_fwhm}, Smooth hrc')
    hr2 = hutils.smooth_highres_connectomes([hr],smoother)[0].astype(np.float16)
    hr2L = hr2[hcp.struct.cortex_left,:][:,hcp.struct.cortex_left].astype(np.float32).toarray()

    print(f'{c.time()}: {pre_hrc_fwhm}, Make scatter')
    n = int(1e5) #get this many random points. Remove non-zero weights and plot
    n2 = int(1e3) #Of the non-zero weights, only plot this many
    inds_i = np.random.randint(0,29696,n) #get n random integers, each between 0 and 29696
    inds_j = np.random.randint(0,29696,n)
    dists = gdistsL[inds_i,inds_j]
    wts = hr2L[inds_i,inds_j]
    fig,ax=plt.subplots()
    non_zeros = np.nonzero(wts)
    dists = dists[non_zeros]
    wts = wts[non_zeros]
    inds = np.random.randint(0,len(dists),n2)

    logwts = np.log10(wts)
    ax.scatter(dists[inds],logwts[inds],s=2)

    #ax.scatter(dists[non_zeros],wts[non_zeros],s=2) 
    ax.set_xlabel('Geodesic distance')
    ax.set_ylabel('Log10(Weight)')
    ax.set_title(f'smooth fwhm={pre_hrc_fwhm}, n={n}')

    #fit linear regression to predict wts based on dists, and return R^2
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(dists[inds].reshape(-1,1),logwts[inds])
    print(f'{pre_hrc_fwhm}: R^2 = {reg.score(dists[inds].reshape(-1,1),logwts[inds]):.2f}')


print(f'{c.time()} done')
plt.show(block=False)
'''