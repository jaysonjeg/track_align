"""
Make random kmeans parcellations in HCP surface grayordinates for cortex (59412)
Use Euclidean distance between vertex xyz coordinates on HCP sphere.surg.gii surface files for subject 100610. This should approximate surface geodesic distance
Uses same number of clusters in each hemisphere
Save pickle in 'D:\FORSTORAGE\Data\Project_Hyperalignment\intermediates\kmeansparcellation'
"""

import nibabel as nib
nib.imageglobals.logger.level = 40  #suppress pixdim error msg
import numpy as np
import os
import hcp_utils as hcp
import hcpalign_utils
from hcpalign_utils import ospath
import pickle
from scipy.io import savemat
c=hcpalign_utils.clock()

save_folder= 'D:\FORSTORAGE\Data\Project_Hyperalignment\intermediates\kmeansparcellation'
hcp_folder="/mnt/d/FORSTORAGE/Data/HCP_S1200"


n_clusters=102
sub="100610"
surface_type='sphere' #default sphere. Otherwise 'midthickness', 'white'

def get_kmean_labels_surf(hemisphere_sphere_file,n_clusters):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    surface=nib.load(hemisphere_sphere_file)
    vertices = surface.darrays[0].data #array (nvertices,3)
    #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vertices)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(vertices)
    return(kmeans.labels_)

for sub in ["100610"]: #["100610","102311","102816"]
    for surface_type in ['sphere']: #['sphere','very_inflated','inflated','midthickness','pial','white']:
        for n_clusters in [300]: #[4,10,30,100,300,1000,3000,10000,20000]
            print([sub,surface_type,n_clusters])

            L_sphere_file=ospath(f"{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.L.{surface_type}.32k_fs_LR.surf.gii")
            R_sphere_file=ospath(f"{hcp_folder}/{sub}/MNINonLinear/fsaverage_LR32k/{sub}.R.{surface_type}.32k_fs_LR.surf.gii")
            n_clusters_per_hemi=int(n_clusters/2)
            print(f'Clusters {n_clusters}')
            print(c.time())
            labels_L=get_kmean_labels_surf(L_sphere_file,n_clusters_per_hemi)
            labels_R=get_kmean_labels_surf(R_sphere_file,n_clusters_per_hemi) + n_clusters_per_hemi
            labels=hcpalign_utils.cortex_64kto59k(np.hstack((labels_L,labels_R)))
            print(c.time())

            
            matrix=hcpalign_utils.parc_char_matrix(labels)[1]
            assert(0)
            #remove empty parcels
            nonempty_parcels = np.array((matrix.sum(axis=1)!=0)).squeeze()
            matrix2 = matrix[nonempty_parcels,:]
            labels2=hcpalign_utils.reverse_parc_char_matrix(matrix2)

            save=ospath(f'{save_folder}/kmeansparc_sub{sub}_{surface_type}_{n_clusters}parcs')
            pickle.dump(labels2,open(f'{save}.p',"wb"))
            pickle.dump(matrix2,open(f'{save}_matrix.p',"wb"))
            #savemat(ospath(f'{save}.mat'),{'data':labels})
            #savemat(ospath(f'{save}_matrix.mat'),{'data':char_matrix})
          

#Code to visualise a parcellation

n_clusters=100
save=ospath(f'{save_folder}/kmeansparc_sub{sub}_{surface_type}_{n_clusters}parcs.p')
labels=pickle.load( open( ospath(save), "rb" ) )  
p=hcpalign_utils.surfplot('/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/figures',mesh=hcp.mesh.inflated,plot_type='open_in_browser',cmap='prism')
p.plot(labels,f'labels_sub{sub}_{surface_type}_{n_clusters}parcs')


#Code to find mean diameter (max euc dist between two points) across parcels in a parcellation
"""
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
verts_mesh=hcp.mesh.midthickness[0]
gray=hcpalign_utils.vertexmap_59kto64k()
verts=verts_mesh[gray,:]

for align_nparcs in [100,300,1000,3000,10000]:
    align_labels=hcpalign_utils.kmeans(align_nparcs)
    align_parc_matrix=hcpalign_utils.kmeans_matrix(align_nparcs) 

    results=[]
    print(f'Start at {c.time()}')
    for nlabel in range(align_labels.max()):
        v=verts[align_labels==nlabel,:]
        if len(v)>4:
            hull=ConvexHull(v)
            hullpoints=v[hull.vertices,:]
            hdist = cdist(hullpoints, hullpoints, metric='euclidean')
            bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
            maxdist=np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]])
            results.append(maxdist)
    print(f'Done at {c.time()}')
    print(f'K={align_nparcs}, mean diameter is {np.mean(results)}')
"""