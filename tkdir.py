"""
Given outputs of tkalign.py using 'RD' and 'RT', try to compute directionality
"""

import hcpalign_utils as hutils
import matplotlib.pyplot as plt, tkalign_utils as utils
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

##### Global variables ######

align_nparcs=300
nparcs=301
nblocks=5

##### Functions ######

def get_a(z):
    b,f,a=utils.load_f_and_a(z)
    return a
def reshape(a):
    if type(a)==list:
        return np.transpose(np.reshape(a,(nblocks,nparcs)))
    else:
        return np.transpose( np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs)) , (0,1,3,2)) #Now a is n_subs_test * n_subs_test * nparcs * nblocksperparc
        
def removenans(arr):
    arr = arr[~np.isnan(arr)]
    return arr
def count_negs_for_each_parc(a):
    if a.ndim==4:
        return [utils.count_negs(a[:,:,i,:]) for i in range(a.shape[2])]    
    elif a.ndim==3:
        return [utils.count_negs(a[:,:,i]) for i in range(a.shape[2])]
    elif a.ndim==2:
        return [utils.count_negs(a[i,:]) for i in range(a.shape[0])]
def minus(a,b):
    #subtract corresponding elements in two lists
    return [i-j for i,j in zip(a,b)]
def get_vals(a):
    array=[]
    for i in range(a.shape[2]):
        if a.ndim==4:
            values = removenans(a[:,:,i,:])
        elif a.ndim==3:
            values = removenans(a[:,:,i])
        array.append(values)
    return np.array(array)
def tstat(a):
    array=[]
    for i in range(a.shape[0]):
        t,p=ttest_1samp(a[i,:],0)
        array.append(t)
    return array
def trinarize(li,cutoff):
    """
    Given a list li, convert values above 'cutoff' to 1, values below -cutoff to -1, and all other values to 0
    """
    greater=li>cutoff
    lesser=li<-cutoff
    neither= -cutoff < li and li < cutoff
    li[greater]=1
    li[lesser]=-1
    li[neither]=0
    return li


##### Initial preparations ######

strings=['0-10','10-20','20-30','30-40','40-50']
zd=[f'0-50_rest_{string}_RD.npy' for string in strings]
zt=[f'0-50_rest_{string}_RT.npy' for string in strings]
zd = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zd]
zt = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zt]
ad = [get_a(i) for i in zd]
at = [get_a(i) for i in zt]
blocks,_,_=utils.load_f_and_a(zd[0]) #get blocks



##### Get the outcome measures ######

#subtract nonscrambled
adn=[utils.subtract_nonscrambled_from_a(i) for i in ad]
atn=[utils.subtract_nonscrambled_from_a(i) for i in at]
adnc=np.concatenate(adn,axis=0) 
atnc=np.concatenate(atn,axis=0) 

#remove the nans
adx=[get_vals(i) for i in adn]
atx=[get_vals(i) for i in atn]
adncx=get_vals(adnc)
atncx=get_vals(atnc)

#count_negs, averaged across subject-pairs
adnN=[count_negs_for_each_parc(i) for i in adx]
atnN=[count_negs_for_each_parc(i) for i in atx]
adncN = count_negs_for_each_parc(adncx)
atncN = count_negs_for_each_parc(atncx)

anND=[minus(a,b) for a,b in zip(adnN,atnN)] #RD-RT difference for count_negs (av across sub-pairs)
ancND=minus(adncN,atncN) #important

axD = [a-b for a,b in zip(adx,atx)] #RD-RT difference for each subjectpair*block values
ancxD=adncx-atncx 

axDT = [tstat(i) for i in axD] #t-test on the difference
ancxDT=tstat(ancxD) #important

##### Use correlations to find if repeated runs are similar ######

#similarity between (0,5) and (5,10)
print(np.corrcoef(adnN)) #RD: count_negs for each parcel, averaged across sub-pairs
print(np.corrcoef(adT)) #RD: t-statistic for each parcel, calculated with all sub-pairs
print(np.corrcoef(anND)) #count_negs(RD) - count_negs(RT)
print(np.corrcoef(axDT)) #t-stat for RD-RT difference, calc. for each subpair*block

#similarity between RD and RT
print([np.corrcoef(adnN[i],atnN[i])[0,1] for i in range(len(strings))]) 


##### Compare with null models ######

#Compare ancND with null model. Cutoffs: loose: |values| > 5, tight: |values| > 10
"""
adncx_null=np.random.normal(size=adncx.shape) #assumes 50% positives and negatives
atncx_null=np.random.normal(size=atncx.shape)
adncN_null = count_negs_for_each_parc(adncx_null)
atncN_null = count_negs_for_each_parc(atncx_null)
ancND_null=minus(adncN_null,atncN_null) 
plt.hist(ancND_null,10,color='b',alpha=0.3)
plt.hist(ancND,10,color='r',alpha=0.3)
plt.show()
"""

#Compare ancxDT with null model. Cutoffs: loose: |values| > 1.5, tight: |values| > 2.5
"""
ancxD_null=np.copy(ancxD)
ancxD_null=ancxD_null.ravel()
np.random.shuffle(ancxD_null)
ancxD_null=np.reshape(ancxD_null,ancxD.shape)
ancxDT_null=tstat(ancxD_null)
plt.hist(ancxDT_null,10,color='b',alpha=0.3)
plt.hist(ancxDT,10,color='r',alpha=0.3)
plt.show()
"""

##### Plot on the brain where the directional connections are ######

#The 2 outcome measures
ancNDr=reshape(ancND)
ancxDTr=reshape(ancxDT)

#Let's focus on ancxDTr
t=ancxDTr
tm=t.mean(axis=1)
tm_1p5=trinarize(tm,1.5)
tm_2=trinarize(tm,2)

p=hutils.surfplot(hutils.results_path,plot_type='open_in_browser')
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)
p.plot(tm,savename='tm')
p.plot(tm_1p5,savename='tm_1p5')
p.plot(tm_2,savename='tm_2')

'''


#Plot spatial distribution of struct-func linkage
p=hutils.surfplot(hutils.results_path,plot_type='save_as_html')
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)
ancN_cortex = ancN @ align_parc_matrix
p.plot(ancN_cortex,savename='ancN',vmin = min(ancN_cortex))
p.vmin=None
p.plot([-i for i in ancm] @ align_parc_matrix,savename='-ancm')
p.plot([-i for i in anct] @ align_parc_matrix,savename='-anct')

#Plot parcel sizes as # of vertices (?confounder)
parc_sizes=np.array(align_parc_matrix.sum(axis=1)).squeeze()
parc_sizes[0]=parc_sizes.mean()
p.plot(parc_sizes @ align_parc_matrix, savename='parc_sizes')

#Plot aligner scale factor (?confounder)
alignfile = 'hcpalign_movie_temp_scaled_orthogonal_10-4-7_TF_0_0_0_FFF_S300_False'
aligner_file = f'{hutils.intermediates_path}/alignpickles/{alignfile}.p'
all_aligners = pickle.load( open( ospath(aligner_file), "rb" ))
scales = np.vstack( [[all_aligners.estimators[i].fit_[nparc].scale for nparc in range(nparcs)] for i in range(len(all_aligners.estimators))] )   
scales_mean=scales.mean(axis=0)    
log_scales_mean=np.log10(scales_mean)
log_scales_mean_adj = log_scales_mean - log_scales_mean.min()
p.plot(scales_mean @ align_parc_matrix, savename='scales_mean')  
p.plot(log_scales_mean_adj @ align_parc_matrix, savename='log_scales_mean_adj')  

#Plot vertex areas (?confounder)
path='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\old_intermediates\\vertex_areas\\vert_area_100610_white.npy'
vertex_areas=np.load(path)
total_vertex_areas_parc = align_parc_matrix @ vertex_areas 
mean_vert_areas = total_vertex_areas_parc / parc_sizes
p.plot(vertex_areas,savename='vertex_areas')

#Plot mesh SFM
path='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\old_intermediates\\SFM\\SFM_white_sub100610.pickle'
import pickle
sfms=pickle.load(open(path,"rb"))[0]
curv_av=[np.trace(i) for i in sfms]
curv_av_abs=np.abs(curv_av)
curv_gauss=[np.linalg.det(i) for i in sfms]
p.plot(curv_av_abs,'curv_av_abs')
curv_av_parc = ( align_parc_matrix @ curv_av ) / parc_sizes
curv_av_abs_parc = ( align_parc_matrix @ curv_av_abs ) / parc_sizes

#Are spatial variations in SC-func linkage associated with these confounders?
print(f'Corr between ancN and scales_mean is {np.corrcoef(ancN,scales_mean)[0,1]}')
print(f'Corr between ancN and parc_sizes is {np.corrcoef(ancN,parc_sizes)[0,1]}')
print(f'Corr between ancN and mean_vert_areas is {np.corrcoef(ancN,mean_vert_areas)[0,1]}')
print(f'Corr between ancN and total_parcel_area is {np.corrcoef(ancN,total_vertex_areas_parc)[0,1]}')
print(f'Corr between ancN and average curvature is {np.corrcoef(ancN,curv_av_parc)[0,1]}')
print(f'Corr between ancN and abs-value of average curvature is {np.corrcoef(ancN,curv_av_abs_parc)[0,1]}')
'''