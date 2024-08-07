"""
Given outputs of tkalign.py using 'RD' and 'RT', try to compute directionality

'd' for 'RD'
't' for 'RT'
'b' for 'both' = 'RDRT'
"""

import hcpalign_utils as hutils
import matplotlib.pyplot as plt
import tkalign_utils as tutils
import dir_utils as dutils
from dir_utils import count, reshape, get_vals, pr, corr
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
import matplotlib.pyplot as plt


##### INITIAL PREPARATIONS ######
show_dir=False
nsubs=50

"""
nsubs=10
strings=['0-5','5-10']
"""

#for niter2
"""
strings=['0-10','10-20','20-30','30-40','40-50']
zb=[f'0-{nsubs}_rest_{string}.npy' for string in strings]
zb = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/maxhpmult/{i}') for i in zb] #RDRT
"""

#for niter1 (default)
"""
strings=['0-9','10-19','20-29','30-39','40-49']
zb=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_5mm_5mm_300p_5b_ma_RDRT.npy' for string in strings]
zb = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1/{i}') for i in zb] #RDRT
"""
#for niter1 with reg 0.05 (GOOD)

strings=['0-9','10-19','20-29','30-39','40-49']
zb=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_5mm_5mm_300p_5b_ma_RDRT.npy' for string in strings]
zb = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1_reg0.05/{i}') for i in zb] #RDRT
xb = [dutils.get_a(i) for i in zb]
xbn=[tutils.subtract_nonscrambled_from_a(i) for i in xb]
xbc=np.concatenate(xbn,axis=0) 
xbcre = count(reshape(xbc)) #for correlating with bcre

#For RD and RT
"""
strings=['0-10','10-20','20-30','30-40','40-50']
zd=[f'0-{nsubs}_rest_{string}_RD.npy' for string in strings]
zt=[f'0-{nsubs}_rest_{string}_RT.npy' for string in strings]
zd = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zd] #RD
zt = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zt] #RT
"""

#For RD+ and RT+ (default)

strings=['0-9','10-19','20-29','30-39','40-49']
zd=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_5mm_5mm_300p_5b_ma_RD+.npy' for string in strings]
zt=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_5mm_5mm_300p_5b_ma_RT+.npy' for string in strings]
zd = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/old/direction/{i}') for i in zd] #RD
zt = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/old/direction/{i}') for i in zt] #RT



#for niter1 smoothing 3,3, RDRT and RD+/RT+
"""
strings=['0-9','10-19','20-29','30-39','40-49']
zb=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_3mm_3mm_300p_5b_ma_RDRT.npy' for string in strings]
zb = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1/{i}') for i in zb] #RDRT
zd=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_3mm_3mm_300p_5b_ma_RD+.npy' for string in strings]
zt=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_3mm_3mm_300p_5b_ma_RT+.npy' for string in strings]
zd = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1/{i}') for i in zd] #RD
zt = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1/{i}') for i in zt] #RT
"""

#for rest_FC, niter1, reg 0.2, smoothing 5,5, RDRT and RD+/RT+ (GOOD)
"""
parcellation='kmeans3000' #'Schaefer1000', 'kmeans3000'
strings=['0-9','10-19','20-29','30-39','40-49']
zb=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_5mm_5mm_300p_5b_ma_RDRT{parcellation}.npy' for string in strings]
zd=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_5mm_5mm_300p_5b_ma_RD+{parcellation}.npy' for string in strings]
zt=[f'corrs_0-{nsubs}_40s_{string}_tracks_5M_1M_end_5mm_5mm_300p_5b_ma_RT+{parcellation}.npy' for string in strings]
zb = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1_restFC_reg0.2/{i}') for i in zb] #RDRT
zd = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1_restFC_reg0.2/{i}') for i in zd] #RD
zt = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/niter1_restFC_reg0.2/{i}') for i in zt] #RT
"""



d = [dutils.get_a(i) for i in zd]
t = [dutils.get_a(i) for i in zt]
b = [dutils.get_a(i) for i in zb]
blocks,_,_=tutils.load_f_and_a(zd[0]) #get blocks

##### GET OUTCOME MEASURES ######

#Get confounders

alignfile_movie = 'hcpalign_movie_temp_scaled_orthogonal_10-4-7_TF_0_0_0_FFF_S300_False'
alignfile_rsfmri = 'hcpalign_rest_FC_temp_scaled_orthogonal_10-4-7_TF_0_0_0_FFF_S300_False_FCSchaefer1000'

nverts_parc = dutils.get_nverts_parc()
scales_mean_movie,log_scales_mean_adj_movie = dutils.get_aligner_scale_factor(alignfile_movie)
scales_mean_rsfmri,log_scales_mean_adj_rsfmri = dutils.get_aligner_scale_factor(alignfile_rsfmri)
aligner_variability_rsfmri,Y = dutils.get_aligner_variability(alignfile_rsfmri)
aligner_variability_movie,_ = dutils.get_aligner_variability(alignfile_movie)
total_areas_parc , mean_vert_areas_parc = dutils.get_vertex_areas()
mean_strengths_50subs = dutils.get_mean_strengths()

#subtract nonscrambled
dn=[tutils.subtract_nonscrambled_from_a(i) for i in d] #'n' means per fold
tn=[tutils.subtract_nonscrambled_from_a(i) for i in t]
bn=[tutils.subtract_nonscrambled_from_a(i) for i in b]
dc=np.concatenate(dn,axis=0) #'c' means concatenated across allf olds
tc=np.concatenate(tn,axis=0) 
bc=np.concatenate(bn,axis=0) 

#remove the nans and collapse 2 subject dimensions into 1
dnx=[get_vals(i) for i in dn]
tnx=[get_vals(i) for i in tn]
bnx=[get_vals(i) for i in bn]
dcx=get_vals(dc)
tcx=get_vals(tc)
bcx=get_vals(bc)

#total negative numbers (orig>scram), for each block, averaged across subject-pairs
dne=[count(i) for i in dn] #'e' for enumerate
tne=[count(i) for i in tn]
bne=[count(i) for i in bn]
dce = count(dc)
tce = count(tc)
bce = count(bc)

#as above, for each parcel (add all the blocks contributing to that parcel)
dnre=[count(reshape(i)) for i in dn] #'r' for reshape
tnre=[count(reshape(i)) for i in tn]
bnre=[count(reshape(i)) for i in bn]
dcre = count(reshape(dc))
tcre = count(reshape(tc))
bcre = count(reshape(bc))

#dnxt=[ttest_1samp(i,0,axis=0) for i in dnx]
dcxt=ttest_1samp(dcx,0,axis=0)[0]
tcxt=ttest_1samp(tcx,0,axis=0)[0]
bcxt=ttest_1samp(bcx,0,axis=0)[0]

dcxtr=reshape(dcxt)
tcxtr=reshape(tcxt)
bcxtr=reshape(bcxt)

ned=[a-b for a,b in zip(dne,tne)] #RD-RT diff in counts, for each block, av across sub-pairs
ced=dce-tce 

nred=[a-b for a,b in zip(dnre,tnre)] #RD-RT diff in counts, for each parc, av across sub-pairs and included blocks
cred=dcre-tcre

nxd = [a-b for a,b in zip(dnx,tnx)] #RD-RT difference for each subjectpair*block values
cxd = dcx-tcx
nxdt=[ttest_1samp(i,0,axis=0)[0] for i in nxd] #t-test on the difference
cxdt=ttest_1samp(cxd,0,axis=0)[0]
cxdT=ttest_rel(dcx,tcx,axis=0)[0] #weirdly similar to cxdt (implies there are no pair-relationships)

cedr=reshape(ced) #important
cxdtr=reshape(cxdt) #important
cxdTr=reshape(cxdT)

#### PRINT SOME OUTPUTS ####
print('\n### Correlations between blocks belonging to same parcel ###')
if show_dir:
    pr(corr(cedr.T),'do blocks belonging to a common parcel have similar RD-RT diff in counts?')
    pr(corr(cxdtr.T), 'do blocks belonging to a common parcel have similar t-stat for RD-RT diff?')
    pr(corr(dcxtr.T), 'do blocks belonging to a common parcel have similar t-stat for RD?')
    pr(corr(tcxtr.T), 'do blocks belonging to a common parcel have similar t-stat for RT?')

pr(corr(bcxtr.T), 'do blocks belonging to a common parcel have similar t-stat for RDRT?')


print('\n### Correlation between 5 folds ###')
print('Counts for each block, summed across sub-pairs: RD, RT, RDRT')
if show_dir:
    pr(corr(dne),'RD') 
    pr(corr(tne),'RT')
pr(corr(bne),'RDRT')
print('Counts for each parc, summed across sub-pairs and blocks: RD, RT, RDRT')
if show_dir:
    pr(corr(dnre),'RD') 
    pr(corr(tnre),'RT')
pr(corr(bnre),'RDRT')
if show_dir:
    pr(corr(ned),'counts(RD) - counts(RT), for each block') 
    pr(corr(nred),'counts(RD) - counts(RT), for each parc') 
    pr(corr(nxdt),'t-stat for RD-RT diff, calc. for each block') 


print('\n### Correlations between RD, RT, and RDRT ###')
print('dce vs tce vs bce: counts, across blocks')
print(np.corrcoef([dce,tce,bce]))
model=dutils.OLS (bce ,[ dce,tce ] )
print('OLS of bce, predicted by dce and tce')
print("R-squared: {:.3f}".format(model.rsquared))
print("R: {:.3f}".format(np.sqrt(model.rsquared)))

print('dcre vs tcre vs bcre: counts, across parcs')
print(np.corrcoef([dcre,tcre,bcre]))
print('dcxt vs tcxt vs bcxt: t-scores, across blocks')
print(np.corrcoef([dcxt,tcxt,bcxt])) 

#similarity between RD and RT
if show_dir:
    print("\nSimilarity between RD and RT (in counts for each block)")
    print([f"{np.corrcoef(dne[i],tne[i])[0,1]:.{n}f}" for i in range(len(strings))]) 


#Are spatial variations in SC-func linkage associated with these confounders?
print(f'\n#### Correlation between bcre and () is () ####')
print(f'aligner scale (mean) (movie), {np.corrcoef(bcre,scales_mean_movie)[0,1]}')
print(f'aligner scale (mean) (rsfmri), {np.corrcoef(bcre,scales_mean_rsfmri)[0,1]}')
print(f'aligner_variability (movie), {np.corrcoef(bcre,aligner_variability_movie)[0,1]}')
print(f'aligner_variability (rsfmri), {np.corrcoef(bcre,aligner_variability_rsfmri)[0,1]}')
print(f'no of vertices in the parcel, {np.corrcoef(bcre,nverts_parc)[0,1]}')
print(f'mean vert area in the parcel, {np.corrcoef(bcre,mean_vert_areas_parc)[0,1]}')
print(f'total parcel area, {np.corrcoef(bcre,total_areas_parc)[0,1]}')
print(f'mean node strength in parcel, {np.corrcoef(bcre,mean_strengths_50subs)[0,1]}')

##### GET NULL MODELS ######

#Compare ced with null model. Cutoffs: loose: |values| > 5, tight: |values| > 10
"""
dcx_null=np.random.normal(size=dcx.shape) #assumes 50% positives and negatives
tcx_null=np.random.normal(size=tcx.shape)
dce_null = count(dcx_null)
tce_null = count(tcx_null)
ced_null=dce_null-tce_null #important
plt.hist(ced_null,10,color='b',alpha=0.3)
plt.hist(ced,10,color='r',alpha=0.3)
plt.show()
"""

#Compare cxdt with null model. Cutoffs: loose: |values| > 1.5, tight: |values| > 2.5
"""
cxd_null=np.copy(cxd)
cxd_null=cxd_null.ravel()
np.random.shuffle(cxd_null)
cxd_null=np.reshape(cxd_null,cxd.shape)
cxdt_null=tstat(cxd_null)
plt.hist(cxdt_null,10,color='b',alpha=0.3)
plt.hist(cxdt,10,color='r',alpha=0.3)
plt.show()
"""

##### PLOTS ######

def ploto(data,savename='',cmap=None):
    p.plot(data @ align_parc_matrix, savename=savename,cmap=cmap)

def plotstr(string_data,savename='',cmap='tab20'):
    #like ploto except convert string data to numbers
    uniques=np.unique(string_data)
    int_data = [np.where(uniques==string)[0][0] for string in string_data]
    ploto(int_data,savename=savename,cmap=cmap)

align_nparcs=300
p=hutils.surfplot(hutils.results_path,plot_type='open_in_browser')
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)


### Plot non-directional stuff ###

"""
#counts, for each parc
if show_dir:
    ploto(dcre,'dcre')
    ploto(tcre,'tcre')
ploto(bcre,'bcre')

#t-stat (mean across blocks), for each parc
if show_dir:
    ploto(-dcxtr.mean(axis=1),'dcxtrm') #compare to tkspat.py, line p.plot([-i for i in anct] @ align_parc_matrix,savename='-anct')
    ploto(-tcxtr.mean(axis=1),'tcxtrm')
ploto(-bcxtr.mean(axis=1),'bcxtrm')

### Plot directional stuff ###
if show_dir:
    ploto(cxdtr.mean(axis=1),'cxdtrm')

#Let's focus on cxdtr
tm=cxdtr.mean(axis=1)
#p.plot(dutils.trinarize(tm,1.5) @ align_parc_matrix,savename='tm_1p5')
#p.plot(dutils.trinarize(tm,2) @ align_parc_matrix,savename='tm_2')

"""

### Plot confounders ###

ploto(log_scales_mean_adj_movie,'log_scales_mean_adj_movie') 
ploto(log_scales_mean_adj_rsfmri ,'log_scales_mean_adj_rsfmri') 
"""
ploto(aligner_variability_rsfmri, cmap='inferno' ,savename='aligner_variability_rsfmri') 
ploto(nverts_parc ,savename='nverts_parc')
ploto(mean_vert_areas_parc ,'mean_vert_areas_parc')
ploto(total_areas_parc ,'total_areas_parc')
ploto(mean_strengths_50subs,'mean_node_strength')
"""

regs=[log_scales_mean_adj_movie,log_scales_mean_adj_rsfmri,aligner_variability_rsfmri,nverts_parc,mean_vert_areas_parc,total_areas_parc,mean_strengths_50subs]

"""
plt.scatter(xbcre,bcre)
plt.xlabel('Coupling from movie viewing')
plt.ylabel('Coupling from resting state functional connectivity')
"""


#### Mean coupling in each functional network or anatomical cluster (Yeolab)
"""
from make_schaefer_parcellation import get_Schaefer_parcelnames
nparcs=300
s300=get_Schaefer_parcelnames(nparcs=nparcs)
s300=np.vstack([['','','',''],s300]) #top row is the '0' parcel

s300_hemi=s300[:,0]
s300_networks=s300[:,1]
s300_subregions=s300[:,2]
s300_regions=[f'{i}_{j}' for i,j in zip(s300_networks,s300_subregions)]
s300_unetworks=np.unique(s300_networks)
s300_usubregions=np.unique(s300_subregions)
s300_uregions=np.unique(s300_regions)

bcre_networkmeans=[]
bcre_networksize=[]
for network in s300_unetworks:
    inds=np.where(s300_networks==network)[0]
    bcre_networkmeans.append(bcre[inds].mean())
    bcre_networksize.append(len(inds))
bcre_subregionmeans=[]
bcre_subregionsize=[]
for subregion in s300_usubregions:
    inds=np.where(s300_subregions==subregion)[0]
    bcre_subregionmeans.append(bcre[inds].mean())
    bcre_subregionsize.append(len(inds))

nulls = dutils.get_all_nulls(bcre)   
bcre_networkpercs=dutils.get_percentiles(bcre_networkmeans,bcre_networksize,nulls)
bcre_subregionpercs=dutils.get_percentiles(bcre_subregionmeans,bcre_subregionsize,nulls)

cmap_networks=plt.get_cmap('nipy_spectral',len(s300_unetworks))
cmap_subregions=plt.get_cmap('nipy_spectral',len(s300_usubregions))
plotstr(s300_networks,cmap=cmap_networks) #17 networks
plotstr(s300_subregions,cmap=cmap_subregions) #subregions out of 36
dutils.bar_plot(bcre_networkmeans[:],s300_unetworks[:],'17 networks', [65,76],cmap=cmap_networks,leftmargin=0.3)
dutils.bar_plot(bcre_networkmeans[:],s300_unetworks[:],'17 networks', [55,85],cmap=cmap_networks,leftmargin=0.3)
dutils.bar_plot(bcre_subregionmeans[:],s300_usubregions[:],'36 components', [55,85],cmap_subregions,leftmargin=0.4)

dutils.bar_plot(bcre_networkpercs[:],s300_unetworks[:],'17 networks (perc)', [0,100],cmap=cmap_networks,vertlines=True,leftmargin=0.3)
dutils.bar_plot(bcre_subregionpercs[:],s300_usubregions[:],'36 components (perc)', [0,100],cmap_subregions,vertlines=True,leftmargin=0.4)

plt.show()
"""