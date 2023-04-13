"""
Given outputs of tkalign.py using 'RD' and 'RT', try to compute directionality

'd' for 'RD'
't' for 'RT'
'b' for 'both' = 'RDRT'
"""

import hcpalign_utils as hutils
import matplotlib.pyplot as plt
import tkalign_utils as tutils, dir_utils as dutils
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
import matplotlib.pyplot as plt

##### Global variables ######

align_nparcs=300
nparcs=301
nblocks=5
n=4 #decimal places for answers

##### Functions ######

##### Initial preparations ######

strings=['0-10','10-20','20-30','30-40','40-50']
zd=[f'0-50_rest_{string}_RD.npy' for string in strings]
zt=[f'0-50_rest_{string}_RT.npy' for string in strings]
zb=[f'0-50_rest_{string}.npy' for string in strings]
zd = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zd] #RD
zt = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zt] #RT
zb = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/maxhpmult/{i}') for i in zb] #RDRT
d = [get_a(i) for i in zd]
t = [get_a(i) for i in zt]
b = [get_a(i) for i in zb]
blocks,_,_=tutils.load_f_and_a(zd[0]) #get blocks

##### Get the outcome measures ######

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


pr(corr(cedr.T),'do blocks belonging to a common parcel have similar RD-RT diff in counts?')
pr(corr(cxdtr.T), 'do blocks belonging to a common parcel have similar t-stat for RD-RT diff?')
pr(corr(dcxtr.T), 'do blocks belonging to a common parcel have similar t-stat for RD?')
pr(corr(tcxtr.T), 'do blocks belonging to a common parcel have similar t-stat for RT?')
pr(corr(bcxtr.T), 'do blocks belonging to a common parcel have similar t-stat for RDRT?')


##### Use correlations to find if repeated runs are similar ######
print('### Correlation between 5 folds ###')
print('Counts for each block, summed across sub-pairs: RD, RT, RDRT')
pr(corr(dne)) 
pr(corr(tne))
pr(corr(bne))

print('Counts for each parc, summed across sub-pairs and blocks: RD, RT, RDRT')
pr(corr(dnre)) 
pr(corr(tnre))
pr(corr(bnre))

pr(corr(ned),'counts(RD) - counts(RT), for each block') 
pr(corr(nred),'counts(RD) - counts(RT), for each parc') 
pr(corr(nxdt),'t-stat for RD-RT diff, calc. for each block') 

print('\n### Correlations between RD, RT, and RDRT')
print('dce vs tce vs bce: counts, across blocks')
print(np.corrcoef([dce,tce,bce]))
print('OLS of bce, predicted by dce and tce')
OLS (bce , np.column_stack((dce, tce, np.ones(len(dce)))) )
print('dcre vs tcre vs bcre: counts, across parcs')
print(np.corrcoef([dcre,tcre,bcre]))
print('dcxt vs tcxt vs bcxt: t-scores, across blocks')
print(np.corrcoef([dcxt,tcxt,bcxt])) 

#similarity between RD and RT
print("\nSimilarity between RD and RT (in counts for each block)")
print([f"{np.corrcoef(dne[i],tne[i])[0,1]:.{n}f}" for i in range(len(strings))]) 

##### Compare with null models ######

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

##### Plots on the brain ######
p=hutils.surfplot(hutils.results_path,plot_type='open_in_browser')
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)

#counts, for each parc
p.plot(dcre@align_parc_matrix,'dcre') 
p.plot(tcre@align_parc_matrix,'tcre')
p.plot(bcre@align_parc_matrix,'bcre')

#t-stat (mean across blocks), for each parc
p.plot(-dcxtr.mean(axis=1) @ align_parc_matrix, 'dcxtrm')  #compare to tkspat.py, line p.plot([-i for i in anct] @ align_parc_matrix,savename='-anct')
p.plot(-tcxtr.mean(axis=1) @ align_parc_matrix, 'tcxtrm')
p.plot(-bcxtr.mean(axis=1) @ align_parc_matrix, 'bcxtrm')

##### Plot on the brain where the directional connections are ######

p.plot(cxdtr.mean(axis=1) @ align_parc_matrix ,savename='cxdtrm')
p.plot(cxdTr.mean(axis=1) @ align_parc_matrix ,savename='cxdTrm')

#Let's focus on cxdtr
t=cxdtr
tm=t.mean(axis=1)
tm_1p5=trinarize(tm,1.5)
tm_2=trinarize(tm,2)
#p.plot(tm_1p5 @ align_parc_matrix,savename='tm_1p5')
#p.plot(tm_2 @ align_parc_matrix,savename='tm_2')
