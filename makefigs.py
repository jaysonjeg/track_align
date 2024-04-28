"""
Script to make simple figures from data
Can use conda env nilearn

Use for figures in AlignmentAnatFunc paper
"""


#Generate example figure for the problem with in-sample templates. Also tests DOOMSDAY
'''
import matplotlib.pyplot as plt
from fmralign.alignment_methods import scaled_procrustes
import numpy as np

def plot_time_series(array,title=''):
    """
    Given an array of dimensions (time,features), show lineplot the time series for each feature in a single plot using a different colour for each line
    """
    fig,axs = plt.subplots(1,array.shape[1],figsize=(6,4))
    for i in range(array.shape[1]):
        axs[i].plot(array[:,i])
        axs[i].set_title(f'Vertex {i+1}')
        axs[i].set_ylim([0,1])
        axs[i].set_xlabel('Time')
    fig.suptitle(title)
    fig.tight_layout()

nsamples = 1000
nvertices_covarying = 200  #number of covarying vertices. Needs to be over 200
nvertices_independent = 200 #number of independent vertices. Any number okay
nvertices = nvertices_independent + nvertices_covarying
ntemplateimages = 1 #how many images in template
#x = np.random.randn(nsamples,nvertices)

#Generate template images
imgs_template = [np.random.randn(nsamples,nvertices) for i in range(ntemplateimages)] 
mean_imgs_template = np.stack(imgs_template).mean(axis=0)

#Generate correlated vertex time series
cov = np.random.random((nvertices_covarying,nvertices_covarying))
cov = (cov+40)/41 #covariances 0.8 - 1
cov = (cov+cov.T)/2
np.fill_diagonal(cov,1)
x_correlated = np.random.multivariate_normal(np.zeros(nvertices_covarying),cov,nsamples)

#Generate independent vertex time series
x_independent = np.random.randn(nsamples,nvertices_independent)

#Combine the two
x = np.concatenate((x_correlated,x_independent),axis=1) 

#Procrustes alignment with gamma regularization
gamma = 0.3
align_template_to_imgs = True #align from template to image (True) or image to template (False). Default True
add_source_to_target = True #gamma regularization adds some of the source image to the target image (True), or vice versa (False). Default True

if align_template_to_imgs: #align from the mean of template images
    source = mean_imgs_template
    target = x
else:
    source = x
    target = mean_imgs_template
if add_source_to_target:
    target = target*(1-gamma) + source*gamma
else: #add target to source
    source = source*(1-gamma) + target*gamma


R, sc = scaled_procrustes(source,target)
absum0 = np.sum(np.abs(R),axis=0)
absum1 = np.sum(np.abs(R),axis=1)

def diag0(array):
    """
    Given a 2D array, set diagonal to zero
    """
    copy = array.copy()
    np.fill_diagonal(copy,0)
    return copy

def symmetry(array):
    """
    Evaluate symmetry of a square array after setting diagonal to zeros
    +1: symmetric
    -1: antisymmetric
    0: neither
    """
    array = diag0(array)
    Asym = 0.5*(array+array.T)
    Aanti = 0.5*(array-array.T)
    norm = lambda x: np.linalg.norm(x)
    return (norm(Asym)-norm(Aanti))/(norm(Asym)+norm(Aanti))

R_correlated = R[0:nvertices_covarying,0:nvertices_covarying]
R_independent = R[nvertices_covarying:,nvertices_covarying:]
sym_R = symmetry(R)
sym_R_correlated = symmetry(R_correlated)
sym_R_independent = symmetry(R_independent)
print(f'Symmetry of R: {sym_R:.3f}')
print(f'Symmetry of R_correlated: {sym_R_correlated:.3f}')
print(f'Symmetry of R_independent: {sym_R_independent:.3f}')

upperv_correlated = R_correlated[np.triu_indices(nvertices_covarying,1)]
lowerv_correlated = R_correlated.T[np.triu_indices(nvertices_covarying,1)]
upperv_independent = R_independent[np.triu_indices(nvertices_independent,1)]
lowerv_independent = R_independent.T[np.triu_indices(nvertices_independent,1)]


fig,axs=plt.subplots(3,5,figsize=(15,7))
axs = axs.flatten()
i=0
cax=axs[i].imshow(x,aspect='auto')
axs[i].set_title('x')
plt.colorbar(cax,ax=axs[i])
i+=1
cax=axs[i].imshow(mean_imgs_template,aspect='auto')
axs[i].set_title('mean_imgs_template')
plt.colorbar(cax,ax=axs[i])
i+=1
cax=axs[i].imshow(diag0(R),aspect='auto')
axs[i].set_title('R')
plt.colorbar(cax,ax=axs[i])
i+=1
cax=axs[i].imshow(np.abs(diag0(R)),aspect='auto')
axs[i].set_title('abs(R)')
plt.colorbar(cax,ax=axs[i])
i+=1
cax=axs[i].imshow(diag0(np.corrcoef(source.T)),aspect='auto')
axs[i].set_title('Corrs between source verts ')
plt.colorbar(cax,ax=axs[i])
i+=1
cax=axs[i].imshow(diag0(np.corrcoef(target.T)),aspect='auto')
axs[i].set_title('Corrs between target verts')
plt.colorbar(cax,ax=axs[i])
i+=1
axs[i].plot(absum0)
axs[i].set_title('absum0')
i+=1
axs[i].plot(absum1)
axs[i].set_title('absum1')
i+=1
axs[i].plot(np.diag(R))
axs[i].set_title('Diagonal of R')
i+=1
axs[i].scatter(upperv_correlated,lowerv_correlated,1,alpha=0.1,color='k')
axs[i].set_title('R_correlated upper vs lower')
i+=1
axs[i].scatter(upperv_independent,lowerv_independent,1,alpha=0.1,color='k')
axs[i].set_title('R_independent upper vs lower')
i+=1

fig.tight_layout()
plt.show(block=False)
assert(0)

"""
plot_time_series(source,'Source')
plot_time_series(target,'Target')
plot_time_series(target,'Target=Mean(Source,Other)')
"""

fig,axs=plt.subplots(source.shape[1],figsize=(4,8))
for i in range(source.shape[1]):
    axs[i].plot(source[:,i],label='Source')
    axs[i].plot(others[0][:,i],label='Other')
    axs[i].plot(target[:,i],label='Target=Mean(Source,Other)')
    axs[i].set_title(f'Vertex {i+1}')
    axs[i].set_ylim([0,1])
    axs[i].set_xlabel('Time')
    if i==0:
        axs[i].legend()
fig.tight_layout()

fig,ax=plt.subplots(figsize=(4,4))
im = ax.imshow(R)
cbar = fig.colorbar(im, ax=ax)
ax.set_xlabel('Source vertex')
ax.set_ylabel('Target vertex')
ax.set_axis_off()
fig.tight_layout()
'''

#This section tests whether vertices with high correlation with their neighbours, are related to those with high rowsum
"""
mpc = np.corrcoef(x.T)
import hcpalign_utils as hutils
mpcm  = np.max(hutils.diag0(mpc),axis=0)
R = np.abs(R)
R_diag = np.diagonal(R)
print(np.corrcoef(mpcm,R_diag))

fig,axs=plt.subplots(2)
ax=axs[0]
cax=ax.imshow(x)
fig.colorbar(cax,ax=ax)
ax=axs[1]
cax=ax.imshow(np.abs(R))
fig.colorbar(cax,ax=ax)

fig,ax=plt.subplots(4)
ax[0].plot(np.sum(R,axis=0),color='r')
ax[0].plot(np.sum(R,axis=1),color='b')
ax[1].plot(np.sum(hutils.diag0(R),axis=0),color='r')
ax[1].plot(np.sum(hutils.diag0(R),axis=1),color='b')
ax[2].plot(np.diagonal(R),color='b')
ax[3].scatter(np.diagonal(R),np.sum(R,axis=0))
plt.show(block=False)
"""



import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def dict2df(dictionary,label_name='group',data_name='data'):
    """
    Convert dictionary to a dataframe with two columns: label and data. For each key do the following. For each value in the list, append a row to the dataframe with the key as the label and the value as the data.

    Parameters:
    ----------
    dictionary: keys are labels, values are data. data is a list of numbers.
    """
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import pandas as pd
    df=pd.DataFrame(columns=[label_name,data_name])
    for key,data in dictionary.items():
        for d in data:
            #df=df.append({label_name:key,data_name:d},ignore_index=True)
            df = pd.concat([df,pd.DataFrame([{label_name:key,data_name:d}])], ignore_index=True)
    return df

def make_boxplot_transparent(fig,ax):
    for patch in ax.artists: #Make boxplot transparent
        fc = patch.get_facecolor()
        patch.set_facecolor(mpl.colors.to_rgba(fc, 0.0))
        fig.tight_layout()

def plot(dictionary,label_name='group',data_name='data',title=None):
    """
    Plot a strip plot of the data in the dictionary. Each key is a label, and each value is a list of data. The data is plotted as a strip plot, with a line at the mean
    
    Parameters:
    ----------
    dictionary: keys are labels, values are data. data is a list of numbers.
    label_name: name of the column in the dataframe that contains the labels
    data_name: name of the column in the dataframe that contains the data
    """
    df = dict2df(dictionary,label_name,data_name)
    fig,ax=plt.subplots(1,figsize=(7,4)) #(4,4)
    sns.set_context('talk')#,font_scale=1)
    sns.stripplot(ax=ax,data=df,y=label_name,x=data_name,size=6)
    #sns.pointplot(ax=ax,data=df,y=label_name,x=data_name,join=False,markers='o',color='k',scale=0.5)
    #sns.boxplot(data=df,y=label_name,x=data_name,linewidth=1,width=0.5,showfliers=False)
    #make_boxplot_transparent(fig,ax)
    ax=sns.boxplot(showmeans=False, meanline=False, meanprops={'color': 'k', 'ls': '-', 'lw': 2}, medianprops={'visible': True,'lw':1}, whiskerprops={'visible': True}, x=data_name, y=label_name, data=df, showfliers=False, showbox=True, showcaps=True, linewidth=1, width=0.5, ax=ax, boxprops={'facecolor':'none', 'edgecolor':'k'})
    #make_boxplot_transparent(fig,ax)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

def plot_paired(dictionary,x_jitter=0,y_jitter=0.05,figsize=(8,5),xlabel='',ylabel='',title='',fontsize=14,markersize=7):
    """
    Given a dictionary where each key is a condition, and the values are paired, plot a paired plot of the data. The vertical axis is the condition, and the horizontal axis is the data. The data is plotted as a strip plot, with dotted lines joining the pairs of data. The data is jittered in the x and y directions to avoid overplotting. Add a asterisk at the mean of each condition.
    """
    import pandas as pd
    sns.set_context('talk')
    df = pd.DataFrame(dictionary).loc[:,::-1]
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=x_jitter, size=df.values.shape), columns=df.columns)
    df_y_jitter = pd.DataFrame(np.random.normal(loc=0, scale=y_jitter, size=df.values.shape), columns=df.columns)
    df_zeros = pd.DataFrame(np.zeros(df.values.shape), columns=df.columns)

    df += df_x_jitter
    df_y = df_zeros + np.arange(len(df.columns)) + df_y_jitter

    fig, ax = plt.subplots(figsize = figsize)
    for col in df.columns:
        ax.plot(df[col], df_y[col], 'o', alpha=0.7, zorder=1, markersize=markersize, markeredgewidth=1) #markersize 8 alpha 0.7
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_ylim(-0.5,len(df.columns)-0.5)
    for col_number in range(len(df.columns)-1):
        column0 = df.columns[col_number]
        column1 = df.columns[col_number+1]
        #Add grey lines connecting paired data
        for idx in df.index:
            ax.plot(df.loc[idx,[column0,column1]], df_y.loc[idx,[column0,column1]], color = 'grey', linewidth = 1.0, linestyle = '--', zorder=-1) #linewidth 0.8
    fig.tight_layout()
    ax.set_xlabel(xlabel)    
    ax.set_ylabel(ylabel)  
    ax.set_title(title)
    #add asterisk at the mean of each condition, with same color as the condition
    for i in range(len(df.columns)):
        color = ax.get_children()[i].get_color()
        #color = mpl.colors.to_rgba(color, 1) #to make it darker
        #color2 = [i*1.3 for i in color]
        #color = [np.min([1.0,i]) for i in color2]
        mean = df[df.columns[i]].mean()
        ax.plot(mean,i,'*',color=color,markersize=15,markeredgewidth=1,markeredgecolor='k')
    #Set font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    return fig,ax           

def lineplot(xdata,ydata,xlabel=None,ylabel=None,title=None,ylim=None,xlim=None,figsize=(5,4),hline=0.84,fontsize=None):
    """
    Make a lineplot of xdata and ydata. xdata and ydata are lists of numbers. xlabel and ylabel are strings.
    """
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(xdata,ydata,marker='o', color='k')
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if hline is not None: ax.axhline(hline,linewidth=1, color='k',linestyle='--')
    fig.tight_layout()
    if fontsize is not None:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
    return ax

gammas = [0,0.02,0.05,0.1,0.2,0.5,1]
ylim=[0.8,1.0]

"""
### WithOUT M&S ###
#Pairwise with subs 0-10
lineplot(gammas,[.74,.79,.79,.83,.87,.87,.84],'Parameter','Mean classification accuracy','Pairwise Procrustes',ylim)
lineplot(gammas[:-1],[.72,.77,.82,.86,.89,.89],'Parameter','Mean classification accuracy','Pairwise Ridge',ylim) #alpha 1e4
lineplot(gammas,[.89,.89,.89,.88,.85,.83,.83],'Parameter','Mean classification accuracy','Pairwise Optimal Transport',ylim) #reg 1

#Template with subs 0-10, template generated from subs 10-20
lineplot(gammas, [0.10,0.10,0.18,0.33,0.65,0.97,1.0],'Parameter','Ratio within source ROI',"GPA template Procrustes",hline=None)
lineplot(gammas, [0.81,0.82,0.85,0.87,0.90,0.89,0.84],'Parameter','Mean classification accuracy','GPA template Procrustes',ylim) 
lineplot(gammas[:-1],[.74,.76,.77,.81,.86,.91],'Parameter','Mean classification accuracy','GPA template Ridge',ylim) #alpha 1e3
lineplot(gammas,[.92,.92,.91,.89,.85,.83,.83],'Parameter','Mean classification accuracy','GPA template Optimal Transport',ylim) #reg 1
lineplot(gammas,[.83,.85,.88,.89,.89,.86,.84],'Parameter','Mean classification accuracy','PCA template Procrustes',ylim) 
lineplot(gammas[:-1],[.81,.82,.84,.89,.91,.88],'Parameter','Mean classification accuracy','PCA template Ridge',ylim) #alpha 1e3

#ProMises with subs 0-10, template generated from subs 10-20
lineplot([0,.003,.01,.03,.1,.3,1,3,10],[.81,.8,.8,.8,.84,.9,.87,.86,.84],'Parameter k','Mean classification accuracy','GPA template ProMises',ylim)
"""
### With M&S ###

#Stats for Supp methods 2.2. Appending parcel-specific means
#For Procrustes method
x = [0.83,0.92,0.78,0.78,0.75,] #without means
y =  [0.83,0.92,0.81,0.81,0.81,] #with means
t,p = stats.ttest_rel(y,x,alternative='greater')

#Figure 3. Out-of-sample vs in-sample template
def plot_figure3(dictionary):
    #plot(dictionary,label_name,dataname)
    plot_paired(dictionary,figsize=(7,4),x_jitter=0,y_jitter=0.05,fontsize=13,xlabel='Classification accuracy',markersize=10)
    p_in_vs_outA = stats.ttest_rel(dictionary['In-sample\ntemplate'],dictionary['Out-of-sample\ntemplate A'])
    p_in_vs_outB = stats.ttest_rel(dictionary['In-sample\ntemplate'],dictionary['Out-of-sample\ntemplate B'])
    print(f'p_in_vs_outA: T({p_in_vs_outA.df})={p_in_vs_outA[0]:.3f} p={p_in_vs_outA[1]:.3f}\np_in_vs_outB: T({p_in_vs_outB.df})={p_in_vs_outB[0]:.3f} p={p_in_vs_outB[1]:.3f}')


dictionary = {'In-sample\ntemplate':[0.94,0.94,0.89,0.92,0.86,],\
              'Out-of-sample\ntemplate A':[0.83,0.92,0.78,0.78,0.75,],\
                'Out-of-sample\ntemplate B':[0.78,0.89,0.78,0.75,0.72,],\
                'Anatomical\nMSMSulc':[0.89, 0.92, 0.81, 0.81, 0.81]}
print('\nFigure 3: align subs 1-10 to template from 1-10 (in), 11-20 (out) or 21-30 (out)')
plot_figure3(dictionary)

plt.show(block=False)

"""
#Figure 3 with LOO / 10-fold
dictionary = {'In-sample\ntemplate':[0.889,1.000,0.889,0.944,0.833,1.000,0.889,0.778,0.889,0.889,],\
              'Out-of-sample\ntemplate A':[0.778,0.944,0.833,0.778,0.778,0.944,0.889,0.722,0.944,0.778,],\
                'Out-of-sample\ntemplate B':[0.722,0.944,0.889,0.722,0.722,0.889,0.889,0.778,0.833,0.778,],\
                'Anatomical':[0.833,0.889,0.833,0.889,0.833,0.889,0.944,0.778,0.833,0.722,]}
plot_figure3(dictionary)
"""

#Figure 3 with 20 subjects, 10-fold
dictionary = {'In-sample\ntemplate':[0.833,0.861,0.889,0.944,0.722,0.917,0.833,0.833,0.917,0.917,],\
              'Out-of-sample\ntemplate A':[0.806,0.833,0.833,0.889,0.694,0.944,0.778,0.722,0.861,0.861,],\
                'Out-of-sample\ntemplate B':[0.778,0.833,0.861,0.889,0.694,0.944,0.778,0.750,0.806,0.861,],\
                }
print('\nSupp Figure 1: align subs 80-100 to template from 80-100 (in), 100-120 (out) or 120-140 (out)')
plot_figure3(dictionary)


#Figure 4e Ratio at different sample sizes
lineplot([5,10,20,50,100],[.82,.69,.53,.40,.28],'Number of participants','Spatial constraint ratio')

#Not a plot in the paper
lineplot([0.02,0.05,0.1,0.2,0.5,1], [0.20, 0.35, 0.65, 0.79, 0.97,1],'Parameter','Ratio within source ROI',ylim=[0,1])

#Supp Figure 3. Pairwise, subs 0-10
anat_acc = 0.84
lineplot(gammas, [0.8, 0.83, 0.87, 0.87, 0.88, 0.86, 0.84],'Parameter','Classification Accuracy','Pairwise Procrustes',ylim,hline=anat_acc) 
lineplot(gammas, [0.89, 0.89, 0.92, 0.92, 0.84, 0.84, 0.84],'Parameter','Classification Accuracy','Pairwise optimal transport',ylim,hline=anat_acc) 

anat_corr = 0.070
anat_acc = 0.87
ylim_corr = [.065, .180]

#### Figure 7. Compare integrated alignment to ProMises model ###

## ALIGN WITH MOVIE

# Parameter optimization cohort: subs 0-20 with subs 20-40 as template
ks = [0,.01,.03,.1,.3,1,3,10]
title='GPA template Procrustes\nProMises'
ax=lineplot(ks, [0.86, 0.86, 0.87, 0.89, 0.94, 0.93, 0.91,0.89],'Parameter','Classification Accuracy',title,ylim,hline=anat_acc) #Figure 7a
ax.set_xscale('log')

# Test cohort: subs 40-60, 60-80, and 80-100 with subs 20-40 as template
sulc_comb=[0.88,0.93,0.93,0.86,0.86,]+[0.93,0.90,0.93,0.94,0.96,]+[0.875,0.875,0.875,0.944,0.764,]
sulc_prom = [0.889,0.889,0.931,0.875,0.917,]+[0.917,0.931,0.958,0.958,0.931,]+[0.917,0.875,0.889,0.931,0.875,]
sulc_comb_vs_sulc_prom=stats.ttest_rel(sulc_comb,sulc_prom)
print(f'\nFig 7 {title}')
print(f'sulc_comb_vs_sulc_prom: T({sulc_comb_vs_sulc_prom.df})={sulc_comb_vs_sulc_prom[0]:.3f} p={sulc_comb_vs_sulc_prom[1]:.3f}')


#### Figure 6A. Parameter optimization. Subs 0-20 with subs 20-40 as template, then 6B in test cohort ###

def print_ttest(title,x,y):
    #Given vectors x and y, do a paired t-test and print the results
    statistics = stats.ttest_rel(x,y)
    print(f'{title}: T({statistics.df})={statistics[0]:.3f} p={statistics[1]:.3f}')

def print_wilcoxon(title,x,y):
    statistics = stats.wilcoxon(x, y, zero_method='wilcox', method='approx')
    print(f'{title}: W({len(x)-1})={statistics[0]:.3f} p={statistics[1]:.3f}')

def plot_figure6b(sulc,all,sulc_func,sulc_comb,all_comb,title,best_gamma):

    #Subs 40-60 and 60-80 with subs 20-40 as template, with gamma optimised for subs 0-20. m&s. Procrustes template
    dictionary = {'MSMSulc only':sulc,\
                'MSMSulc & FuncAlign\n\u03B3=0':sulc_func,\
                    f'MSMSulc & FuncAlign\n\u03B3={best_gamma} (integrated)':sulc_comb,\
                    'MSMAll only':all,\
                            f'MSMAll & FuncAlign\n\u03B3={best_gamma} (integrated)':all_comb,\
                        } 
    label_name=''
    data_name='Classification accuracy'
    #plot(dictionary,label_name,data_name,title='GPA template Procrustes')
    plot_paired(dictionary,figsize=(7,3.5),x_jitter=0,y_jitter=0.05,xlabel=data_name,title=title,fontsize=14)

    print(f'Fig 6 {title}')
    print_ttest('sulc_comb_vs_sulc_func',sulc_comb,sulc_func)
    print_ttest('sulc_comb_vs_sulc',sulc_comb,sulc)
    print_ttest('sulc_comb_vs_all',sulc_comb,all)
    print_ttest('all_comb_vs_all',all_comb,all)

    print_wilcoxon('sulc_comb_vs_sulc_func',sulc_comb,sulc_func)
    print_wilcoxon('sulc_comb_vs_sulc',sulc_comb,sulc)
    print_wilcoxon('sulc_comb_vs_all',sulc_comb,all)
    print_wilcoxon('all_comb_vs_all',all_comb,all)
    print('')


#Anatomical (MSMSUlc or MSMAll only) in test cohort
sulc=[0.792,0.903,0.889,0.861,0.806,]+[0.889,0.861,0.903,0.875,0.875,]+[0.833,0.875,0.806,0.875,0.722,] 
all=[0.847,0.958,0.917,0.861,0.903,]+[0.931,0.931,0.931,0.944,0.944,]+[0.875,0.903,0.889,0.931,0.833,] 

print('ALIGN WITH MOVIE')
#Movie: GPA template Procrustes
title='GPA template Procrustes'
lineplot(gammas, [.856, .875, .892, .9, .933, .936, .872],'Parameter','Classification Accuracy',title,ylim,hline=anat_acc,fontsize=14) #Figure 6A top
best_gamma=0.5
sulc_func=[0.76,0.86,0.88,0.78,0.86,]+[0.89,0.83,0.92,0.90,0.82,]+[0.847,0.792,0.778,0.889,0.764,] 
sulc_comb=[0.88,0.93,0.93,0.86,0.86,]+[0.93,0.90,0.93,0.94,0.96,]+[0.875,0.875,0.875,0.944,0.764,]
all_comb=[0.889,0.972,0.958,0.903,0.903,]+[0.958,0.931,0.972,0.972,0.972,]+[0.903,0.903,0.903,0.958,0.833,]
plot_figure6b(sulc,all,sulc_func,sulc_comb,all_comb,title,best_gamma) #Figure 6B top

#Movie: PCA template Ridge
title='PCA template Ridge'
lineplot(gammas, [.892, .897, .911, .925, .931, .917, .883],'Parameter','Classification Accuracy',title,ylim,hline=anat_acc,fontsize=14) #Figure 6A bottom
best_gamma=0.2
sulc_func=[0.861,0.917,0.917,0.847,0.875,]+[0.917,0.903,0.917,0.903,0.889,]+[0.889,0.889,0.889,0.931,0.861,]
sulc_comb=[0.861,0.958,0.944,0.847,0.889,]+[0.944,0.903,0.958,0.931,0.931,]+[0.931,0.944,0.917,0.944,0.819,] 
all_comb=[0.889,0.958,0.944,0.833,0.903,] + [0.944,0.944,1.000,0.958,0.972,]+[0.931,0.917,0.917,0.958,0.875,]
plot_figure6b(sulc,all,sulc_func,sulc_comb,all_comb,title,best_gamma) #Figure 6B bottom

print('ALIGN WITH REST_FC')
#Rest: GPA template Procrustes
title='GPA template Procrustes'
lineplot(gammas, [0.894, 0.894, 0.894, 0.894, 0.894, 0.9, 0.872],'Parameter','Classification Accuracy',title,ylim,hline=anat_acc,fontsize=14) #Figure 6A top
best_gamma=0.5
sulc_func=[0.833,0.889,0.917,0.847,0.903,]+ [0.917,0.917,0.903,0.903,0.889,]+[0.903,0.889,0.833,0.931,0.833,]
sulc_comb=[0.806,0.889,0.903,0.833,0.833,]+[0.903,0.861,0.903,0.889,0.861,]+[0.847,0.889,0.819,0.903,0.764,]
all_comb=[0.875,0.958,0.944,0.875,0.917,]+[0.944,0.931,0.944,0.944,0.972,]+[0.875,0.931,0.903,0.944,0.889,]
plot_figure6b(sulc,all,sulc_func,sulc_comb,all_comb,title,best_gamma) #Figure 6B top

#Rest: PCA template Ridge
title='PCA template Ridge'
lineplot(gammas, [0.897, 0.9, 0.897, 0.894, 0.903, 0.892, 0.881],'Parameter','Classification Accuracy',title,ylim,hline=anat_acc,fontsize=14) #Figure 6A top
best_gamma=0.2
sulc_func=[0.833,0.903,0.917,0.819,0.847,]+[0.903,0.917,0.889,0.875,0.861,]+[0.875,0.875,0.833,0.889,0.806,]
sulc_comb=[0.819,0.889,0.931,0.806,0.875,]+[0.875,0.903,0.903,0.917,0.875,]+[0.889,0.889,0.833,0.889,0.792,]
all_comb=[0.861,0.903,0.944,0.806,0.944,]+[0.931,0.917,0.931,0.944,0.944,]+[0.889,0.903,0.903,0.903,0.847,]
plot_figure6b(sulc,all,sulc_func,sulc_comb,all_comb,title,best_gamma) #Figure 6B top


plt.show()


