"""
Script to make simple figures from data
"""

'''
#Generate example figure for the problem with in-sample templates

import matplotlib.pyplot as plt
from fmralign.alignment_methods import scaled_procrustes

nothers = 1
source = np.random.random((6,3))
others = [np.random.random((6,3)) for i in range(nothers)]
target = [source] + others
target = np.stack(target).mean(axis=0)

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

R, sc = scaled_procrustes(source,target)

corrs = np.zeros((3,3))
for source_verts in range(source.shape[1]):
    for target_verts in range(target.shape[1]):
        corrs[source_verts,target_verts] = np.corrcoef(source[:,source_verts],target[:,target_verts])[0,1]

plot_time_series(source,'Source')
plot_time_series(others[0],'Target')
plot_time_series(target,'Target=Mean(Source,Other)')

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


plt.show(block=False)
assert(0)
'''

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
    import pandas as pd
    df=pd.DataFrame(columns=[label_name,data_name])
    for key,data in dictionary.items():
        for d in data:
            df=df.append({label_name:key,data_name:d},ignore_index=True)
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
    sns.boxplot(showmeans=True, meanline=True, meanprops={'color': 'k', 'ls': '-', 'lw': 2}, medianprops={'visible': True,'lw':1}, whiskerprops={'visible': True}, x=data_name, y=label_name, data=df, showfliers=False, showbox=True, showcaps=True, linewidth=1, width=0.5, ax=ax)
    make_boxplot_transparent(fig,ax)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()


def lineplot(xdata,ydata,xlabel=None,ylabel=None,title=None,ylim=None,xlim=None,figsize=(4,4),hline=0.84):
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

"""
#Figure. Out-of-sample vs in-sample template
dictionary = {'In-sample\ntemplate':[0.94,0.94,0.89,0.92,0.86,],\
              'Out-of-sample\ntemplate A':[0.83,0.92,0.78,0.78,0.75,],\
                'Out-of-sample\ntemplate B':[0.78,0.89,0.78,0.75,0.72,],\
                'Anatomical':[0.89, 0.92, 0.81, 0.81, 0.81]}
label_name='Condition' #'Participants used to generate template'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name)


#Figure. Gamma regularization. 1-10 for alignment testing, 11-20 for template generation, 21-30 for parameter optimization 
dictionary = {'In-sample template':[0.94,0.94,0.89,0.92,0.86,],\
              'Out-of-sample template':[0.83,0.92,0.78,0.78,0.75,],\
                'Out-of-sample template\nwith regularization':[0.94,0.94,0.86,0.89,0.83,],} 
label_name='Condition'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name)
lineplot([0.02,0.05,0.1,0.2,0.5,1], [0.20, 0.35, 0.65, 0.79, 0.97,1],'Regularization parameter','Ratio within source ROI')
"""


gammas = [0,0.02,0.05,0.1,0.2,0.5,1]
ylim=[0.7,1.0]
"""
#Pairwise with subs 0-10
lineplot(gammas,[.74,.79,.79,.83,.87,.87,.84],'Regularization parameter','Mean classification accuracy','Pairwise Procrustes',ylim)
lineplot(gammas[:-1],[.72,.77,.82,.86,.89,.89],'Regularization parameter','Mean classification accuracy','Pairwise Ridge',ylim) #alpha 1e4
lineplot(gammas,[.89,.89,.89,.88,.85,.83,.83],'Regularization parameter','Mean classification accuracy','Pairwise Optimal Transport',ylim) #reg 1

#Template with subs 0-10, template generated from subs 10-20
lineplot(gammas, [0.10,0.10,0.18,0.33,0.65,0.97,1.0],'Regularization parameter','Ratio within source ROI',"GPA template Procrustes",hline=None)
lineplot(gammas, [0.81,0.82,0.85,0.87,0.90,0.89,0.84],'Regularization parameter','Mean classification accuracy','GPA template Procrustes',ylim) 
lineplot(gammas[:-1],[.74,.76,.77,.81,.86,.91],'Regularization parameter','Mean classification accuracy','GPA template Ridge',ylim) #alpha 1e3
lineplot(gammas,[.92,.92,.91,.89,.85,.83,.83],'Regularization parameter','Mean classification accuracy','GPA template Optimal Transport',ylim) #reg 1
lineplot(gammas,[.83,.85,.88,.89,.89,.86,.84],'Regularization parameter','Mean classification accuracy','PCA template Procrustes',ylim) 
lineplot(gammas[:-1],[.81,.82,.84,.89,.91,.88],'Regularization parameter','Mean classification accuracy','PCA template Ridge',ylim) #alpha 1e3

#ProMises with subs 0-10, template generated from subs 10-20
lineplot([0,.003,.01,.03,.1,.3,1,3,10],[.81,.8,.8,.8,.84,.9,.87,.86,.84],'Parameter k','Mean classification accuracy','GPA template ProMises',ylim)
"""
### With M&S ###

#Subs 0-10 pairwise
anat_acc = 0.84
lineplot(gammas, [0.8, 0.83, 0.87, 0.87, 0.88, 0.86, 0.84],'Regularization parameter','Classification accuracy','Pairwise Procrustes',ylim,hline=anat_acc) 
lineplot(gammas, [0.89, 0.89, 0.92, 0.92, 0.84, 0.84, 0.84],'Regularization parameter','Classification accuracy','Pairwise optimal transport',ylim,hline=anat_acc) 

#Subs 0-20 with subs 20-40 as template
anat_corr = 0.070
anat_acc = 0.87
ylim_corr = [.065, .180]

lineplot(gammas, [.0921, .0977, .1035, .1053, .1006, .0861, .0702],'Regularization parameter','Within-parcel correlation','GPA template Procrustes',ylim_corr,hline=anat_corr) 
lineplot(gammas, [.856, .875, .892, .9, .933, .936, .872],'Regularization parameter','Classification accuracy','GPA template Procrustes',ylim,hline=anat_acc) 
lineplot(gammas, [.1469, .1525, .1588, .1628, .1531, .1093, .0808],'Regularization parameter','Within-parcel correlation','PCA template Ridge',ylim_corr,hline=anat_corr) 
lineplot(gammas, [.892, .897, .911, .925, .931, .917, .883],'Regularization parameter','Classification accuracy','PCA template Ridge',ylim,hline=anat_acc) 

lineplot([0,.01,.03,.1,.3,1,3,10], [0.86, 0.86, 0.87, 0.89, 0.94, 0.93, 0.91, 0.89],'Regularization parameter','Classification accuracy','GPA template Procrustes ProMises',ylim,hline=anat_acc) 
lineplot([0,.01,.03,.1,.3,1,3,10], [0.86, 0.86, 0.87, 0.89, 0.94, 0.93, 0.91,0.89],'Regularization parameter','Classification accuracy','GPA template Procrustes ProMises',ylim,hline=anat_acc) 

#Subs 40-60 with with subs 20-40 as template, with gamma optimised for subs 0-20. m&s. Procrustes template
dictionary = {'\u03B3=1\n(anatomical)':[0.792,0.903,0.889,0.861,0.806,0.889,0.861,0.903,0.875,0.875,],\
              '\u03B3=0\n(functional)':[0.76,0.86,0.88,0.78,0.86,0.89,0.83,0.92,0.90,0.82,],\
                '\u03B3=0.5\n(combined)':[0.88,0.93,0.93,0.86,0.86,0.93,0.90,0.93,0.94,0.96,],\
                    'Parcel-specific \u03B3':[0.88,0.92,0.94,0.86,0.89,0.93,0.93,0.97,0.94,0.90,]} 
label_name='Condition'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name,title='GPA template Procrustes')

p_0_vs_05=stats.ttest_rel([0.76,0.86,0.88,0.78,0.86,0.89,0.83,0.92,0.90,0.82,],[0.88,0.93,0.93,0.86,0.86,0.93,0.90,0.93,0.94,0.96,])
p_05_vs_custom=stats.ttest_rel([0.88,0.92,0.94,0.86,0.89,0.93,0.93,0.97,0.94,0.90,],[0.88,0.93,0.93,0.86,0.86,0.93,0.90,0.93,0.94,0.96,])
print(f'p_0_vs_05: T={p_0_vs_05[0]:.3f} p={p_0_vs_05[1]:.3f}\np_05_vs_custom: T={p_05_vs_custom[0]:.3f} p={p_05_vs_custom[1]:.3f}')


#Subs 40-60 and 60-80 with with subs 20-40 as template, with gamma optimised for subs 0-20. m&s. Ridge LowDim Template
dictionary = {'\u03B3=1\n(anatomical)':[0.792,0.903,0.889,0.861,0.806,0.889,0.861,0.903,0.875,0.875,],\
              '\u03B3=0\n(functional)':[0.861,0.917,0.917,0.847,0.875,0.917,0.903,0.917,0.903,0.889,],\
                '\u03B3=0.2\n(combined)':[0.861,0.958,0.944,0.847,0.889,0.944,0.903,0.958,0.931,0.931,],\
                    'Parcel-specific \u03B3':[0.889,0.931,0.917,0.819,0.903,0.917,0.931,0.931,0.917,0.944,]} 
label_name='Condition'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name,title='PCA template Ridge')

p_0_vs_05=stats.ttest_rel([0.861,0.917,0.917,0.847,0.875,0.917,0.903,0.917,0.903,0.889,],[0.861,0.958,0.944,0.847,0.889,0.944,0.903,0.958,0.931,0.931,])
p_05_vs_custom=stats.ttest_rel([0.889,0.931,0.917,0.819,0.903,0.917,0.931,0.931,0.917,0.944,],[0.861,0.958,0.944,0.847,0.889,0.944,0.903,0.958,0.931,0.931,])
print(f'p_0_vs_05: T={p_0_vs_05[0]:.3f} p={p_0_vs_05[1]:.3f}\np_05_vs_custom: T={p_05_vs_custom[0]:.3f} p={p_05_vs_custom[1]:.3f}')


#Subs 40-60 with with subs 20-40 as template, with gamma optimised for subs 0-20. m&s. Procrustes template
dictionary = {'\u03B3=1\n(anatomical)':[0.792,0.903,0.889,0.861,0.806,0.889,0.861,0.903,0.875,0.875,],\
              '\u03B3=0\n(functional)':[0.76,0.86,0.88,0.78,0.86,0.89,0.83,0.92,0.90,0.82,],\
                '\u03B3=0.5\n(combined)':[0.88,0.93,0.93,0.86,0.86,0.93,0.90,0.93,0.94,0.96,],\
                    'Parcel-specific \u03B3':[0.88,0.92,0.94,0.86,0.89,0.93,0.93,0.97,0.94,0.90,]} 
label_name='Condition'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name,title='GPA template Procrustes')

p_0_vs_05=stats.ttest_rel([0.76,0.86,0.88,0.78,0.86,0.89,0.83,0.92,0.90,0.82,],[0.88,0.93,0.93,0.86,0.86,0.93,0.90,0.93,0.94,0.96,])
p_05_vs_custom=stats.ttest_rel([0.88,0.92,0.94,0.86,0.89,0.93,0.93,0.97,0.94,0.90,],[0.88,0.93,0.93,0.86,0.86,0.93,0.90,0.93,0.94,0.96,])
print(f'p_0_vs_05: T={p_0_vs_05[0]:.3f} p={p_0_vs_05[1]:.3f}\np_05_vs_custom: T={p_05_vs_custom[0]:.3f} p={p_05_vs_custom[1]:.3f}')


#Subs 40-60 and 60-80 with with subs 20-40 as template, with gamma optimised for subs 0-20. m&s. Ridge LowDim Template
dictionary = {'\u03B3=1\n(anatomical)':[0.792,0.903,0.889,0.861,0.806,0.889,0.861,0.903,0.875,0.875,],\
              '\u03B3=0\n(functional)':[0.861,0.917,0.917,0.847,0.875,0.917,0.903,0.917,0.903,0.889,],\
                '\u03B3=0.2\n(combined)':[0.861,0.958,0.944,0.847,0.889,0.944,0.903,0.958,0.931,0.931,],\
                    'Parcel-specific \u03B3':[0.889,0.931,0.917,0.819,0.903,0.917,0.931,0.931,0.917,0.944,]} 
label_name='Condition'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name,title='PCA template Ridge')

p_0_vs_05=stats.ttest_rel([0.861,0.917,0.917,0.847,0.875,0.917,0.903,0.917,0.903,0.889,],[0.861,0.958,0.944,0.847,0.889,0.944,0.903,0.958,0.931,0.931,])
p_05_vs_custom=stats.ttest_rel([0.889,0.931,0.917,0.819,0.903,0.917,0.931,0.931,0.917,0.944,],[0.861,0.958,0.944,0.847,0.889,0.944,0.903,0.958,0.931,0.931,])
print(f'p_0_vs_05: T={p_0_vs_05[0]:.3f} p={p_0_vs_05[1]:.3f}\np_05_vs_custom: T={p_05_vs_custom[0]:.3f} p={p_05_vs_custom[1]:.3f}')





plt.show()


