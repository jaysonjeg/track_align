"""
Utility functions for tkdir.py
"""



def pr(value, string=None):
    if string is None:
        print(f'{value:.{n}f}')
    else:
        print(f'{string} {value:.{n}f}')
def get_a(z):
    b,f,a=tutils.load_f_and_a(z)
    return a
def reshape(a):
    if a.ndim==1:
        return np.transpose(np.reshape(a,(nblocks,nparcs)))
    elif a.ndim==2:
        return np.transpose(np.reshape(a,(a.shape[0],nblocks,nparcs)),(0,2,1))
    elif a.ndim==3:
        return np.transpose( np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs)) , (0,1,3,2)) #Now a is n_subs_test * n_subs_test * nparcs * nblocksperparc  
    else:
        assert(0)  
def removenans(arr):
    arr = arr[~np.isnan(arr)]
    return np.array(arr)
def count(a):
    if a.ndim==4:
        out= [tutils.count_negs(a[:,:,i,:]) for i in range(a.shape[2])]    
    elif a.ndim==3:
        out= [tutils.count_negs(a[:,:,i]) for i in range(a.shape[-1])]
    elif a.ndim==2:
        out= [tutils.count_negs(a[:,i]) for i in range(a.shape[-1])]
    return np.array(out)
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
    output = np.array(array).T
    assert(output.ndim==2)
    return output
def corr(x):
    #Correlations between columns in x, averaged across all pairs of columns (ignoring corr between a col and itself)
    values = np.corrcoef(x)
    non_ones = values!=1
    return np.mean(values[non_ones])
def OLS(Y,X):
    #Y is array(n,1). X is array of regressors (n,nregressor). Outputs R
    #e.g. X = np.column_stack((dce, tce, np.ones(len(dce))))
    import statsmodels.api as sm
    model = sm.OLS(Y, X).fit()
    print("R-squared: {:.3f}".format(model.rsquared))
    print("R: {:.3f}".format(np.sqrt(model.rsquared)))
def trinarize(mylist,cutoff):
    """
    Given a list li, convert values above 'cutoff' to 1, values below -cutoff to -1, and all other values to 0
    """
    li=np.copy(mylist)
    greater=li>cutoff
    lesser=li<-cutoff
    neither= np.logical_not(np.logical_or(greater,lesser))
    li[greater]=1
    li[lesser]=-1
    li[neither]=0
    return li