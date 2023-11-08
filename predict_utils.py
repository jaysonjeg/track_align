from joblib import Parallel, delayed
import numpy as np

def get_regress_function(classifier):
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
    if classifier=='svr':
        reg = SVR(kernel='linear',C=1) 
    elif classifier=='ridge':
        reg = Ridge(alpha=1.0)
    elif classifier=='ridgeCV':
        reg = RidgeCV(alphas=(1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5))
    elif classifier=='lasso':
        reg = Lasso(alpha=1.0)
    elif classifier=='lassoCV':
        reg = LassoCV(n_alphas=10)    
    return reg

def construct_pipeline(X_impute,X_StandardScaler,X_pca_features,classifier):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    pipelist = []
    if X_impute: 
        pipelist.append(SimpleImputer(missing_values=np.nan, strategy='mean'))
    if X_StandardScaler: 
        pipelist.append(StandardScaler())
    if X_pca_features: 
        pipelist.append(PCA(n_components=0.5))
    pipelist.append(get_regress_function(classifier))
    pipeline = make_pipeline(*pipelist)
    return pipeline

def reshape_and_cross_validate(c, t, pipeline,imgs_X, y):
    from sklearn.model_selection import cross_val_score, KFold
    reshape = lambda arr: arr.reshape(1,-1)
    X = np.vstack(Parallel(n_jobs=-1,prefer="threads")(delayed(reshape)(arr) for arr in imgs_X))
    if t is not None: t.print(f'{c.time()} X rearrange done') 
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    if t is not None: t.print(f'{c.time()} start CV')
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2',n_jobs=-1)
    if t is not None: t.print(f'r2 mean {np.mean(r2_scores):.3f}: {[round(i,3) for i in r2_scores]}\n')
