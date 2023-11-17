from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def get_and_cache_data():
    """
    Get data from HCP and cache it in the intermediates folder
    """
    import hcpalign_utils as hutils
    c=hutils.clock()  
    df = hutils.get_hcp_behavioral_data()
    cognitive_measures, rows_with_cognitive, rows_with_3T_taskfMRI, rows_with_3T_rsfmri, rows_with_7T_rsfmri, rows_with_7T_movie = hutils.get_rows_in_behavioral_data()
    eligible_rows = rows_with_3T_rsfmri & rows_with_3T_taskfMRI
    eligible_subjects = [str(i) for i in df.loc[eligible_rows,'Subject']]  
    print(len(eligible_rows))
    sub_slices = [slice(i, i + 10) for i in range(0, 1200, 10)]
    for sub_slice in sub_slices:
        subs = eligible_subjects[sub_slice]
        print(f'{c.time()} {hutils.memused()} {sub_slice} Get data start')
        imgs_X, X_string = hutils.get_movie_or_rest_data(subs,'rest_FC',runs=[0,1,2,3],fwhm=0,clean=True,MSMAll=False,FC_parcellation_string='S1000',FC_normalize=False)
        #imgs_X,X_string = hutils.get_task_data(subs,hutils.tasks,MSMAll=MSMAll)
        print(f'{c.time()} {sub_slice} Get data end')


def get_task_data_and_align(subs, MSMAll, func_align_file, c, t):
    import hcpalign_utils as hutils
    from hcpalign_utils import ospath
    t.print(f'{c.time()} get aligners: start')
    original_imgs,original_imgs_string = hutils.get_task_data(subs,hutils.tasks,MSMAll=MSMAll)
    X_string = f"{func_align_file}{original_imgs_string}"
    t.print(f'{c.time()} get original img end, transform img start')
    prefix = ospath(f'{hutils.intermediates_path}/alignpickles3/{func_align_file}')
    imgs = Parallel(n_jobs=-1,prefer='processes')(delayed(apply_aligner_to_img)(original_img,prefix,sub) for original_img,sub in zip(original_imgs,subs))
    t.print(f'{c.time()} transform img end')
    return imgs, X_string

def func_add_parcelwise_mean(c, t, clustering, original_imgs, imgs_aligned, X_data_aligned_string):
    import hcpalign_utils as hutils
    X_data_aligned_string = X_data_aligned_string + '&m'
    t.print(f'{c.time()} add means: start')
    all_parcelwise_means = Parallel(n_jobs=-1,prefer='threads')(delayed(hutils.get_parcelwise_mean)(img,clustering) for img in original_imgs)
    t.print(f'{c.time()} add means: made means')
    imgs_aligned = Parallel(n_jobs=-1,prefer='threads')(delayed(np.hstack)([img,parcelwise_means]) for img,parcelwise_means in zip(imgs_aligned,all_parcelwise_means))
    t.print(f'{c.time()} add means: end')
    return imgs_aligned,X_data_aligned_string

import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning
def apply_aligner_to_img(original_img,prefix,sub):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        load_sub_aligner = lambda sub: pickle.load(open(f'{prefix}/{sub}.p',"rb"))
        aligner = load_sub_aligner(sub)
    return aligner.transform(original_img)

def get_aligner_data(sub,prefix,nvertices_per_parcel):
    #get aligner matrix the subject and flatten it into a vector
    aligner_array_length = np.sum([i**2 for i in nvertices_per_parcel])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        load_sub_aligner = lambda sub: pickle.load(open(f'{prefix}/{sub}.p',"rb"))
        aligner = load_sub_aligner(sub)
        array = np.zeros(aligner_array_length,dtype=np.float16)
        j=0
        for i in range(len(nvertices_per_parcel)):
            array[j:j+nvertices_per_parcel[i]**2] = aligner.fit_[i].R.coef_ravel()
            j+=nvertices_per_parcel[i]**2
    return array

def get_feature_groups(what_feature_groups, clustering, imgs, which_parcel):
    #Generate feature_groups which allocates each datapoint/feature in each subject to a group. Groups are labelled 0, 1, 2, etc.
    if what_feature_groups == None:
        return None
    elif what_feature_groups == 'both':
        """
        Each parcel is a separate group. Each task contrast is a separate group
        Each element of feature_groups_array has the following value:  (group_number*total_number_of_rows)+row_number.
        """
        total_rows = imgs.shape[1]
        row_numbers = np.arange(total_rows).reshape(-1, 1) 
        feature_groups_array = (clustering * total_rows) + row_numbers
    elif what_feature_groups =='contrasts':
        #Each row is a separate group
        total_rows = imgs.shape[1]
        row_numbers = np.arange(total_rows).reshape(-1, 1)
        feature_groups_array = np.tile(row_numbers,(1,imgs.shape[-1]))
    elif what_feature_groups == 'parcels':
        #Each parcel is a separate group
        feature_groups_array = np.tile(clustering,(imgs.shape[1],1))

    if type(which_parcel)==int:
        feature_groups=feature_groups_array[:,clustering==which_parcel].ravel()
    elif which_parcel=='w':
        feature_groups=feature_groups_array.ravel()
    elif which_parcel=='a':
        feature_groups=feature_groups_array

    return feature_groups


def get_regress_function(classifier,feature_groups=None):
    alphas = (1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4)#,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e5,1e6,1e7,1e8,1e9,1e10)
    if classifier=='svr':
        from sklearn.svm import SVR
        reg = SVR(kernel='linear',C=1) 
    elif classifier=='ridge':
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0)
    elif classifier=='ridgeCV':
        from sklearn.linear_model import RidgeCV
        reg = RidgeCV(alphas=alphas)
    elif classifier=='lasso':
        from sklearn.linear_model import Lasso
        reg = Lasso(alpha=1.0)
    elif classifier=='lassoCV':
        from sklearn.linear_model import LassoCV
        reg = LassoCV(n_alphas=100,n_jobs=None)   
    elif classifier=='lassolarsCV':
        from sklearn.linear_model import LassoLarsCV
        reg = LassoLarsCV(max_n_alphas=1000,n_jobs=-1)
    elif classifier=='sgd':
        from sklearn.linear_model import SGDRegressor
        reg = SGDRegressor(random_state=0)
    elif classifier=='adaboost':
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.linear_model import RidgeCV
        estimator = RidgeCV(alphas=alphas)
        reg = AdaBoostRegressor(estimator=estimator,random_state=0, n_estimators=50)
    elif classifier=='GBR':
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(random_state=0)
    elif classifier=='HistGBR':
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(random_state=0)
    elif classifier=='grouped':
        from sklearn.linear_model import Ridge, RidgeCV
        from sklearn.svm import SVR
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        
        base_estimator = RidgeCV(alphas=alphas)
        final_estimator = RidgeCV(alphas=alphas)

        from sklearn.pipeline import make_pipeline
        pipelist = []
        if False: 
            from sklearn.preprocessing import StandardScaler
            pipelist.append(StandardScaler())
        pipelist.append(final_estimator)
        final_estimator = make_pipeline(*pipelist)

        reg = GroupedFeaturesEstimator(cv=2,n_jobs=-1,base_estimator=base_estimator,final_estimator=final_estimator,feature_groups=feature_groups)
    return reg


def construct_pipeline(X_impute,X_StandardScaler,X_pca_features,classifier,feature_groups=None):
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
    pipelist.append(get_regress_function(classifier,feature_groups))
    pipeline = make_pipeline(*pipelist)
    return pipeline

def reshape(list_of_arrays):
    reshape = lambda arr: arr.reshape(1,-1)
    return np.vstack(Parallel(n_jobs=-1,prefer="threads")(delayed(reshape)(arr) for arr in list_of_arrays))

def do_prediction(pipeline,X, y, n_jobs=-1):
    from sklearn.model_selection import cross_val_score, KFold

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning) #for classifier == 'LassoLarsCV'
    #print('5 FOLDS!!!')
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2',n_jobs=n_jobs)
    return r2_scores

def print_r2_result(t,r2_scores):
    t.print(f'r2 mean {np.mean(r2_scores):.3f}: {[round(i,3) for i in r2_scores]}')

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin

class GroupedFeaturesEstimator(BaseEstimator, RegressorMixin):

    """
    A custom estimator that trains separate base estimators on each group of features and combines them into a final estimator. Its "fit" method uses cross-validation, to separate the samples on which base estimators are trained, and those samples on which the parameters of the final estimator are fitted.

    Attributes:
    -----------
    cv : int
        The number of cross-validation folds.
    n_jobs: int
        The number of jobs to run in parallel for cross-validation.
    base_estimator : estimator object
        The estimator used for each feature group.
    final_estimator : estimator object
        The final estimator that combines predictions from all base estimators.
    feature_groups : numpy array
        Array indicating the group of each feature.
    feature_groups_unique : numpy array
        List of unique values in feature_groups.
    n_feature_groups : int
        Number of unique feature groups.
    base_estimators : list
        List of fitted base estimators for each feature group.
    final_estimators: list
        List of fitted final estimators for each cross-validation fold.

    """

    def __init__(self, cv=2,n_jobs=-1,base_estimator=None,final_estimator=None,feature_groups=None):
        self.cv = cv
        self.n_jobs = n_jobs
        self.base_estimator = base_estimator if base_estimator is not None else Ridge(alpha=1000)
        self.final_estimator = final_estimator if final_estimator is not None else Ridge(alpha=1000)
        self.feature_groups = feature_groups
        self.feature_groups_unique = np.sort(np.unique(self.feature_groups))
        self.n_feature_groups = len(self.feature_groups_unique)
        self.base_estimators = None
        self.final_estimators = None

    def fit(self, X, y):
        """
        Split samples in X and y into training and test sets. Within the training set, train a separate base_estimator_clone for each feature group. Each base_estimator_clone produces a different prediction for each sample in the test set. The final_estimator_clone is trained on those predicted outputs in the test set. That is, the prediction from each feature group is treated as a feature for the final_estimator_clone. 

        The above procedure is repeated for separate training/test set splits using cross-validation, producing a different final_estimator_clone for each fold. Fitted parameters from these final_estimator_clones are averaged to produce the fitted final_estimator.
        
        Then, base_estimator_clones are refit on the entire data (X, y). The final_estimators retain their parameters from the previous cross-validation step.

        Parameters:
        -----------
        X: numpy array (nsamples, nfeatures)
        y: numpy array (nsamples, )
        feature_groups: numpy array of ints (nfeatures, )
        """

        
        kf = KFold(n_splits=self.cv,shuffle=True, random_state=0)
        self.final_estimators = Parallel(n_jobs=self.n_jobs,prefer='processes')(delayed(self.new_method)(train_index, test_index,X,y) for train_index, test_index in kf.split(X))
        self.base_estimators = [clone(self.base_estimator) for _ in range(self.n_feature_groups)]
        for group_index, group in enumerate(self.feature_groups_unique):
            group_indices = self.feature_groups == group
            self.base_estimators[group_index].fit(X[:,group_indices], y)
        """

        base_estimator_predictions = np.zeros((X.shape[0], self.n_feature_groups),dtype=np.float32)
        self.base_estimators = [clone(self.base_estimator) for _ in range(self.n_feature_groups)]
        for group_index, group in enumerate(self.feature_groups_unique):
            group_indices = self.feature_groups == group
            self.base_estimators[group_index].fit(X[:,group_indices], y) 
            base_estimator_predictions[:,group_index] = self.base_estimators[group_index].predict(X[:,group_indices])
        self.final_estimator.fit(base_estimator_predictions,y)
        """

    def new_method(self, train_index, test_index,X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nsamples_in_train = X_train.shape[0]
        base_estimator_predictions = np.zeros((nsamples_in_train, self.n_feature_groups),dtype=np.float32)
        for group_index, group in enumerate(self.feature_groups_unique):
            group_indices = self.feature_groups == group
            base_estimator_clone = clone(self.base_estimator)
            base_estimator_clone.fit(X_train[:,group_indices], y_train)
            y_test_predicted = base_estimator_clone.predict(X_test[:,group_indices])
            base_estimator_predictions[:,group_index] = y_test_predicted
        final_estimator_clone = clone(self.final_estimator)
        final_estimator_clone.fit(base_estimator_predictions, y_test)
        return final_estimator_clone

    def predict(self, X):
        """
        Fitted base_estimators produce predictions of y for each feature group. These predictions are used as features for each of the fitted final_estimators. The predictions of the final_estimators are averaged.
        """

        base_estimator_predictions = np.zeros((X.shape[0], self.n_feature_groups),dtype=np.float32)
        for group_index, group in enumerate(self.feature_groups_unique):
            group_indices = self.feature_groups == group
            y_predicted = self.base_estimators[group_index].predict(X[:,group_indices])
            base_estimator_predictions[:,group_index] = y_predicted

        
        predictions = [estimator.predict(base_estimator_predictions) for estimator in self.final_estimators]
        return np.mean(predictions,axis=0)     
        """
        return self.final_estimator.predict(base_estimator_predictions)
        """

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        -----------
        X : numpy array
            Test samples, shape (n_samples, n_features).
        y : numpy array
            True values for X, shape (n_samples,).

        Returns:
        --------
        float
            R^2 of self.predict(X) wrt. y.
        """
        # Using the predict method to get predictions and then scoring
        predictions = self.predict(X)
        u = ((y - predictions) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        --------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return {"cv": self.cv, 
                "base_estimator": self.base_estimator, 
                "final_estimator": self.final_estimator, 
                "feature_groups": self.feature_groups,
                "n_jobs": self.n_jobs}

