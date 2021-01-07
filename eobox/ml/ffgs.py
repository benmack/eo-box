
import numpy as np
import pandas as pd

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# THE FIRST PART OF THIS FILE (UNTIL END OF PART ONE) IS A COPY OF PARTS OF
# https://github.com/rasbt/mlxtend/blob/c928fd20e649615299128f3109d0ef9204ba08ac/mlxtend/feature_selection/sequential_feature_selector.py
# REQUIRED FOR IMPLEMENTING THE CLASS ForwardFeatureGroupSelection.
# ONLY METHODS OF SequentialFeatureSelector WHICH NEEDED TO BE CHANGED FOR THE INTENDED FUNCTONALITY
# HAVE BEEN COPIED AND MODIFIED HERE FROM THE ABOVE LINK
# 
# Author of the minor changes needed for ForwardFeatureGroupSelection: Benjamin Mack
#   
# License: BSD 3 clause
# <<<<< HEADER OF THE ORIGINAL FILE:
# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Algorithm for sequential feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
# END OF HEADER OF THE ORIGINAL FILE >>>>>

import datetime
import numpy as np
import sys
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection.sequential_feature_selector import _calc_score, _get_featurenames
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

# no change was necessary here but it has been included because else a 
# Exception has occurred: IndexError
# index 1 is out of bounds for axis 1 with size 1 ...
def _calc_score(selector, X, y, indices, groups=None, **fit_params):
    if selector.cv:
        scores = cross_val_score(selector.est_,
                                 X, y,
                                 groups=groups,
                                 cv=selector.cv,
                                 scoring=selector.scorer,
                                 n_jobs=1,
                                 pre_dispatch=selector.pre_dispatch,
                                 fit_params=fit_params)
    else:
        selector.est_.fit(X, y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X, y)])
    return indices, scores

class ForwardFeatureGroupSelection(SequentialFeatureSelector):

    """Sequential Feature Selection for Classification and Regression.
    This class is a child of the `SequentialFeatureSelector`. To implement the
    desired functionality to allow for feature *group* selection, the code of
    the `fit` and `fit_transform` methods has been copied and adjusted slightly. 
    More specifically the code was taken from
    `committ c928fd20e649615299128f3109d0ef9204ba08ac <https://github.com/rasbt/mlxtend/blob/c928fd20e649615299128f3109d0ef9204ba08ac/mlxtend/feature_selection/sequential_feature_selector.py>`_.
    Since the `ForwardFeatureGroupSelection` is forward feature selection only 
    and probably also does not support the `floating` functionality we exclude 
    the following parameters from the constructor: 
    `forward` (set to `True`),
    `floating` (set to `False`).
    Parameters
    ----------
    estimator : scikit-learn classifier or regressor
    k_features : int or tuple or str (default: 1)
        Number of features to select,
        where k_features < the full feature set.
        New in 0.4.2: A tuple containing a min and max value can be provided,
            and the SFS will consider return any feature combination between
            min and max that scored highest in cross-validtion. For example,
            the tuple (1, 4) will return any combination from
            1 up to 4 features instead of a fixed number of features k.
        New in 0.8.0: A string argument "best" or "parsimonious".
            If "best" is provided, the feature selector will return the
            feature subset with the best cross-validation performance.
            If "parsimonious" is provided as an argument, the smallest
            feature subset that is within one standard error of the
            cross-validation performance will be selected.
    verbose : int (default: 0), level of verbosity to use in logging.
        If 0, no output,
        if 1 number of features in current set, if 2 detailed logging i
        ncluding timestamp and cv scores at step.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    cv : int (default: 5)
        Integer or iterable yielding train, test splits. If cv is an integer
        and `estimator` is a classifier (or y consists of integer class
        labels) stratified k-fold. Otherwise regular k-fold cross-validation
        is performed. No cross-validation if cv is None, False, or 0.
    n_jobs : int (default: 1)
        The number of CPUs to use for evaluating different feature subsets
        in parallel. -1 means 'all CPUs'.
    pre_dispatch : int, or string (default: '2*n_jobs')
        Controls the number of jobs that get dispatched
        during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
        Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
        None, in which case all the jobs are immediately created and spawned.
            Use this for lightweight and fast-running jobs,
            to avoid delays due to on-demand spawning of the jobs
        An int, giving the exact number of total jobs that are spawned
        A string, giving an expression as a function
            of n_jobs, as in `2*n_jobs`
    clone_estimator : bool (default: True)
        Clones estimator if True; works with the original estimator instance
        if False. Set to False if the estimator doesn't
        implement scikit-learn's set_params and get_params methods.
        In addition, it is required to set cv=0, and n_jobs=1.
    fixed_features : tuple (default: None)
        If not `None`, the feature indices provided as a tuple will be
        regarded as fixed by the feature selector. For example, if
        `fixed_features=(1, 3, 7)`, the 2nd, 4th, and 8th feature are
        guaranteed to be present in the solution. Note that if
        `fixed_features` is not `None`, make sure that the number of
        features to be selected is greater than `len(fixed_features)`.
        In other words, ensure that `k_features > len(fixed_features)`.
        New in mlxtend v. 0.18.0.
    Attributes
    ----------
    k_feature_idx_ : array-like, shape = [n_predictions]
        Feature Indices of the selected feature subsets.
    k_feature_names_ : array-like, shape = [n_predictions]
        Feature names of the selected feature subsets. If pandas
        DataFrames are used in the `fit` method, the feature
        names correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. New in v 0.13.0.
    k_score_ : float
        Cross validation average score of the selected subset.
    subsets_ : dict
        A dictionary of selected feature subsets during the
        sequential selection, where the dictionary keys are
        the lengths k of these feature subsets. The dictionary
        values are dictionaries themselves with the following
        keys: 'feature_idx' (tuple of indices of the feature subset)
              'feature_names' (tuple of feature names of the feat. subset)
              'cv_scores' (list individual cross-validation scores)
              'avg_score' (average cross-validation score)
        Note that if pandas
        DataFrames are used in the `fit` method, the 'feature_names'
        correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. The 'feature_names' is new in v 0.13.0.
    Examples
    -----------
    For usage examples, please see
    TODO: ADD LINK!
    """
    def __init__(self, estimator, k_features=1,
                 verbose=0, scoring=None,
                 cv=5, n_jobs=1,
                 pre_dispatch='2*n_jobs',
                 clone_estimator=True,
                 fixed_features=None):

        super().__init__(estimator=estimator,
                         k_features=k_features,
                         forward=True,
                         floating=False,
                         verbose=verbose,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=n_jobs,
                         pre_dispatch=pre_dispatch,
                         clone_estimator=clone_estimator,
                         fixed_features=fixed_features)

    def fit(self, X, y, custom_feature_names=None, groups=None, fgroups=None, custom_early_stop=None, **fit_params):
        """Perform feature selection and learn model from training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for y.
        custom_feature_names : None or tuple (default: tuple)
            Custom feature names for `self.k_feature_names` and
            `self.subsets_[i]['feature_names']`.
            (new in v 0.13.0)
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fgroups : array-like, shape = [n_features], optional
            Grouping indices for features that should be treated as if they
            where one feature.
        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.
        Returns
        -------
        self : object
        """

        # reset from a potential previous fit run
        self.subsets_ = {}
        self.fitted = False
        self.interrupted_ = False
        self.k_feature_idx_ = None
        self.k_feature_names_ = None
        self.k_score_ = None

        self.fixed_features_ = self.fixed_features
        self.fixed_features_set_ = set()

        if hasattr(X, 'loc'):
            X_ = X.values
            if self.fixed_features is not None:
                self.fixed_features_ = tuple(X.columns.get_loc(c)
                                             if isinstance(c, str) else c
                                             for c in self.fixed_features
                                             )
        else:
            X_ = X

        if self.fixed_features is not None:
            self.fixed_features_set_ = set(self.fixed_features_)

        if (custom_feature_names is not None
                and len(custom_feature_names) != X.shape[1]):
            raise ValueError('If custom_feature_names is not None, '
                             'the number of elements in custom_feature_names '
                             'must equal the number of columns in X.')

        if fgroups is not None and not self.forward:
            raise NotImplementedError("Selection of grouped feature is only "
                                      "implemented with 'forward=True'.")

        if not isinstance(self.k_features, int) and\
                not isinstance(self.k_features, tuple)\
                and not isinstance(self.k_features, str):
            raise AttributeError('k_features must be a positive integer'
                                 ', tuple, or string')

        if (isinstance(self.k_features, int) and (
                self.k_features < 1 or self.k_features > X_.shape[1])):
            raise AttributeError('k_features must be a positive integer'
                                 ' between 1 and X.shape[1], got %s'
                                 % (self.k_features, ))

        if isinstance(self.k_features, tuple):
            if len(self.k_features) != 2:
                raise AttributeError('k_features tuple must consist of 2'
                                     ' elements a min and a max value.')

            if self.k_features[0] not in range(1, X_.shape[1] + 1):
                raise AttributeError('k_features tuple min value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[1] not in range(1, X_.shape[1] + 1):
                raise AttributeError('k_features tuple max value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[0] > self.k_features[1]:
                raise AttributeError('The min k_features value must be smaller'
                                     ' than the max k_features value.')

        if isinstance(self.k_features, tuple) or\
                isinstance(self.k_features, str):

            select_in_range = True

            if isinstance(self.k_features, str):
                if self.k_features not in {'best', 'parsimonious'}:
                    raise AttributeError('If a string argument is provided, '
                                         'it must be "best" or "parsimonious"')
                else:
                    min_k = 1
                    max_k = X_.shape[1]
            else:
                min_k = self.k_features[0]
                max_k = self.k_features[1]

        else:
            select_in_range = False
            k_to_select = self.k_features

        if fgroups is None:
            orig_set = set(range(X_.shape[1]))
            n_features = X_.shape[1]
        else:
            orig_set = set(fgroups)
            n_features = len(orig_set)

        if self.forward and self.fixed_features is not None:
            orig_set = set(range(X_.shape[1])) - self.fixed_features_set_
            n_features = len(orig_set)

        if self.forward:
            if select_in_range:
                k_to_select = max_k

            if self.fixed_features is not None:
                k_idx = self.fixed_features_
                k = len(k_idx)
                k_idx, k_score = _calc_score(self, X_[:, k_idx], y, k_idx,
                                             groups=groups, **fit_params)
                self.subsets_[k] = {
                    'feature_idx': k_idx,
                    'cv_scores': k_score,
                    'avg_score': np.nanmean(k_score)
                }

            else:
                k_idx = ()
                k = 0
        else:
            if select_in_range:
                k_to_select = min_k
            k_idx = tuple(orig_set)
            k = len(k_idx)
            k_idx, k_score = _calc_score(self, X_[:, k_idx], y, k_idx,
                                         groups=groups, **fit_params)
            self.subsets_[k] = {
                'feature_idx': k_idx,
                'cv_scores': k_score,
                'avg_score': np.nanmean(k_score)
            }
        best_subset = None
        k_score = 0

        try:
            while k < k_to_select:  # BM: this was != in the original version
                if k > 0 and custom_early_stop is not None:  # BM: early stopping add on
                    # BM: we temporarily set fitted to true before the end of the loop
                    #   to be able to call self.get_metric_dict().
                    #   This raises an error when self.fitted is False.  
                    self.fitted = True
                    metrics = self.get_metric_dict(confidence_interval=0.95)
                    self.fitted = False
                    stop = custom_early_stop(metrics)
                    if stop:
                        print("Stopping early due to custom early stopping criteria.")
                        break
                prev_subset = set(k_idx) # 16

                if self.forward:
                    k_idx, k_score, cv_scores = self._inclusion(
                        orig_set=orig_set,
                        subset=prev_subset,
                        X=X_,
                        y=y,
                        groups=groups,
                        fgroups=fgroups,
                        **fit_params
                    )
                else:
                    k_idx, k_score, cv_scores = self._exclusion(
                        feature_set=prev_subset,
                        X=X_,
                        y=y,
                        groups=groups,
                        fgroups=fgroups,
                        fixed_feature=self.fixed_features_set_,
                        **fit_params
                    )

                if self.floating:

                    if self.forward:
                        continuation_cond_1 = len(k_idx)
                    else:
                        continuation_cond_1 = n_features - len(k_idx)

                    continuation_cond_2 = True
                    ran_step_1 = True
                    new_feature = None

                    while continuation_cond_1 >= 2 and continuation_cond_2:
                        k_score_c = None

                        if ran_step_1:
                            (new_feature,) = set(k_idx) ^ prev_subset

                        if self.forward:

                            fixed_features_ok = True
                            if self.fixed_features is not None and \
                                    len(self.fixed_features) - len(k_idx) <= 1:
                                fixed_features_ok = False
                            if fixed_features_ok:
                                k_idx_c, k_score_c, cv_scores_c = \
                                    self._exclusion(
                                        feature_set=k_idx,
                                        fixed_feature=(
                                            {new_feature} |
                                            self.fixed_features_set_),
                                        X=X_,
                                        y=y,
                                        groups=groups,
                                        **fit_params
                                    )

                        else:
                            k_idx_c, k_score_c, cv_scores_c = self._inclusion(
                                orig_set=orig_set - {new_feature},
                                subset=set(k_idx),
                                X=X_,
                                y=y,
                                groups=groups,
                                **fit_params
                            )

                        if k_score_c is not None and k_score_c > k_score:

                            if len(k_idx_c) in self.subsets_:
                                cached_score = self.subsets_[len(
                                    k_idx_c)]['avg_score']
                            else:
                                cached_score = None

                            if cached_score is None or \
                                    k_score_c > cached_score:
                                prev_subset = set(k_idx)
                                k_idx, k_score, cv_scores = \
                                    k_idx_c, k_score_c, cv_scores_c
                                continuation_cond_1 = len(k_idx)
                                ran_step_1 = False

                            else:
                                continuation_cond_2 = False

                        else:
                            continuation_cond_2 = False

                k = len(k_idx)
                # floating can lead to multiple same-sized subsets
                if k not in self.subsets_ or (k_score >
                                              self.subsets_[k]['avg_score']):

                    k_idx = tuple(sorted(k_idx))
                    self.subsets_[k] = {
                        'feature_idx': k_idx,
                        'cv_scores': cv_scores,
                        'avg_score': k_score
                    }

                if self.verbose == 1:
                    sys.stderr.write('\rFeatures: %d/%s' % (
                        len(k_idx),
                        k_to_select
                    ))
                    sys.stderr.flush()
                elif self.verbose > 1:
                    sys.stderr.write('\n[%s] Features: %d/%s -- score: %s' % (
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(k_idx),
                        k_to_select,
                        k_score
                    ))

                if self._TESTING_INTERRUPT_MODE:
                    self.subsets_, self.k_feature_names_ = \
                        _get_featurenames(self.subsets_,
                                          self.k_feature_idx_,
                                          custom_feature_names,
                                          X)
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.interrupted_ = True
            sys.stderr.write('\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...')

        if select_in_range:
            max_score = float('-inf')

            max_score = float('-inf')
            for k in self.subsets_:
                if k < min_k or k > max_k:
                    continue
                if self.subsets_[k]['avg_score'] > max_score:
                    max_score = self.subsets_[k]['avg_score']
                    best_subset = k
            k_score = max_score
            k_idx = self.subsets_[best_subset]['feature_idx']

            if self.k_features == 'parsimonious':
                for k in self.subsets_:
                    if k >= best_subset:
                        continue
                    if self.subsets_[k]['avg_score'] >= (
                            max_score - np.std(self.subsets_[k]['cv_scores']) /
                            self.subsets_[k]['cv_scores'].shape[0]):
                        max_score = self.subsets_[k]['avg_score']
                        best_subset = k
                k_score = max_score
                k_idx = self.subsets_[best_subset]['feature_idx']

        self.k_feature_idx_ = k_idx
        self.k_score_ = k_score
        self.fitted = True
        self.subsets_, self.k_feature_names_ = \
            _get_featurenames(self.subsets_,
                              self.k_feature_idx_,
                              custom_feature_names,
                              X)
        return self

    def _inclusion(self, orig_set, subset, X, y, ignore_feature=None,
                   fgroups=None,
                   groups=None, **fit_params):
        all_avg_scores = []
        all_cv_scores = []
        all_subsets = []
        res = (None, None, None)
        if fgroups is None:
            remaining = orig_set - subset
        else:
            remaining = orig_set - set(fgroup for i, fgroup in enumerate(fgroups) if i in subset)
        if remaining:
            features = len(remaining)
            n_jobs = min(self.n_jobs, features)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)
            if fgroups is None:
                work = parallel(delayed(_calc_score)
                                (self, X[:, tuple(subset | {feature})], y,
                                tuple(subset | {feature}),
                                groups=groups, **fit_params)
                                for feature in remaining
                                if feature != ignore_feature)
            else:
                work = parallel(delayed(_calc_score)
                                (self, X[:, tuple(subset | set([i for i, fgroup in enumerate(fgroups) if fgroup==feature]))], y,
                                tuple(subset | set([i for i, fgroup in enumerate(fgroups) if fgroup==feature])), 
                                groups=groups, **fit_params)
                                for feature in remaining
                                if feature != ignore_feature)

            for new_subset, cv_scores in work:
                all_avg_scores.append(np.nanmean(cv_scores))
                all_cv_scores.append(cv_scores)
                all_subsets.append(new_subset)

            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def fit_transform(self, X, y, groups=None, **fit_params):
        """Fit to training data then reduce X to its most important features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.
            New in v 0.13.0: a pandas Series are now also accepted as
            argument for y.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fgroups : array-like, shape = [n_features], optional
            Grouping indices for features that should be treated as if they
            where one feature.
        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.
        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}
        """
        self.fit(X, y, groups=groups, fgroups=None, **fit_params)
        return self.transform(X)

# END OF PART ONE (see comment above)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def sfs_metrics_to_dataframe(sfs, explode_cv_scores=None, fgroups=None, fgroup_names=None):
    """Derive a datafrome from the metrics dict of a SequentialFeatureSelector object.
    Args:
        sfs (SequentialFeatureSelector instance or dict): An instance of the 
            SequentialFeatureSelector or ForwardFeatureGroupSelection class or the 
            thereof derived metrics dict (with .get_metrics_dict()).
        explode_cv_scores (str or None, optional): 
            None: Cross-validation (CV) scores are stored as list in column cv_scores;
            'wide': CV score of each fold are in a separate column;
            'long': Each CV score is stored in a unique row. 
            Defaults to None.
        fgroups (list of int, or None): List of feature group IDs as passed to ForwardFeatureGroupSelection.fit().
        fgroup_names (dict, or None): List of feature group names as group id (key): group name (value) dictionary. 
            Only used when fgroups is given.
    Returns:
        pd.DataFrame: Metrics as dataframe.
    """

    if not isinstance(sfs, dict):
        sfs_metrics_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    else:
        sfs_metrics_df = pd.DataFrame.from_dict(sfs).T

    k_cv_scores = len(sfs_metrics_df.iloc[0]['cv_scores'])
    # when using this for early stopping in ffgs we do not yet have the feature names
    # in the dictionary. therefore we simulate it here
    if "feature_names" not in sfs_metrics_df.columns:
        sfs_metrics_df["feature_names"] = sfs_metrics_df["feature_idx"]

    sfs_metrics_df["feature_idx_new"] = sfs_metrics_df["feature_idx"].copy()
    sfs_metrics_df["feature_names_new"] = sfs_metrics_df["feature_names"].copy()
    for i, (index, df) in enumerate(sfs_metrics_df.iterrows()):
        if index == sfs_metrics_df.index[0]:
            new_idx = set(sfs_metrics_df.loc[index, 'feature_idx'])
            new_feature_names = set(sfs_metrics_df.loc[index, 'feature_names'])
        else:
            last_index = sfs_metrics_df.index[i-1]
            new_idx = set(sfs_metrics_df.loc[index, 'feature_idx']) \
                .difference(
                    set(sfs_metrics_df.loc[last_index, 'feature_idx']
                    ))
            new_feature_names = set(sfs_metrics_df.loc[index, 'feature_names']) \
                .difference(
                    set(sfs_metrics_df.loc[last_index, 'feature_names']))
        sfs_metrics_df.loc[index, "feature_idx_new"] = tuple(new_idx)
        sfs_metrics_df.loc[index, "feature_names_new"] = tuple(new_feature_names)
    if (sfs_metrics_df.feature_idx_new.apply(len) == 1).all():
        sfs_metrics_df.feature_idx_new = sfs_metrics_df.feature_idx_new.apply(lambda x: x[0])
    if (sfs_metrics_df.feature_names_new.apply(len) == 1).all():
        sfs_metrics_df.feature_names_new = sfs_metrics_df.feature_names_new.apply(lambda x: x[0])

    sfs_metrics_df.index.name = "n_features"
    sfs_metrics_df = sfs_metrics_df.reset_index()
    sfs_metrics_df.index = sfs_metrics_df.index + 1
    sfs_metrics_df.index.name = "iter"

    if explode_cv_scores:
        sfs_metrics_df_exploded = sfs_metrics_df.cv_scores.explode().to_frame()
        sfs_metrics_df_exploded["fold"] = list(range(len(sfs_metrics_df.cv_scores.iloc[0]))) * sfs_metrics_df.shape[0]
        if explode_cv_scores == "wide":
            sfs_metrics_df_exploded = sfs_metrics_df_exploded.reset_index().rename({"iter":"rank"}, axis=1).pivot(index="rank", columns="fold")
            sfs_metrics_df_exploded.columns = [f"cv_score_{i}" for i in range(sfs_metrics_df_exploded.shape[1])]
            sfs_metrics_df = sfs_metrics_df.join(sfs_metrics_df_exploded)
        elif explode_cv_scores == "long":
            sfs_metrics_df = sfs_metrics_df.drop("cv_scores", axis=1) \
                .join(sfs_metrics_df_exploded.rename({"cv_scores": "cv_score"}, axis=1)) \
                .reset_index()
            sfs_metrics_df = pd.concat([sfs_metrics_df[["iter", "fold"]],
                                        sfs_metrics_df.drop(["iter", "fold"], axis=1)], axis=1)
            sfs_metrics_df.index = sfs_metrics_df.iter.astype(str) + "-" + sfs_metrics_df.fold.astype(str)
    
    if fgroups is not None:
        gids = np.unique(fgroups)
        gid_fidx_map = {gid: tuple(np.where(fgroups == gid)[0]) for gid in gids}
        sfs_metrics_df["feature_groups"] = None
        sfs_metrics_df["feature_group_new"] = None
        for gid, fidx in gid_fidx_map.items():
            row_idx_group = sfs_metrics_df["feature_idx_new"].apply(lambda x: set(x) == set(fidx))
            if sfs_metrics_df.index.name == 'iter' and (explode_cv_scores == 'wide' or explode_cv_scores is None):
                assert row_idx_group.sum() == (sfs_metrics_df.index == 1).sum()
            elif sfs_metrics_df.index.name != 'iter' and (explode_cv_scores == 'wide' or explode_cv_scores is None):
                assert row_idx_group.sum() == (sfs_metrics_df['iter'] == 1).sum()
            else:
                assert row_idx_group.sum() == k_cv_scores
            sfs_metrics_df.loc[row_idx_group, "feature_group_new"] = gid

        growing_col_element = []
        col_values = []
        for i, row in sfs_metrics_df.iterrows():
            growing_col_element.append(row["feature_group_new"])
            col_values.append(tuple(sorted(np.unique(growing_col_element))))
        sfs_metrics_df["feature_groups"] = col_values

        if fgroup_names is not None:
            sfs_metrics_df["feature_group_names"] = None
            sfs_metrics_df["feature_group_name_new"] = None

            sfs_metrics_df["feature_group_name_new"] = sfs_metrics_df["feature_group_new"].map(fgroup_names)
            growing_col_element = []
            col_values = []
            for i, row in sfs_metrics_df.iterrows():
                growing_col_element.append(row["feature_group_name_new"])
                col_values.append(tuple(sorted(np.unique(growing_col_element))))
            sfs_metrics_df["feature_group_names"] = col_values

    return sfs_metrics_df

def get_number_of_ffs_iterations(n):
    """Calculate the number of iterations for a forward feature selection given n features.
    Args:
        n (int): Number of features or feature groups.
    Returns:
        int: Number of iterations.
    """
    if n == 1:
        return 0
    else:
        return n + get_number_of_ffs_iterations(n-1)
