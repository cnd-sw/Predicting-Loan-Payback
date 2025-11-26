# PG S5E11 - Bartz - [CV 0.92563 LB 0.92642]

import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)

train.head(3)

TARGET = 'loan_paid_back'
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
BASE = [col for col in train.columns if col not in ['id', TARGET]]


DIGIT = []

cols_to_digitize = {
    'debt_to_income_ratio': 1000,
    'credit_score': 'direct',
    'interest_rate': 100,
}

for col, multiplier in cols_to_digitize.items():
    temp_col_name = f'{col}_TEMP_INT'
    
    for df in [train, test, orig]:
        if multiplier == 'direct':
            df[temp_col_name] = df[col]
        else:
            df[temp_col_name] = (df[col] * multiplier).round(0).astype(int)

        temp_str = df[temp_col_name].astype(str)
        
        if col == 'credit_score':
            max_len = 3
        elif col == 'debt_to_income_ratio':
            max_len = 3 # (0.627 * 1000 = 627)
        elif col == 'interest_rate':
            max_len = 4 # (20.99 * 100 = 2099)
        temp_str_padded = temp_str.str.zfill(max_len)
        for i in range(max_len):
            new_col_name = f'{col}_DIGIT_{i+1}'
            
            if df is train: 
                DIGIT.append(new_col_name)
            df[new_col_name] = temp_str_padded.str[i].astype(int)
            
    for df in [train, test, orig]:
        df.drop(columns=[temp_col_name], inplace=True)


print(f'{len(DIGIT)} DIGIT Features created.')

ROUND = []

rounding_levels = {
    '1s': 0,   
    '10s': -1,
    '100s': -2,
    '1000s': -3,
}

for col in ['annual_income', 'loan_amount']:
    for suffix, level in rounding_levels.items():
        new_col_name = f'{col}_ROUND_{suffix}'
        ROUND.append(new_col_name)
        
        for df in [train, test, orig]:
            df[new_col_name] = df[col].round(level).astype(int)

print(f'{len(ROUND)} ROUND Features created.')

from itertools import combinations

INTER = []

for col1, col2 in combinations(BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test, orig]:
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
        
print(f'{len(INTER)} Features.')


ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(len(ORIG), 'Orig Features Created!!')

train[ORIG] = train[ORIG].fillna(orig[TARGET].mean())
test[ORIG] = test[ORIG].fillna(orig[TARGET].mean())


FEATURES = BASE + ORIG + INTER + ROUND + DIGIT
print(len(FEATURES), 'Features.')

X = train[FEATURES]
y = train[TARGET]

from sklearn.model_selection import StratifiedKFold, KFold

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

from bartz.BART import gbart
from sklearn.metrics import roc_auc_score


from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    cols_to_encode : list of str
        List of column names to be target encoded.

    aggs : list of str, default=['mean']
        List of aggregation functions to apply. Any function accepted by
        pandas' `.agg()` method is supported, such as:
        'mean', 'std', 'var', 'min', 'max', 'skew', 'nunique', 
        'count', 'sum', 'median'.
        Smoothing is applied only to the 'mean' aggregation.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.

    smooth : float or 'auto', default='auto'
        The smoothing parameter `m`. A larger value puts more weight on the 
        global mean. If 'auto', an empirical Bayes estimate is used.
        
    drop_original : bool, default=False
        If True, the original columns to be encoded are dropped.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """
        Learn mappings from the entire dataset.
        These mappings are used for the transform method on validation/test data.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # Learn global statistics for each aggregation
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # Learn category-specific mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        
        return self

    def transform(self, X):
        """
        Apply learned mappings to the data.
        Unseen categories are filled with global statistics.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get global mappings for transform method
        self.fit(X, y)

        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # --- Calculate mappings only on the training part of the fold ---
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    
                    # Calculate global stat for this fold
                    fold_global_stat = y_train.agg(agg_func)
                    
                    # Calculate category stats for this fold
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # --- Apply smoothing only for 'mean' aggregation ---
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        
                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance between groups
                        
                        # Apply smoothing formula
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    # Store encoded values for the validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

params = {'maxdepth': 6,
          'ntree': 400,
          'k': 5,
          'sigdf': 3,
          'sigquant': 0.9}

BASE_TE_COL = ['debt_to_income_ratio', 'credit_score'] + ROUND + DIGIT


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test[FEATURES].copy()

    TE = TargetEncoder(cols_to_encode=INTER, cv=5, smooth='auto', aggs=['mean'], drop_original=True)
    X_train = TE.fit_transform(X_train, y_train)
    X_val = TE.transform(X_val)
    X_test = TE.transform(X_test)

    TE = TargetEncoder(cols_to_encode=BASE_TE_COL, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
    X_train = TE.fit_transform(X_train, y_train)
    X_val = TE.transform(X_val)
    X_test = TE.transform(X_test)

    for c in CATS:
        combined = pd.concat([X_train[c], X_val[c], X_test[c]])
        combined, _ = combined.factorize()
        X_train[c] = combined[:len(X_train)]
        X_val[c] = combined[len(X_train):len(X_train) + len(X_val)]
        X_test[c] = combined[len(X_train) + len(X_val):]

    model = gbart(
        X_train.to_numpy().T,
        y_train.to_numpy(),
        **params,
        ndpost=1000,
        nskip=200,
        keepevery=2
    )

    val_preds = model.predict(X_val.to_numpy().T).mean(axis=0)
    oof_preds[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.6f}')
    test_preds += model.predict(X_test.to_numpy().T).mean(axis=0) / N_SPLITS

overall_auc = roc_auc_score(y, oof_preds)
print(f'====================')
print(f'Overall OOF AUC: {overall_auc:.4f}')
print(f'====================')


pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv(f'oof_bartz_cv_{overall_auc}.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv(f'test_bartz_cv_{overall_auc}.csv', index=False)