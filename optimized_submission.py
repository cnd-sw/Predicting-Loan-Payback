import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from itertools import combinations
from scipy.optimize import minimize

warnings.simplefilter('ignore')

# Configuration
# Adjust these paths to match your environment
TRAIN_PATH = 'playground-series-s5e11/train.csv'
TEST_PATH = 'playground-series-s5e11/test.csv'
ORIG_PATH = 'loan_dataset_20000.csv' # Update if you have the original dataset
SUBMISSION_PATH = 'submission.csv'
TARGET = 'loan_paid_back'
N_SPLITS = 10
SEED = 42

# Check if running on Kaggle
if os.path.exists('/kaggle/input'):
    TRAIN_PATH = '/kaggle/input/playground-series-s5e11/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s5e11/test.csv'
    ORIG_PATH = '/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv'

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder with smoothing and CV handling.
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
        temp_df = X.copy()
        temp_df['target'] = y
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                if col in self.mappings_ and agg_func in self.mappings_[col]:
                    map_series = self.mappings_[col][agg_func]
                    X_transformed[new_col_name] = X[col].map(map_series)
                    X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
                else:
                    X_transformed[new_col_name] = self.global_stats_[agg_func]
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train
            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

def feature_engineering(train, test, orig=None):
    print("Starting Feature Engineering...")
    
    # Base columns
    CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
    BASE = [col for col in train.columns if col not in ['id', TARGET]]
    
    # Combined df for consistent processing
    train['is_train'] = 1
    test['is_train'] = 0
    test[TARGET] = np.nan
    
    if orig is not None:
        orig['is_train'] = 1
        combined = pd.concat([train, test, orig], axis=0, ignore_index=True)
    else:
        combined = pd.concat([train, test], axis=0, ignore_index=True)
        
    # 1. Log Transforms (New)
    print("Creating Log Features...")
    for col in ['annual_income', 'loan_amount']:
        combined[f'log_{col}'] = np.log1p(combined[col])
        
    # 2. Ratio Features (New)
    print("Creating Ratio Features...")
    combined['loan_to_income'] = combined['loan_amount'] / (combined['annual_income'] + 1)
    combined['monthly_debt_est'] = (combined['annual_income'] / 12) * combined['debt_to_income_ratio']
    combined['total_interest_est'] = combined['loan_amount'] * (combined['interest_rate'] / 100)
    
    # 3. Digit Features
    print("Creating Digit Features...")
    cols_to_digitize = {
        'debt_to_income_ratio': 1000,
        'credit_score': 'direct',
        'interest_rate': 100,
    }
    
    DIGIT = []
    for col, multiplier in cols_to_digitize.items():
        temp_col_name = f'{col}_TEMP_INT'
        if multiplier == 'direct':
            combined[temp_col_name] = combined[col]
        else:
            combined[temp_col_name] = (combined[col] * multiplier).round(0).astype(int)
        
        temp_str = combined[temp_col_name].astype(str)
        if col == 'credit_score': max_len = 3
        elif col == 'debt_to_income_ratio': max_len = 3
        elif col == 'interest_rate': max_len = 4
        
        temp_str_padded = temp_str.str.zfill(max_len)
        for i in range(max_len):
            new_col_name = f'{col}_DIGIT_{i+1}'
            DIGIT.append(new_col_name)
            combined[new_col_name] = temp_str_padded.str[i].replace('.', '0').replace('-', '0').astype(int) # Handle potential formatting issues
            
        combined.drop(columns=[temp_col_name], inplace=True)

    # 4. Round Features
    print("Creating Round Features...")
    ROUND = []
    rounding_levels = {'1s': 0, '10s': -1, '100s': -2, '1000s': -3}
    for col in ['annual_income', 'loan_amount']:
        for suffix, level in rounding_levels.items():
            new_col_name = f'{col}_ROUND_{suffix}'
            ROUND.append(new_col_name)
            combined[new_col_name] = combined[col].round(level).fillna(0).astype(int)

    # 5. Interaction Features
    print("Creating Interaction Features...")
    INTER = []
    # Limit interactions to categorical/discrete features to avoid explosion
    INTER_BASE = CATS + ['credit_score'] 
    for col1, col2 in combinations(INTER_BASE, 2):
        new_col_name = f'{col1}_{col2}'
        INTER.append(new_col_name)
        combined[new_col_name] = combined[col1].astype(str) + '_' + combined[col2].astype(str)

    # Split back
    train_processed = combined[combined['is_train'] == 1].copy()
    test_processed = combined[combined['is_train'] == 0].copy()
    
    # Remove orig data from train if it was added for feature engineering but we want to train only on train set or both?
    # Usually we want to train on train + orig if orig is good.
    # The user's code trained on train only but used orig for features.
    # Let's stick to returning train (which might include orig rows now) and test.
    # But wait, we need to separate the original 'train' rows from 'orig' rows if we want to validate properly?
    # For simplicity, let's assume we use all available training data.
    
    if orig is not None:
        # If we want to distinguish, we can check indices, but here we just return the full set
        pass
        
    train_processed.drop(columns=['is_train'], inplace=True)
    test_processed.drop(columns=['is_train', TARGET], inplace=True)
    
    # Update feature lists
    NEW_NUMERICS = ['log_annual_income', 'log_loan_amount', 'loan_to_income', 'monthly_debt_est', 'total_interest_est']
    FEATURES = BASE + NEW_NUMERICS + INTER + ROUND + DIGIT
    
    # Remove duplicates from FEATURES
    FEATURES = list(set(FEATURES))
    
    # Ensure all features exist
    FEATURES = [f for f in FEATURES if f in train_processed.columns]
    
    print(f"Total Features: {len(FEATURES)}")
    
    return train_processed, test_processed, FEATURES, CATS, INTER, ROUND, DIGIT

def train_models(train, test, FEATURES, CATS, INTER, ROUND, DIGIT):
    X = train[FEATURES]
    y = train[TARGET]
    X_test_final = test[FEATURES].copy()
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    oof_preds_xgb = np.zeros(len(X))
    test_preds_xgb = np.zeros(len(test))
    
    oof_preds_lgb = np.zeros(len(X))
    test_preds_lgb = np.zeros(len(test))
    
    oof_preds_cat = np.zeros(len(X))
    test_preds_cat = np.zeros(len(test))
    
    # Enhanced Parameters
    xgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.005, # Lower LR
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'n_jobs': -1,
        'random_state': SEED,
        'tree_method': 'hist',
        'early_stopping_rounds': 200,
        'eval_metric': 'auc',
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    lgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.005,
        'max_depth': 10,
        'num_leaves': 64,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'n_jobs': -1,
        'random_state': SEED,
        'metric': 'auc',
        'verbosity': -1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    cat_params = {
        'iterations': 3000,
        'learning_rate': 0.005,
        'depth': 8,
        'l2_leaf_reg': 5,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 200,
        'task_type': 'CPU'
    }

    print("Starting Training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f'--- Fold {fold}/{N_SPLITS} ---')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_test_fold = X_test_final.copy()
        
        # Target Encoding inside fold
        # Encode Interactions
        TE_INTER = TargetEncoder(cols_to_encode=INTER, cv=5, smooth='auto', aggs=['mean'], drop_original=True)
        X_train = TE_INTER.fit_transform(X_train, y_train)
        X_val = TE_INTER.transform(X_val)
        X_test_fold = TE_INTER.transform(X_test_fold)
        
        # Encode Base + Round + Digit
        # We need to be careful not to encode columns that don't exist or are already encoded
        # Filter cols that are in X_train
        cols_to_encode_base = ['debt_to_income_ratio', 'credit_score'] + ROUND + DIGIT
        cols_to_encode_base = [c for c in cols_to_encode_base if c in X_train.columns]
        
        TE_BASE = TargetEncoder(cols_to_encode=cols_to_encode_base, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
        X_train = TE_BASE.fit_transform(X_train, y_train)
        X_val = TE_BASE.transform(X_val)
        X_test_fold = TE_BASE.transform(X_test_fold)
        
        # Factorize Categoricals
        for c in CATS:
            if c in X_train.columns:
                combined = pd.concat([X_train[c], X_val[c], X_test_fold[c]])
                combined_codes, _ = combined.factorize()
                X_train[c] = combined_codes[:len(X_train)]
                X_val[c] = combined_codes[len(X_train):len(X_train)+len(X_val)]
                X_test_fold[c] = combined_codes[len(X_train)+len(X_val):]
                
                X_train[c] = X_train[c].astype('category')
                X_val[c] = X_val[c].astype('category')
                X_test_fold[c] = X_test_fold[c].astype('category')

        # XGBoost
        xgb = XGBClassifier(**xgb_params, enable_categorical=True)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pred_xgb = xgb.predict_proba(X_val)[:, 1]
        oof_preds_xgb[val_idx] = val_pred_xgb
        test_preds_xgb += xgb.predict_proba(X_test_fold)[:, 1] / N_SPLITS
        print(f"XGB AUC: {roc_auc_score(y_val, val_pred_xgb):.5f}")
        
        # LightGBM
        lgb = LGBMClassifier(**lgb_params)
        lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        val_pred_lgb = lgb.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_idx] = val_pred_lgb
        test_preds_lgb += lgb.predict_proba(X_test_fold)[:, 1] / N_SPLITS
        print(f"LGB AUC: {roc_auc_score(y_val, val_pred_lgb):.5f}")
        
        # CatBoost
        # CatBoost needs original categorical columns or indices. 
        # Since we factorized them, we can pass them as integers or declare them as cat features.
        # However, we already converted them to 'category' dtype which CatBoost might not like if we pass cat_features indices.
        # Let's pass the column names.
        cat_features_indices = [c for c in CATS if c in X_train.columns]
        
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features_indices)
        val_pred_cat = cat.predict_proba(X_val)[:, 1]
        oof_preds_cat[val_idx] = val_pred_cat
        test_preds_cat += cat.predict_proba(X_test_fold)[:, 1] / N_SPLITS
        print(f"CAT AUC: {roc_auc_score(y_val, val_pred_cat):.5f}")
        
    return oof_preds_xgb, test_preds_xgb, oof_preds_lgb, test_preds_lgb, oof_preds_cat, test_preds_cat, y

def optimize_ensemble_weights(oof_preds_list, y_true):
    """
    Find optimal weights for the ensemble using SLSQP.
    """
    print("Optimizing ensemble weights...")
    
    def loss_func(weights):
        final_pred = np.zeros_like(oof_preds_list[0])
        for i, pred in enumerate(oof_preds_list):
            final_pred += weights[i] * pred
        return -roc_auc_score(y_true, final_pred)
    
    starting_values = [1/len(oof_preds_list)] * len(oof_preds_list)
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(oof_preds_list)
    
    res = minimize(loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=constraints)
    
    print(f"Optimal Weights: {res.x}")
    print(f"Optimized AUC: {-res.fun:.5f}")
    return res.x

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    if os.path.exists(ORIG_PATH):
        print(f"Found original dataset at {ORIG_PATH}")
        orig = pd.read_csv(ORIG_PATH)
    else:
        print("Original dataset not found. Proceeding without it.")
        orig = None
        
    # 2. Feature Engineering
    train, test, FEATURES, CATS, INTER, ROUND, DIGIT = feature_engineering(train, test, orig)
    
    # 3. Train Models
    oof_xgb, pred_xgb, oof_lgb, pred_lgb, oof_cat, pred_cat, y = train_models(train, test, FEATURES, CATS, INTER, ROUND, DIGIT)
    
    # 4. Ensemble Optimization
    weights = optimize_ensemble_weights([oof_xgb, oof_lgb, oof_cat], y)
    
    final_preds = (pred_xgb * weights[0] + pred_lgb * weights[1] + pred_cat * weights[2])
    
    # 5. Submission
    submission = pd.DataFrame({'id': test['id'], TARGET: final_preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
