import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from itertools import combinations

warnings.simplefilter('ignore')

# Configuration
TRAIN_PATH = 'playground-series-s5e11/train.csv'
TEST_PATH = 'playground-series-s5e11/test.csv'
ORIG_PATH = 'loan-prediction-dataset-2025/loan_dataset_20000.csv' # Optional
SUBMISSION_PATH = 'submission.csv'
TARGET = 'loan_paid_back'
N_SPLITS = 10 # Increased folds for better stability
SEED = 42

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    if os.path.exists(ORIG_PATH):
        print(f"Found original dataset at {ORIG_PATH}")
        orig = pd.read_csv(ORIG_PATH)
    else:
        print("Original dataset not found. Proceeding without it.")
        orig = None
        
    return train, test, orig

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder from existing solution.
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
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
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
    
    # 1. Digit Features
    print("Creating Digit Features...")
    cols_to_digitize = {
        'debt_to_income_ratio': 1000,
        'credit_score': 'direct',
        'interest_rate': 100,
    }
    
    DIGIT = []
    dfs = [train, test]
    if orig is not None: dfs.append(orig)
    
    for col, multiplier in cols_to_digitize.items():
        temp_col_name = f'{col}_TEMP_INT'
        for df in dfs:
            if multiplier == 'direct':
                df[temp_col_name] = df[col]
            else:
                df[temp_col_name] = (df[col] * multiplier).round(0).astype(int)
            
            temp_str = df[temp_col_name].astype(str)
            if col == 'credit_score': max_len = 3
            elif col == 'debt_to_income_ratio': max_len = 3
            elif col == 'interest_rate': max_len = 4
            
            temp_str_padded = temp_str.str.zfill(max_len)
            for i in range(max_len):
                new_col_name = f'{col}_DIGIT_{i+1}'
                if df is train: DIGIT.append(new_col_name)
                df[new_col_name] = temp_str_padded.str[i].astype(int)
        
        for df in dfs:
            df.drop(columns=[temp_col_name], inplace=True)

    # 2. Round Features
    print("Creating Round Features...")
    ROUND = []
    rounding_levels = {'1s': 0, '10s': -1, '100s': -2, '1000s': -3}
    for col in ['annual_income', 'loan_amount']:
        for suffix, level in rounding_levels.items():
            new_col_name = f'{col}_ROUND_{suffix}'
            ROUND.append(new_col_name)
            for df in dfs:
                df[new_col_name] = df[col].round(level).astype(int)

    # 3. Interaction Features
    print("Creating Interaction Features...")
    INTER = []
    for col1, col2 in combinations(BASE, 2):
        new_col_name = f'{col1}_{col2}'
        INTER.append(new_col_name)
        for df in dfs:
            df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)

    # 4. Original Data Features (if available)
    ORIG_FEATS = []
    if orig is not None:
        print("Creating Original Data Features...")
        for col in BASE:
            # Mean
            mean_map = orig.groupby(col)[TARGET].mean()
            new_mean_col_name = f"orig_mean_{col}"
            mean_map.name = new_mean_col_name
            train = train.merge(mean_map, on=col, how='left')
            test = test.merge(mean_map, on=col, how='left')
            ORIG_FEATS.append(new_mean_col_name)
            
            # Count
            new_count_col_name = f"orig_count_{col}"
            count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
            train = train.merge(count_map, on=col, how='left')
            test = test.merge(count_map, on=col, how='left')
            ORIG_FEATS.append(new_count_col_name)
            
        train[ORIG_FEATS] = train[ORIG_FEATS].fillna(orig[TARGET].mean())
        test[ORIG_FEATS] = test[ORIG_FEATS].fillna(orig[TARGET].mean())

    # Feature Lists
    FEATURES = BASE + ORIG_FEATS + INTER + ROUND + DIGIT
    print(f"Total Features: {len(FEATURES)}")
    
    return train, test, FEATURES, CATS, INTER, ROUND, DIGIT

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
    
    # Model Parameters (Tuned for general performance)
    xgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': SEED,
        'tree_method': 'hist',
        'early_stopping_rounds': 100,
        'eval_metric': 'auc'
    }
    
    lgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 8,
        'num_leaves': 32,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': SEED,
        'metric': 'auc',
        'verbosity': -1
    }
    
    cat_params = {
        'iterations': 2000,
        'learning_rate': 0.01,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 100
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
        BASE_TE_COL = ['debt_to_income_ratio', 'credit_score'] + ROUND + DIGIT
        TE_BASE = TargetEncoder(cols_to_encode=BASE_TE_COL, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
        X_train = TE_BASE.fit_transform(X_train, y_train)
        X_val = TE_BASE.transform(X_val)
        X_test_fold = TE_BASE.transform(X_test_fold)
        
        # Factorize Categoricals for XGB/LGB (CatBoost handles them natively usually, but for consistency we'll use numeric)
        # Actually, let's keep them as category type for LGBM and CatBoost if possible, but factorizing is safer for all.
        for c in CATS:
            combined = pd.concat([X_train[c], X_val[c], X_test_fold[c]])
            combined_codes, _ = combined.factorize()
            X_train[c] = combined_codes[:len(X_train)]
            X_val[c] = combined_codes[len(X_train):len(X_train)+len(X_val)]
            X_test_fold[c] = combined_codes[len(X_train)+len(X_val):]
            
            # Ensure categorical type for LGBM
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
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=CATS)
        val_pred_cat = cat.predict_proba(X_val)[:, 1]
        oof_preds_cat[val_idx] = val_pred_cat
        test_preds_cat += cat.predict_proba(X_test_fold)[:, 1] / N_SPLITS
        print(f"CAT AUC: {roc_auc_score(y_val, val_pred_cat):.5f}")
        
    return oof_preds_xgb, test_preds_xgb, oof_preds_lgb, test_preds_lgb, oof_preds_cat, test_preds_cat, y

def main():
    train, test, orig = load_data()
    train, test, FEATURES, CATS, INTER, ROUND, DIGIT = feature_engineering(train, test, orig)
    
    oof_xgb, pred_xgb, oof_lgb, pred_lgb, oof_cat, pred_cat, y = train_models(train, test, FEATURES, CATS, INTER, ROUND, DIGIT)
    
    # Ensemble Weights (Simple Average or Weighted)
    # Let's check individual performances
    auc_xgb = roc_auc_score(y, oof_xgb)
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_cat = roc_auc_score(y, oof_cat)
    
    print(f"\nOverall XGB AUC: {auc_xgb:.5f}")
    print(f"Overall LGB AUC: {auc_lgb:.5f}")
    print(f"Overall CAT AUC: {auc_cat:.5f}")
    
    # Simple Average
    ensemble_oof = (oof_xgb + oof_lgb + oof_cat) / 3
    ensemble_auc = roc_auc_score(y, ensemble_oof)
    print(f"Ensemble (Average) AUC: {ensemble_auc:.5f}")
    
    # Weighted Average (give more weight to better models)
    total_auc = auc_xgb + auc_lgb + auc_cat
    w_xgb = auc_xgb / total_auc
    w_lgb = auc_lgb / total_auc
    w_cat = auc_cat / total_auc
    
    ensemble_oof_w = (oof_xgb * w_xgb + oof_lgb * w_lgb + oof_cat * w_cat)
    ensemble_auc_w = roc_auc_score(y, ensemble_oof_w)
    print(f"Ensemble (Weighted) AUC: {ensemble_auc_w:.5f}")
    
    # Final Predictions
    final_preds = (pred_xgb + pred_lgb + pred_cat) / 3
    
    # Save Submission
    submission = pd.DataFrame({'id': test['id'], TARGET: final_preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
