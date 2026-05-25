import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.utils.class_weight import compute_sample_weight
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore')

# 1. DATA LOADING & ROBUST CLEANING
df = pd.read_csv('13Final_Thesis_Dataset.csv').dropna(subset=['worthwhileness_rating'])

# 1.3 City noise reduction
threshold = 75 
city_counts = df['city'].value_counts()
df['city'] = df['city'].apply(lambda x: 'Other_Small_City' if city_counts[x] < threshold else x)
df['Purpose_Leisure'] = df['purp_Leisure_Hobby'].fillna(0)

# 2. FEATURE PREP
numerical_features = [
    'EFroutes_traffic_infrastructure', 'EFability_to_do_what_i_wanted', 
    'EFcomfort_pleasure', 'activity_count', 'education_level_numeric',
    'did_you_have_to_arrive', 'number_people_household',
    'Purpose_Leisure', 'leg_duration', 'act_Browsing', 'act_Accompanying', 'leg_distance'
] 

categorical_features = [
    'transport_category', 'gender', 'age_range', 'city', 
    'marital_status_clean', 'weather_group', 
]

X = df[numerical_features + categorical_features].copy()
y = df['worthwhileness_rating'].values

# Ensure Categorical Dtypes for the Boosters
for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str).astype('category')

def get_1off_accuracy(y_true, y_pred):
    rounded = np.clip(np.round(y_pred), 1, 5)
    return np.mean(np.abs(y_true - rounded) <= 1)

# 3. 80/20 SUPERVISOR SPLIT
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. MODELS
models = {
    "Baseline: Dummy": DummyRegressor(strategy='mean'),
    "Baseline: Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=20, max_features=0.3, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(loss_function='Poisson', iterations=1000, learning_rate=0.04, depth=7, l2_leaf_reg=20, subsample=0.8, random_seed=42, verbose=False, bootstrap_type='Bernoulli', cat_features=categorical_features),
    "XGBoost": xgb.XGBRegressor(objective='count:poisson', tree_method="hist", enable_categorical=True, n_estimators=1000, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_weight=30, reg_lambda=10, reg_alpha=1, random_state=42)
}

# 5. 5-FOLD CV WITH MODEL-SPECIFIC IMPUTATION
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev, y_dev)):
    print(f"Processing CV Fold {fold+1}/5")
    X_tr_cv, X_va_cv = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
    y_tr_cv, y_va_cv = y_dev[train_idx], y_dev[val_idx]
    w = compute_sample_weight(class_weight='balanced', y=y_tr_cv)

    for name, model in models.items():
        # Model specific preprocessing
        if any(x in name for x in ["Ridge", "Random Forest", "Dummy"]):
            # Path A: Fill NaNs (using training median to avoid leakage) + Target Encode
            X_tr_proc = X_tr_cv.fillna(X_tr_cv.median(numeric_only=True))
            X_va_proc = X_va_cv.fillna(X_tr_cv.median(numeric_only=True))
            
            te = TargetEncoder(cols=categorical_features)
            X_tr, X_va = te.fit_transform(X_tr_proc, y_tr_cv), te.transform(X_va_proc)
        else:
            # Path B: Give the Boosters the raw data (Native NaN handling)
            X_tr, X_va = X_tr_cv, X_va_cv
        
        model.fit(X_tr, y_tr_cv, sample_weight=w)
        preds = model.predict(X_va)
        cv_results.append({
            "Model": name, 
            "MAE": mean_absolute_error(y_va_cv, preds), 
            "R2": r2_score(y_va_cv, preds), 
            "1-Off": get_1off_accuracy(y_va_cv, preds)
        })

# 6. FINAL EVALUATION ON 20% TEST SET
print("\nFinal Retraining on 20% Hold-out Set...")
final_results = []
dev_w = compute_sample_weight(class_weight='balanced', y=y_dev)

for name, model in models.items():
    if any(x in name for x in ["Ridge", "Random Forest", "Dummy"]):
        # Baselines get Imputation
        X_dev_p = X_dev.fillna(X_dev.median(numeric_only=True))
        X_test_p = X_test.fillna(X_dev.median(numeric_only=True))
        
        te_f = TargetEncoder(cols=categorical_features)
        X_tr_f, X_te_f = te_f.fit_transform(X_dev_p, y_dev), te_f.transform(X_test_p)
    else:
        # Boosters get Raw Data
        X_tr_f, X_te_f = X_dev, X_test
    
    model.fit(X_tr_f, y_dev, sample_weight=dev_w)
    t_preds = model.predict(X_te_f)
    final_results.append({
        "Model": name, 
        "MAE": mean_absolute_error(y_test, t_preds), 
        "R2": r2_score(y_test, t_preds), 
        "1-Off": get_1off_accuracy(y_test, t_preds)
    })

# 7. FINAL TABLES & VISUALIZATION
cv_df = pd.DataFrame(cv_results)
final_df = pd.DataFrame(final_results).set_index("Model")

# TABLE 1: 5-FOLD CV
stats = cv_df.groupby("Model").agg(['mean', 'std'])
cv_table = pd.DataFrame(index=stats.index)
cv_table['MAE (Mean ± SD)'] = stats['MAE']['mean'].map('{:.4f}'.format) + " ± " + stats['MAE']['std'].map('{:.4f}'.format)
cv_table['R2 (Mean ± SD)'] = stats['R2']['mean'].map('{:.4f}'.format) + " ± " + stats['R2']['std'].map('{:.4f}'.format)
cv_table['1-Off Accuracy (Mean ± SD)'] = (stats['1-Off']['mean']*100).map('{:.2f}%'.format) + " ± " + (stats['1-Off']['std']*100).map('{:.2f}%'.format)

# TABLE 2: FINAL HOLD-OUT
test_table = final_df.copy()
for col in ['MAE', 'R2']: test_table[col] = test_table[col].map('{:.4f}'.format)
test_table['1-Off'] = (test_table['1-Off']*100).map('{:.2f}%'.format)

print("\n" + "="*80)
print(f"{'TABLE 1: 5-FOLD CV RESULTS (DEVELOPMENT SET)':^80}")
print("="*80)
print(cv_table)
print("\n" + "="*80)
print(f"{'TABLE 2: FINAL HOLD-OUT RESULTS (20% SEPARATE TEST SET)':^80}")
print("="*80)
print(test_table)

# THE 3-PANEL BOXPLOT
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
metrics = ["MAE", "R2", "1-Off"]
titles = ["Mean Absolute Error", "R-Squared Score", "1-Off Accuracy"]

for i, m in enumerate(metrics):
    sns.boxplot(data=cv_df, x="Model", y=m, ax=axes[i], palette="viridis")
    axes[i].set_title(titles[i], fontweight='bold', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

