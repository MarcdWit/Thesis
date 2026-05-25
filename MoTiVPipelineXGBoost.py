import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight


# 1. DATA LOADING & PREP
print("Loading")
df = pd.read_csv('13Final_Thesis_Dataset.csv')
df = df.dropna(subset=['worthwhileness_rating'])

# City pre-processing (N=75 Threshold for stability)
threshold = 75 
city_counts = df['city'].value_counts()
small_cities = city_counts[city_counts < threshold].index
df['city'] = df['city'].apply(lambda x: 'Other_Small_City' if x in small_cities else x)

# Feature Engineering
df['Purpose_Leisure'] = df['purp_Leisure_Hobby'].fillna(0)

# Included features
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
y = df['worthwhileness_rating']

for col in categorical_features:
    X[col] = X[col].astype('category')

# 2. SPLIT & WEIGHTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 3. TRAINING 

xgb_model = xgb.XGBRegressor(
    objective='count:poisson',
    tree_method="hist",
    enable_categorical=True,
    n_estimators=1000,        
    learning_rate=0.04,
    max_depth=7,              
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=30,      
    reg_lambda=10,           
    reg_alpha=1,              
    random_state=42
)

xgb_model.fit(X_train, y_train, sample_weight=weights)

# 4. PERFORMANCE EVALUATION
train_preds = xgb_model.predict(X_train)
test_preds = xgb_model.predict(X_test)

# Metrics: Standard
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

# Metrics: 1-Off Accuracy 
# We round the continuous Poisson prediction to the nearest integer 1-5
y_train_rounded = np.clip(np.round(train_preds), 1, 5)
y_test_rounded = np.clip(np.round(test_preds), 1, 5)

train_1off = np.mean(np.abs(y_train - y_train_rounded) <= 1)
test_1off = np.mean(np.abs(y_test - y_test_rounded) <= 1)

print("\n" + "="*70)
print(f"{'Metric':<25} | {'Training Set':<15} | {'Test Set':<15}")
print("-" * 70)
print(f"{'R-Squared (Accuracy)':<25} | {train_r2:<15.4f} | {test_r2:<15.4f}")
print(f"{'Mean Absolute Error':<25} | {train_mae:<15.4f} | {test_mae:<15.4f}")
print(f"{'1-Off Accuracy (%)':<25} | {train_1off:<15.2%} | {test_1off:<15.2%}")
print("-" * 70)
print(f"{'Generalization Gap (R2)':<25} | {train_r2 - test_r2:<15.4f}")
print("="*70)

# 5. VISUALIZATIONS
# SHAP Interpretation
print("\nGenerating SHAP Driver Analysis...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# PLOT 1: Global Feature Importance (Bar Plot) 
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Global Feature Importance (SHAP)", fontsize=16)
plt.xlabel("Average Impact on Worthwhileness Rating (SHAP)", fontsize=12)
plt.tight_layout()
plt.show()

# PLOT 2: Summary Directionality Plot (Beeswarm/Dot Plot)
plt.figure(figsize=(12, 12))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title("SHAP Summary: Directional Impact on Worthwhileness", fontsize=16)
plt.tight_layout()
plt.show()

# 5B. Normalized Confusion Matrix
print("\nGenerating Confusion Matrix...")
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_rounded, cmap='Blues', normalize='true', ax=ax)
plt.title('Confusion Matrix xgb model')
plt.show()